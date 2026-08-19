//! C 티어 — 소프트(CPU) 합성 파이프라인 (INTEGRATION.md §1.5).
//!
//! 진입 조건: 합성 GPU 불가 (드라이버가 셰이더 컴파일에서 죽는 기기 등 —
//! wgpu를 아예 만들 수 없어 A/B 티어가 원천 불가). 추론은 ai-cpu(NEON),
//! 합성은 이 파일의 스칼라 루프 — GPU 심볼 의존 0.
//!
//! 효과 범위(설계 확정): 배경(단색/이미지)·블러·밝기/흑백(배경만). 윤곽
//! 정제(JBF/refine)·조명·터치업/메이크업·3D·프레이밍은 B 이상 — 여기선 마스크
//! bilinear 업샘플 + coverage smoothstep 근사만 쓴다 (웹 lite 등가).
//! mirror/degree 프레임 변환은 호스트 몫(ffi flip) — 이 파일과 무관.

use crate::error::TaskError;
use crate::session::cpu::CpuSession;
use super::params::{Background, EffectsState};

/// EMA 상수 — mask_ingest.rs 파리티 (diff>0.3이면 max, 아니면 min)
const EMA_DIFF: f32 = 0.3;
const EMA_ALPHA: (f32, f32) = (0.6, 0.9); // 알파 직출 (RVM pha)
const EMA_LOGITS: (f32, f32) = (0.03, 0.9); // 2ch 로짓

/// 배경 블러 — GPU BgBlur(bg_blur.wgsl) 등가: 1/5 해상도, 7탭 분리 가우시안
/// (오프셋 0..9 lo-px), h/v 교대 6패스, 인물 마스크 제외 가중(색 번짐 차단) +
/// 가중 부족분은 원본 색으로 채움. bilinear 업샘플.
const BLUR_SCALE: usize = 5;
const BLUR_PASSES: usize = 6;
const BLUR_OFFS: [f32; 7] = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0];
const BLUR_WTS: [f32; 7] = [0.25, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02];

pub struct SoftPipeline {
    pub state: EffectsState,
    seg_bytes: Option<Vec<u8>>,
    seg: Option<CpuSession>,
    /// (입력 w, 입력 h, 마스크 w, 마스크 h, 마스크 이름, 알파 직출 여부)
    io: Option<(usize, usize, usize, usize, String, bool)>,
    in_rgb: Vec<f32>,
    mask_raw: Vec<f32>,
    ema: Vec<f32>,
    ema_init: bool,
    /// 배경 이미지 원본 (RGBA, w, h)
    bg_src: Option<(Vec<u8>, usize, usize)>,
    /// 프레임 크기 cover 크롭 + 블러/밝기/흑백 베이크 캐시 — (픽셀, w, h, 키)
    bg_fit: Option<(Vec<u8>, usize, usize, u64)>,
    /// 블러 저해상 스크래치 (프레임 모드 블러용)
    blur_lo: Vec<[f32; 3]>,
    blur_tmp: Vec<[f32; 3]>,
    /// 블러 패스용 저해상 인물 마스크 (EMA 리샘플 — 프레임당 1회)
    mask_lo: Vec<f32>,
}

impl Default for SoftPipeline {
    fn default() -> Self {
        SoftPipeline {
            state: EffectsState::default(),
            seg_bytes: None,
            seg: None,
            io: None,
            in_rgb: Vec::new(),
            mask_raw: Vec::new(),
            ema: Vec::new(),
            ema_init: false,
            bg_src: None,
            bg_fit: None,
            blur_lo: Vec::new(),
            blur_tmp: Vec::new(),
            mask_lo: Vec::new(),
        }
    }
}

impl SoftPipeline {
    /// 세그 모델 바이트 주입 — 실제 로드는 첫 process에서 (지연)
    pub fn set_model(&mut self, bytes: Vec<u8>) {
        self.seg_bytes = Some(bytes);
        self.seg = None;
        self.io = None;
        self.ema_init = false;
    }

    pub fn apply_json(&mut self, json: &str) -> Result<(), String> {
        self.state.apply_json(json)
    }

    pub fn set_background_image(&mut self, rgba: &[u8], w: usize, h: usize) {
        self.bg_src = Some((rgba.to_vec(), w, h));
        self.bg_fit = None;
    }

    /// 켜진 효과가 하나도 없으면 프레임을 건드릴 필요가 없다
    pub fn passthrough(&self) -> bool {
        !self.state.any_active()
    }

    /// 프레임 → 모델 입력 리샘플 → CPU 추론 → mask_raw. (mw, mh, alpha_kind) 반환.
    fn run_infer(
        &mut self,
        rgba: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(usize, usize, bool), TaskError> {
        self.ensure_seg()?;
        let (iw, ih, mw, mh, mask_name, alpha_kind) = self.io.clone().unwrap();

        // 프레임 → 모델 입력 (bilinear 스트레치, 0..1 — 웹 CPU 티어와 동일 규약)
        self.in_rgb.resize(iw * ih * 3, 0.0);
        for y in 0..ih {
            let sy = (y as f32 + 0.5) * h as f32 / ih as f32 - 0.5;
            for x in 0..iw {
                let sx = (x as f32 + 0.5) * w as f32 / iw as f32 - 0.5;
                let px = bilinear_rgba(rgba, w, h, sx, sy);
                let o = (y * iw + x) * 3;
                self.in_rgb[o] = px[0] / 255.0;
                self.in_rgb[o + 1] = px[1] / 255.0;
                self.in_rgb[o + 2] = px[2] / 255.0;
            }
        }
        let seg = self.seg.as_mut().unwrap();
        seg.infer_frame(&self.in_rgb)?;
        seg.read_output_into(&mask_name, &mut self.mask_raw)?;
        Ok((mw, mh, alpha_kind))
    }

    /// B 티어용 CPU 추론 — **원시** 마스크(EMA 없음: 시간 상태는 GPU ingest 소유)를
    /// (mask, mw, mh, ch)로 반환. ch 1=알파 직출(pha), 2=로짓 [bg, person] —
    /// VideoPipeline 외부 마스크 주입(process_mask) 규약 그대로.
    pub fn infer_mask(
        &mut self,
        rgba: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(&[f32], u32, u32, u32), TaskError> {
        let (mw, mh, alpha_kind) = self.run_infer(rgba, w, h)?;
        let ch = if alpha_kind { 1 } else { 2 };
        Ok((&self.mask_raw, mw as u32, mh as u32, ch))
    }

    /// 프레임 1장 in-place 합성. 반환 false = 프레임 무수정 (passthrough).
    pub fn process(&mut self, rgba: &mut [u8], w: usize, h: usize) -> Result<bool, TaskError> {
        if self.passthrough() {
            return Ok(false);
        }
        let (mw, mh, alpha_kind) = self.run_infer(rgba, w, h)?;

        // 3) 시간 EMA (마스크 해상도 — mask_ingest 파리티)
        let (min_a, max_a) = if alpha_kind { EMA_ALPHA } else { EMA_LOGITS };
        self.ema.resize(mw * mh, 0.0);
        for i in 0..mw * mh {
            let cur = if alpha_kind {
                self.mask_raw[i].clamp(0.0, 1.0)
            } else {
                // NHWC 2ch [bg, person] → sigmoid(person-bg)
                let bg = self.mask_raw[i * 2];
                let person = self.mask_raw[i * 2 + 1];
                1.0 / (1.0 + (-(person - bg)).exp())
            };
            if !self.ema_init {
                self.ema[i] = cur;
            } else {
                let prev = self.ema[i];
                let a = if (cur - prev).abs() > EMA_DIFF { max_a } else { min_a };
                self.ema[i] = prev + a * (cur - prev);
            }
        }
        self.ema_init = true;

        // 4) 배경 소스 준비
        let d = self.state.derived();
        let (cov0, cov1) = (d.coverage[0], d.coverage[1]);
        let blur = self.state.blur.clamp(0.0, 1.0);
        let mode: u32 = match &self.state.background {
            Background::None if blur > 0.0 => 1,
            Background::None => 0,
            Background::Color(_) => 2,
            Background::Image => 3,
        };
        if mode == 1 {
            self.build_frame_blur(rgba, w, h, true);
        }
        if mode == 3 {
            self.ensure_bg_fit(w, h, blur);
        }
        let color_u8 = if let Background::Color(c) = &self.state.background {
            let mut c3 = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0];
            adjust_bg(&mut c3, self.state.brightness, self.state.grayscale);
            [c3[0] as u8, c3[1] as u8, c3[2] as u8]
        } else {
            [0, 0, 0]
        };
        let (bw, bh) = (w.div_ceil(BLUR_SCALE), h.div_ceil(BLUR_SCALE));
        let brightness = self.state.brightness;
        let grayscale = self.state.grayscale;

        // 5) 합성 — out = fg·m + bg·(1-m), 밝기/흑백은 배경만 (웹 규약)
        for y in 0..h {
            let mv = (y as f32 + 0.5) * mh as f32 / h as f32 - 0.5;
            for x in 0..w {
                let mu = (x as f32 + 0.5) * mw as f32 / w as f32 - 0.5;
                let m = smoothstep(cov0, cov1, bilinear_f32(&self.ema, mw, mh, mu, mv));
                if m >= 0.999 {
                    continue; // 완전 전경 — 원본 유지
                }
                let o = (y * w + x) * 4;
                let fg = [rgba[o] as f32, rgba[o + 1] as f32, rgba[o + 2] as f32];
                let mut bg = match mode {
                    1 => {
                        let lo = bilinear_rgb3(
                            &self.blur_lo,
                            bw,
                            bh,
                            (x as f32 + 0.5) / BLUR_SCALE as f32 - 0.5,
                            (y as f32 + 0.5) / BLUR_SCALE as f32 - 0.5,
                        );
                        [
                            fg[0] + (lo[0] - fg[0]) * blur,
                            fg[1] + (lo[1] - fg[1]) * blur,
                            fg[2] + (lo[2] - fg[2]) * blur,
                        ]
                    }
                    2 => [color_u8[0] as f32, color_u8[1] as f32, color_u8[2] as f32],
                    3 => {
                        let (fit, fw2, _, _) = self.bg_fit.as_ref().unwrap();
                        let o2 = (y * fw2 + x) * 4;
                        [fit[o2] as f32, fit[o2 + 1] as f32, fit[o2 + 2] as f32]
                    }
                    _ => fg,
                };
                // 단색(2)·이미지(3)는 베이크 시 보정 완료 — 원본/블러(0,1)만 여기서
                if mode < 2 {
                    adjust_bg(&mut bg, brightness, grayscale);
                }
                rgba[o] = (fg[0] * m + bg[0] * (1.0 - m)).clamp(0.0, 255.0) as u8;
                rgba[o + 1] = (fg[1] * m + bg[1] * (1.0 - m)).clamp(0.0, 255.0) as u8;
                rgba[o + 2] = (fg[2] * m + bg[2] * (1.0 - m)).clamp(0.0, 255.0) as u8;
            }
        }
        Ok(true)
    }

    fn ensure_seg(&mut self) -> Result<(), TaskError> {
        if self.seg.is_some() {
            return Ok(());
        }
        let bytes = self
            .seg_bytes
            .as_ref()
            .ok_or_else(|| TaskError::Other("세그 모델 미주입 (C 티어)".into()))?;
        let mut seg = CpuSession::load(bytes)?;
        // A24류 리틀코어 기기 전제 — 과점유 없이 4스레드 상한
        let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(2).min(4);
        let _ = seg.set_threads(n);
        let sw = seg.model().sw();
        let it = &sw.tensors[sw.inputs[0] as usize];
        // 마스크 출력 선택 — pipeline.rs mask_output 파리티 (c==1 우선, 다음 c==2)
        let mut outs: Vec<_> = sw
            .outputs
            .iter()
            .map(|&o| &sw.tensors[o as usize])
            .filter(|t| t.c == 1 || t.c == 2)
            .collect();
        outs.sort_by_key(|t| t.c);
        let mask = outs
            .first()
            .ok_or_else(|| TaskError::Other("세그 마스크 출력 없음 (C 티어)".into()))?;
        self.io = Some((
            it.w as usize,
            it.h as usize,
            mask.w as usize,
            mask.h as usize,
            mask.name.clone(),
            mask.c == 1,
        ));
        self.seg = Some(seg);
        self.ema_init = false;
        Ok(())
    }

    /// 프레임 블러 — GPU bg_blur.wgsl 등가 (헤더 상수 주석 참조).
    /// use_mask=true 면 인물 픽셀을 제외 가중해 배경 블러로 인물 색이 번지지
    /// 않는다 (프레임 모드). 이미지 배경 베이크는 false (정적 이미지 — 마스크 무관).
    fn build_frame_blur(&mut self, rgba: &[u8], w: usize, h: usize, use_mask: bool) {
        let (bw, bh) = (w.div_ceil(BLUR_SCALE), h.div_ceil(BLUR_SCALE));
        self.blur_lo.clear();
        self.blur_lo.resize(bw * bh, [0.0; 3]);
        self.blur_tmp.resize(bw * bh, [0.0; 3]);
        for by in 0..bh {
            for bx in 0..bw {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0f32;
                for dy in 0..BLUR_SCALE {
                    let y = by * BLUR_SCALE + dy;
                    if y >= h {
                        break;
                    }
                    for dx in 0..BLUR_SCALE {
                        let x = bx * BLUR_SCALE + dx;
                        if x >= w {
                            break;
                        }
                        let o = (y * w + x) * 4;
                        acc[0] += rgba[o] as f32;
                        acc[1] += rgba[o + 1] as f32;
                        acc[2] += rgba[o + 2] as f32;
                        n += 1.0;
                    }
                }
                self.blur_lo[by * bw + bx] = [acc[0] / n, acc[1] / n, acc[2] / n];
            }
        }
        // 저해상 인물 마스크 (EMA 리샘플) — use_mask=false 면 0 (제외 없음)
        self.mask_lo.clear();
        self.mask_lo.resize(bw * bh, 0.0);
        if use_mask && self.ema_init {
            if let Some((_, _, mw, mh, _, _)) = &self.io {
                let (mw, mh) = (*mw, *mh);
                for by in 0..bh {
                    let mv = ((by as f32 + 0.5) / bh as f32) * mh as f32 - 0.5;
                    for bx in 0..bw {
                        let mu = ((bx as f32 + 0.5) / bw as f32) * mw as f32 - 0.5;
                        self.mask_lo[by * bw + bx] = bilinear_f32(&self.ema, mw, mh, mu, mv);
                    }
                }
            }
        }
        for pass in 0..BLUR_PASSES {
            let horizontal = pass % 2 == 0;
            blur_pass_7tap(&self.blur_lo, &mut self.blur_tmp, &self.mask_lo, bw, bh, horizontal);
            std::mem::swap(&mut self.blur_lo, &mut self.blur_tmp);
        }
    }

    /// 이미지 배경 → 프레임 크기 cover 크롭 + (블러·밝기·흑백) 베이크 캐시
    fn ensure_bg_fit(&mut self, w: usize, h: usize, blur: f32) {
        let Some((src, sw, sh)) = &self.bg_src else {
            // 이미지 모드인데 업로드 없음 — 검정 배경 (조용한 실패보다 티 나게)
            let key = 0u64;
            if !matches!(&self.bg_fit, Some((_, fw, fh, k)) if *fw == w && *fh == h && *k == key) {
                self.bg_fit = Some((vec![0; w * h * 4], w, h, key));
            }
            return;
        };
        let key = 1
            ^ ((*sw as u64) << 1)
            ^ ((*sh as u64) << 17)
            ^ (((blur * 64.0) as u64) << 33)
            ^ (((self.state.brightness * 64.0) as u64) << 41)
            ^ (((self.state.grayscale * 64.0) as u64) << 49);
        if matches!(&self.bg_fit, Some((_, fw, fh, k)) if *fw == w && *fh == h && *k == key) {
            return;
        }
        // cover 크롭 bilinear
        let scale = (w as f32 / *sw as f32).max(h as f32 / *sh as f32);
        let mut fit = vec![0u8; w * h * 4];
        for y in 0..h {
            let sy = (y as f32 - h as f32 * 0.5) / scale + *sh as f32 * 0.5 - 0.5;
            for x in 0..w {
                let sx = (x as f32 - w as f32 * 0.5) / scale + *sw as f32 * 0.5 - 0.5;
                let px = bilinear_rgba(src, *sw, *sh, sx, sy);
                let mut c = [px[0], px[1], px[2]];
                adjust_bg(&mut c, self.state.brightness, self.state.grayscale);
                let o = (y * w + x) * 4;
                fit[o] = c[0] as u8;
                fit[o + 1] = c[1] as u8;
                fit[o + 2] = c[2] as u8;
                fit[o + 3] = 255;
            }
        }
        // 이미지 자체 블러 (compose.wgsl blur_bg_image 등가 근사) — 정적이라 1회 베이크
        if blur > 0.001 {
            let (bw, bh) = (w.div_ceil(BLUR_SCALE), h.div_ceil(BLUR_SCALE));
            self.build_frame_blur(&fit, w, h, false);
            for y in 0..h {
                for x in 0..w {
                    let lo = bilinear_rgb3(
                        &self.blur_lo,
                        bw,
                        bh,
                        (x as f32 + 0.5) / BLUR_SCALE as f32 - 0.5,
                        (y as f32 + 0.5) / BLUR_SCALE as f32 - 0.5,
                    );
                    let o = (y * w + x) * 4;
                    for c in 0..3 {
                        let v = fit[o + c] as f32;
                        fit[o + c] = (v + (lo[c] - v) * blur).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        self.bg_fit = Some((fit, w, h, key));
    }

    /// 스트림 파기 규약 — 시간 상태·타깃만 끊고 모델 바이트는 유지
    pub fn reset(&mut self) {
        self.seg = None;
        self.io = None;
        self.ema_init = false;
        self.bg_fit = None;
    }
}

/// 배경 전용 보정 — 밝기 스케일 + 흑백 mix (0..255 f32)
fn adjust_bg(c: &mut [f32; 3], brightness: f32, grayscale: f32) {
    let b = brightness.clamp(0.0, 2.0);
    let g = grayscale.clamp(0.0, 1.0);
    for v in c.iter_mut() {
        *v = (*v * b).clamp(0.0, 255.0);
    }
    if g > 0.0 {
        let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        for v in c.iter_mut() {
            *v += (y - *v) * g;
        }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bilinear_f32(buf: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as i64).clamp(0, w as i64 - 1) as usize;
    let y0 = (y.floor() as i64).clamp(0, h as i64 - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
    let a = buf[y0 * w + x0] + (buf[y0 * w + x1] - buf[y0 * w + x0]) * fx;
    let b = buf[y1 * w + x0] + (buf[y1 * w + x1] - buf[y1 * w + x0]) * fx;
    a + (b - a) * fy
}

fn bilinear_rgba(buf: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let x0 = (x.floor() as i64).clamp(0, w as i64 - 1) as usize;
    let y0 = (y.floor() as i64).clamp(0, h as i64 - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let p00 = buf[(y0 * w + x0) * 4 + c] as f32;
        let p10 = buf[(y0 * w + x1) * 4 + c] as f32;
        let p01 = buf[(y1 * w + x0) * 4 + c] as f32;
        let p11 = buf[(y1 * w + x1) * 4 + c] as f32;
        let a = p00 + (p10 - p00) * fx;
        let b = p01 + (p11 - p01) * fx;
        out[c] = a + (b - a) * fy;
    }
    out
}

fn bilinear_rgb3(buf: &[[f32; 3]], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let x0 = (x.floor() as i64).clamp(0, w as i64 - 1) as usize;
    let y0 = (y.floor() as i64).clamp(0, h as i64 - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let a = buf[y0 * w + x0][c] + (buf[y0 * w + x1][c] - buf[y0 * w + x0][c]) * fx;
        let b = buf[y1 * w + x0][c] + (buf[y1 * w + x1][c] - buf[y1 * w + x0][c]) * fx;
        out[c] = a + (b - a) * fy;
    }
    out
}

/// 7탭 분리 가우시안 1패스 — bg_blur.wgsl 등가. 분수 오프셋(1.5 간격)은 축 방향
/// 1D lerp, 경계 clamp. 인물 마스크 가중으로 인물 색을 배경 블러에서 제외하고
/// 가중 부족분(acc_a<1)은 원본 색으로 채운다.
fn blur_pass_7tap(
    src: &[[f32; 3]],
    dst: &mut [[f32; 3]],
    mask: &[f32],
    w: usize,
    h: usize,
    horizontal: bool,
) {
    // 축 방향 분수 위치 샘플 — (색, 인물 마스크)
    let sample = |fx: f32, fy: f32| -> ([f32; 3], f32) {
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let xa = (x0 as i64).clamp(0, w as i64 - 1) as usize;
        let ya = (y0 as i64).clamp(0, h as i64 - 1) as usize;
        let xb = (xa + 1).min(w - 1);
        let yb = (ya + 1).min(h - 1);
        // 오프셋은 축 정렬 — 다른 축 t는 0이라 1D lerp 로 충분
        let (a, b, t) = if horizontal {
            (ya * w + xa, ya * w + xb, tx)
        } else {
            (ya * w + xa, yb * w + xa, ty)
        };
        let c = [
            src[a][0] + (src[b][0] - src[a][0]) * t,
            src[a][1] + (src[b][1] - src[a][1]) * t,
            src[a][2] + (src[b][2] - src[a][2]) * t,
        ];
        let m = mask[a] + (mask[b] - mask[a]) * t;
        (c, m)
    };
    for y in 0..h {
        for x in 0..w {
            let center = src[y * w + x];
            let pm0 = mask[y * w + x];
            let w0 = BLUR_WTS[0] * (1.0 - pm0);
            let mut acc = [center[0] * w0, center[1] * w0, center[2] * w0];
            let mut acc_a = w0;
            for i in 1..7 {
                let off = BLUR_OFFS[i];
                let (dx, dy) = if horizontal { (off, 0.0) } else { (0.0, off) };
                for sign in [1.0f32, -1.0] {
                    let (c, m) =
                        sample(x as f32 + dx * sign, y as f32 + dy * sign);
                    let wt = BLUR_WTS[i] * (1.0 - m);
                    acc[0] += c[0] * wt;
                    acc[1] += c[1] * wt;
                    acc[2] += c[2] * wt;
                    acc_a += wt;
                }
            }
            // 셰이더 등가 그대로 — 가중치 합이 1.41이라 fill이 **음수**가 되며
            // 초과분을 center로 빼서 합 1.0으로 정규화한다. 클램프 금지
            // (min(1.0) 클램프 = 픽셀 1.41배 → 밝기 뜸 — 실기 확인된 함정).
            let fill = 1.0 - acc_a;
            dst[y * w + x] = [
                acc[0] + center[0] * fill,
                acc[1] + center[1] * fill,
                acc[2] + center[2] * fill,
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 모델 없이 합성 수학만 검증하려면 process가 세그를 요구하므로, 여기서는
    // 마스크·배경 헬퍼의 순수 함수를 게이트한다. 실모델 e2e는 tests/vb_soft.rs.
    #[test]
    fn smoothstep_and_bilinear() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        let buf = [0.0f32, 1.0, 0.0, 1.0];
        assert!((bilinear_f32(&buf, 2, 2, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blur_preserves_energy() {
        // 균일 입력은 블러 후에도 그대로여야 한다 (가중치 합 1.41 + 음수 fill 정규화
        // 검증 — 클램프 버그면 1.41배로 밝아진다). 마스크 0/1 혼합도 동일.
        let (w, h) = (16usize, 8usize);
        let src = vec![[100.0f32, 150.0, 200.0]; w * h];
        let mut dst = vec![[0.0f32; 3]; w * h];
        for (mi, mask) in [vec![0.0f32; w * h], vec![0.7f32; w * h]].into_iter().enumerate() {
            blur_pass_7tap(&src, &mut dst, &mask, w, h, true);
            for (i, px) in dst.iter().enumerate() {
                for c in 0..3 {
                    assert!(
                        (px[c] - src[i][c]).abs() < 0.01,
                        "균일장 불보존 (mask셋 {mi}, px {i} ch{c}): {} → {}",
                        src[i][c],
                        px[c]
                    );
                }
            }
        }
    }

    #[test]
    fn adjust_bg_math() {
        let mut c = [100.0, 100.0, 100.0];
        adjust_bg(&mut c, 2.0, 0.0);
        assert_eq!(c, [200.0, 200.0, 200.0]);
        let mut c = [255.0, 0.0, 0.0];
        adjust_bg(&mut c, 1.0, 1.0);
        // 완전 흑백 — R=G=B(휘도)
        assert!((c[0] - c[1]).abs() < 1e-4 && (c[1] - c[2]).abs() < 1e-4);
    }
}
