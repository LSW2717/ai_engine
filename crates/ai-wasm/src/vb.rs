//! vb — **VbEngine 심**(v-ai 4번째 티어)의 wasm 바인딩.
//!
//! 오케스트레이션은 전부 `ai_tasks::Director` — 여기 남는 것은 플랫폼 몫뿐:
//! OffscreenCanvas 서피스(워커엔 DOM 캔버스가 없다), ImageBitmap 프레임 임포트,
//! JS 타입 변환. 3함수 계약(config/destroy/processWorkerFrame)의 JS쪽 절반은
//! `web/vb-engine.js` — 그 파일이 이 익스포트들을 v-ai pipeline.worker 규약으로
//! 감싼다 (passthrough 제로카피·입력 close 순서·VBOptions 단위 변환).
//!
//! 프레임 계약: 호스트(심)가 ①`vb_passthrough()`면 아무것도 안 부르고
//! ②`vb_needs_render()`면 `vb_frame`(합성 → 서피스 캔버스 →
//! transferToImageBitmap) ③아니고 태스크만 켜져 있으면 `vb_analyze`(무합성).
//! `vb_wants_pixels(t)`가 참인 틱에만 u8 RGB를 뽑아 넘긴다 (getImageData 절약).

use std::cell::{Cell, RefCell};

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::Director;
use wasm_bindgen::prelude::*;

use crate::{engine, js_err};

struct VbSurface {
    canvas: web_sys::OffscreenCanvas,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

struct Vb {
    director: Director,
    surface: VbSurface,
}

/// 프레임 비행 중(vb_frame/analyze/frame_mask가 VB를 take한 동안) 도착한 조작.
/// ⚠ 실사고(2026-08-15): 큐 없이는 fetch 콜백/config가 비행 창에 떨어지면
/// "vb_attach 먼저"로 조용히 튕기고, 호스트 fetched-가드 때문에 재시도도 없어
/// 모델/GLB 미주입·설정 유실("클릭이 안 먹힘")이 됐다.
enum Pending {
    Config(String),
    Model(String, Vec<u8>),
    Glb(String, Vec<u8>),
    BgImage(Vec<u8>, u32, u32),
    Layout(String),
    Detach,
}

thread_local! {
    static VB: RefCell<Option<Vb>> = const { RefCell::new(None) };
    static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
    static ATTACHED: Cell<bool> = const { Cell::new(false) };
    // (passthrough, needs_render) 캐시 — 비행 중 판정 질의는 이 값으로 답한다
    static JUDGE: Cell<(bool, bool)> = const { Cell::new((true, false)) };
    static LAST_FOCUS: RefCell<String> = const { RefCell::new(String::new()) };
}

fn apply_op(ctx: &GpuContext, vb: &mut Vb, op: Pending) -> Result<(), JsValue> {
    match op {
        Pending::Config(json) => vb.director.apply_json(&json).map_err(js_err),
        Pending::Model(kind, bytes) => vb.director.set_model(&kind, bytes).map_err(js_err),
        Pending::Glb(kind, bytes) => vb.director.set_item_glb(ctx, &kind, &bytes).map_err(js_err),
        Pending::BgImage(rgba, w, h) => {
            vb.director.set_background_image(ctx, &rgba, w, h);
            Ok(())
        }
        Pending::Layout(json) => vb.director.set_focus_layout_json(&json).map_err(js_err),
        Pending::Detach => {
            vb.director.detach();
            Ok(())
        }
    }
}

/// 조작 제출 — VB가 자리에 있으면 즉시 적용, 프레임 비행 중이면 큐잉(반납 직후
/// drain_pending이 적용). 미attach만 에러.
fn submit(op: Pending) -> Result<(), JsValue> {
    let ctx = engine()?;
    let r = VB.with(|c| {
        let mut b = c.borrow_mut();
        match b.as_mut() {
            Some(vb) => apply_op(&ctx, vb, op).map(|_| true),
            None if ATTACHED.with(|a| a.get()) => {
                PENDING.with(|p| p.borrow_mut().push(op));
                Ok(false)
            }
            None => Err(JsValue::from_str("vb_attach 먼저")),
        }
    });
    if let Ok(true) = r {
        refresh_judge();
    }
    r.map(|_| ())
}

/// 비행 중 쌓인 조작 적용 — 프레임 함수가 VB를 반납한 직후 호출한다.
/// 개별 실패는 로그만 (한 건 실패가 다음 조작·프레임을 못 막게).
fn drain_pending(ctx: &GpuContext) {
    let ops = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    if ops.is_empty() {
        refresh_judge();
        return;
    }
    VB.with(|c| {
        let mut b = c.borrow_mut();
        if let Some(vb) = b.as_mut() {
            for op in ops {
                if let Err(e) = apply_op(&ctx, vb, op) {
                    log::warn!("[vb] 지연 적용 실패: {e:?}");
                }
            }
        }
    });
    refresh_judge();
}

/// 판정·집중도 캐시 갱신 (VB가 자리에 있을 때만 — 비행 중엔 no-op)
fn refresh_judge() {
    VB.with(|c| {
        if let Some(vb) = c.borrow_mut().as_mut() {
            JUDGE.with(|j| j.set((vb.director.passthrough(), vb.director.needs_render())));
            LAST_FOCUS.with(|f| *f.borrow_mut() = vb.director.focus_json());
        }
    });
}

/// 워커 서피스 연결 — OffscreenCanvas(워커 로컬)에 WebGPU 서피스 + Director 생성.
/// 재호출 = 서피스 재구성 (Director는 유지 — 웜).
#[wasm_bindgen]
pub fn vb_attach(canvas: web_sys::OffscreenCanvas) -> Result<(), JsValue> {
    let ctx = engine()?;
    let (w, h) = (canvas.width().max(1), canvas.height().max(1));
    let surface = ctx
        .instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone()))
        .map_err(|e| js_err(format!("서피스 생성: {e:?}")))?;
    let caps = surface.get_capabilities(&ctx.adapter);
    let format = caps.formats[0];
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: w,
        height: h,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        color_space: Default::default(),
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&ctx.device, &config);
    VB.with(|c| {
        let mut b = c.borrow_mut();
        match b.as_mut() {
            Some(vb) => vb.surface = VbSurface { canvas, surface, config },
            None => {
                *b = Some(Vb {
                    director: Director::new(&ctx, format),
                    surface: VbSurface { canvas, surface, config },
                })
            }
        }
    });
    ATTACHED.with(|a| a.set(true));
    refresh_judge();
    Ok(())
}

/// 단일 JSON 설정 (EffectsPatch + faceItems/handDetection/focusDetection —
/// 없음=유지/null=해제/값=설정). 프레임 비행 중이면 큐잉 후 반납 직후 적용.
#[wasm_bindgen]
pub fn vb_config(json: String) -> Result<(), JsValue> {
    submit(Pending::Config(json))
}

/// 모델 바이트 주입 — kind: "seg"|"face_det"|"face_lm"|"gaze"|"gaze_bs"|
/// "hand_det"|"hand_lm" (조달은 호스트 fetch). 비행 중 큐잉.
#[wasm_bindgen]
pub fn vb_model(kind: String, bytes: Vec<u8>) -> Result<(), JsValue> {
    submit(Pending::Model(kind, bytes))
}

/// GLB 바이트 주입 (종류당 1회). 비행 중 큐잉.
#[wasm_bindgen]
pub fn vb_glb(kind: String, bytes: Vec<u8>) -> Result<(), JsValue> {
    submit(Pending::Glb(kind, bytes))
}

/// 배경 이미지 (RGBA8) — background:"image"가 소비. 비행 중 큐잉.
#[wasm_bindgen]
pub fn vb_bg_image(rgba: &[u8], w: u32, h: u32) -> Result<(), JsValue> {
    submit(Pending::BgImage(rgba.to_vec(), w, h))
}

/// 다중 모니터 레이아웃 JSON (focusDetection 켠 뒤). 비행 중 큐잉.
#[wasm_bindgen]
pub fn vb_layout(json: String) -> Result<(), JsValue> {
    submit(Pending::Layout(json))
}

#[wasm_bindgen]
pub fn vb_passthrough() -> Result<bool, JsValue> {
    VB.with(|c| match c.borrow_mut().as_mut() {
        Some(vb) => Ok(vb.director.passthrough()),
        None if ATTACHED.with(|a| a.get()) => Ok(JUDGE.with(|j| j.get()).0),
        None => Err(JsValue::from_str("vb_attach 먼저")),
    })
}

#[wasm_bindgen]
pub fn vb_needs_render() -> Result<bool, JsValue> {
    VB.with(|c| match c.borrow_mut().as_mut() {
        Some(vb) => Ok(vb.director.needs_render()),
        None if ATTACHED.with(|a| a.get()) => Ok(JUDGE.with(|j| j.get()).1),
        None => Err(JsValue::from_str("vb_attach 먼저")),
    })
}

#[wasm_bindgen]
pub fn vb_wants_pixels(t_ms: f64) -> Result<bool, JsValue> {
    VB.with(|c| match c.borrow_mut().as_mut() {
        Some(vb) => Ok(vb.director.wants_pixels(t_ms)),
        // 비행 중 — 이번 틱은 픽셀 생략 (다음 프레임에 재판정)
        None if ATTACHED.with(|a| a.get()) => Ok(false),
        None => Err(JsValue::from_str("vb_attach 먼저")),
    })
}

/// ImageBitmap → 파이프라인 프레임 텍스처 (무복사 임포트)
fn import_bitmap(ctx: &GpuContext, tex: &wgpu::Texture, bmp: &web_sys::ImageBitmap) {
    let (fw, fh) = (bmp.width().max(1), bmp.height().max(1));
    ctx.queue.copy_external_image_to_texture(
        &wgpu::wgt::CopyExternalImageSourceInfo {
            source: wgpu::wgt::ExternalImageSource::ImageBitmap(bmp.clone()),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        },
        wgpu::wgt::CopyExternalImageDestInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
            color_space: wgpu::wgt::PredefinedColorSpace::Srgb,
            premultiplied_alpha: false,
        },
        wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
    );
}

/// 렌더 프레임 (A티어): ImageBitmap → 추론+스택+아이템 → 서피스 캔버스.
/// 호출 후 JS가 canvas.transferToImageBitmap()으로 결과를 떼 간다.
/// rgb = 원본 u8 RGB (vb_wants_pixels 틱에만 — 손/집중도/광원 프로브 소비).
#[wasm_bindgen]
pub async fn vb_frame(
    source: web_sys::ImageBitmap,
    rgb: Option<Vec<u8>>,
    t_ms: f64,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let mut vb = VB.with(|c| c.borrow_mut().take());
    let Some(v) = vb.as_mut() else {
        return Err(JsValue::from_str("vb_attach 먼저"));
    };
    let result = vb_frame_inner(&ctx, v, &source, rgb.as_deref(), t_ms).await;
    VB.with(|c| *c.borrow_mut() = vb);
    drain_pending(&ctx); // 비행 중 큐잉된 config/모델/GLB/배경 적용
    result
}

async fn vb_frame_inner(
    ctx: &GpuContext,
    vb: &mut Vb,
    source: &web_sys::ImageBitmap,
    rgb: Option<&[u8]>,
    t_ms: f64,
) -> Result<(), JsValue> {
    let (fw, fh) = (source.width().max(1), source.height().max(1));
    // 서피스 = 출력 캔버스 — 프레임 크기에 동기 (transferToImageBitmap 규약)
    if vb.surface.config.width != fw || vb.surface.config.height != fh {
        vb.surface.canvas.set_width(fw);
        vb.surface.canvas.set_height(fh);
        vb.surface.config.width = fw;
        vb.surface.config.height = fh;
        vb.surface.surface.configure(&ctx.device, &vb.surface.config);
    }
    vb.director
        .with_frame(ctx, fw, fh, |tex| import_bitmap(ctx, tex, source))
        .await
        .map_err(js_err)?;
    let frame = match vb.surface.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        other => {
            log::warn!("[vb] 서피스 {other:?} — 재구성 후 스킵");
            vb.surface.surface.configure(&ctx.device, &vb.surface.config);
            return Err(JsValue::from_str("서피스 재구성 — 프레임 스킵"));
        }
    };
    let view = frame.texture.create_view(&Default::default());
    vb.director
        .frame(ctx, fw, fh, Some(&view), rgb, t_ms)
        .await
        .map_err(js_err)?;
    drop(frame); // present
    Ok(())
}

/// 분석 프레임 (analyzer-only — 손/집중도만): 합성·서피스 없이 태스크만.
/// 심은 입력 비트맵을 passthrough로 그대로 돌려준다.
#[wasm_bindgen]
pub async fn vb_analyze(
    source: web_sys::ImageBitmap,
    rgb: Option<Vec<u8>>,
    t_ms: f64,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let mut vb = VB.with(|c| c.borrow_mut().take());
    let Some(v) = vb.as_mut() else {
        return Err(JsValue::from_str("vb_attach 먼저"));
    };
    let (fw, fh) = (source.width().max(1), source.height().max(1));
    let result = async {
        v.director
            .with_frame(&ctx, fw, fh, |tex| import_bitmap(&ctx, tex, &source))
            .await
            .map_err(js_err)?;
        v.director
            .frame(&ctx, fw, fh, None, rgb.as_deref(), t_ms)
            .await
            .map_err(js_err)
    }
    .await;
    VB.with(|c| *c.borrow_mut() = vb);
    drain_pending(&ctx);
    result
}

/// B티어 렌더 프레임: 외부(ai-cpu) 마스크 주입 — 세그 모델·세션 불필요
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn vb_frame_mask(
    source: web_sys::ImageBitmap,
    mask: Vec<f32>,
    ch: u32,
    mask_w: u32,
    mask_h: u32,
    rgb: Option<Vec<u8>>,
    t_ms: f64,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let mut vb = VB.with(|c| c.borrow_mut().take());
    let Some(v) = vb.as_mut() else {
        return Err(JsValue::from_str("vb_attach 먼저"));
    };
    let (fw, fh) = (source.width().max(1), source.height().max(1));
    let result = async {
        if v.surface.config.width != fw || v.surface.config.height != fh {
            v.surface.canvas.set_width(fw);
            v.surface.canvas.set_height(fh);
            v.surface.config.width = fw;
            v.surface.config.height = fh;
            v.surface.surface.configure(&ctx.device, &v.surface.config);
        }
        v.director
            .with_frame_mask(&ctx, fw, fh, (mask_w, mask_h), |tex| {
                import_bitmap(&ctx, tex, &source)
            })
            .map_err(js_err)?;
        let frame = match v.surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                log::warn!("[vb] 서피스 {other:?} — 재구성 후 스킵");
                v.surface.surface.configure(&ctx.device, &v.surface.config);
                return Err(JsValue::from_str("서피스 재구성 — 프레임 스킵"));
            }
        };
        let view = frame.texture.create_view(&Default::default());
        v.director
            .frame_mask(&ctx, &mask, ch, mask_w, mask_h, fw, fh, &view, rgb.as_deref(), t_ms)
            .await
            .map_err(js_err)?;
        drop(frame);
        Ok(())
    }
    .await;
    VB.with(|c| *c.borrow_mut() = vb);
    drain_pending(&ctx);
    result
}

/// 마지막 집중도 JSON (FocusResult 7상태 전체) — 비행 중엔 캐시 반환
#[wasm_bindgen]
pub fn vb_focus_state() -> Result<String, JsValue> {
    VB.with(|c| match c.borrow_mut().as_mut() {
        Some(vb) => Ok(vb.director.focus_json()),
        None if ATTACHED.with(|a| a.get()) => Ok(LAST_FOCUS.with(|f| f.borrow().clone())),
        None => Err(JsValue::from_str("vb_attach 먼저")),
    })
}

/// 제스처 이벤트 하나 (FIFO 16) — 없으면 null. 비행 중엔 null (다음 폴에 잡힌다 —
/// 이벤트는 엔진 큐에 남아 유실 없음)
#[wasm_bindgen]
pub fn vb_poll_gesture() -> Result<Option<String>, JsValue> {
    VB.with(|c| match c.borrow_mut().as_mut() {
        Some(vb) => Ok(vb.director.poll_gesture_json()),
        None if ATTACHED.with(|a| a.get()) => Ok(None),
        None => Err(JsValue::from_str("vb_attach 먼저")),
    })
}

/// destroy 대응 — **웜 리셋**: 스트림 리소스·시간 상태만 버리고 세션·모델·
/// 컴파일 결과는 유지 (v-ai destroy 규약 — 재활성화가 즉시 뜬다). 비행 중 큐잉.
#[wasm_bindgen]
pub fn vb_detach() -> Result<(), JsValue> {
    submit(Pending::Detach)
}
