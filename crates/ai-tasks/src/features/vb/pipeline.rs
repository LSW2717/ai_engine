//! (본체는 아래 — 모듈 목차는 vb/mod.rs)
//! VideoPipeline — 가상배경 비디오 파이프라인 (INTEGRATION.md D1의 Pipeline층).
//!
//! 프레임 한 장의 GPU 상주 경로 (CPU 픽셀 왕복 **0**, 리드백 **0**):
//!   프레임 텍스처 → [preprocess 컴퓨트] → 모델 입력 버퍼 → 추론(GpuSession)
//!   → 출력 버퍼 → [mask_ingest: softmax/pha + 시간EMA(핑퐁)] → r8 마스크
//!   → [mask_upsample: joint bilateral] → 프레임 해상도 마스크
//!   → [bg_blur ×6 @1/5 해상도] → [compose: 배경 4모드+coverage+spill+edge] → 타깃
//!
//! 프레임당 CPU→GPU 트래픽 = uniform 몇십 바이트가 전부. 새 스테이지 추가 절차는
//! stage.rs 헤더, 파라미터는 EffectsPatch JSON 머지(params.rs). 시간 상태(EMA
//! history)는 파이프라인이 핑퐁으로 소유한다.
//!
//! 전제: VideoPipeline 하나 = 세그 세션 하나 (모델 입출력 버퍼가 바인드그룹에
//! 박힌다). 세션을 갈아끼우면 `invalidate()`.


use ai_gpu::wgpu;
use ai_gpu::GpuContext;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use crate::error::TaskError;
use crate::session::gpu::GpuSession;
use super::framing::{BBox, Framing};
pub use super::params::{Background, EffectsPatch, EffectsState};
use super::stages::tex2d;
use super::stages::bbox::BboxStage;
use super::stages::bg_blur::{BgBlur, BLUR_ITERS, BLUR_SCALE};
use super::stages::compose::Compose;
use super::stages::mask_ingest::{MaskIngest, MaskKind};
use super::stages::mask_refine::MaskRefine;
use super::stages::mask_upsample::MaskUpsample;
use super::stages::preprocess::Preprocess;

/// 해상도·세션 종속 리소스 — 바인드그룹은 전부 여기서 만들어 프레임 루프에서
/// 재생성 0 (매 프레임 만들면 분당 수천 개 — Compositor에서 실측한 함정)
struct Res {
    fw: u32,
    fh: u32,
    frame_tex: wgpu::Texture,
    staging: wgpu::Texture,
    mask_lo: Vec<wgpu::TextureView>, // [2] EMA 핑퐁
    mask_hi: wgpu::TextureView,
    refine_v: Vec<wgpu::TextureView>, // [3] tmp/blurred/refined — compose는 [2]를 소비
    blur_v: Vec<wgpu::TextureView>, // [2]
    bw: u32,
    bh: u32,
    pre_bind: wgpu::BindGroup,
    ingest_bind: Vec<wgpu::BindGroup>, // [parity]
    up_bind: Vec<wgpu::BindGroup>,
    refine_binds: Vec<wgpu::BindGroup>, // [3] h/v/refine
    blur_binds: Vec<(wgpu::BindGroup, usize)>, // (bind, 타깃 인덱스)
    comp_bind: wgpu::BindGroup,
    mask_bytes_per_row: u32,
    mw: u32,
    mh: u32,
    out_cg: u32,
    out_name: String,
    /// RVM 전경색 출력 (c==3) — 있으면 매팅 합성에 사용
    fgr: Option<(String, wgpu::Texture, u32)>, // (이름, 스테이징, bytes_per_row)
    mask_kind_alpha: bool,
    in_f16: bool,
    in_w: u32,
    in_h: u32,
    in_cg: u32,
    /// 외부 마스크 주입(픽셀 diff 게이트 · P2 B티어) 스테이징 — (ch, 텍스처,
    /// 인제스트 바인드 핑퐁). 지연 생성 (주입 경로를 안 쓰면 비용 0)
    ext_mask: Option<(u32, wgpu::Texture, Vec<wgpu::BindGroup>)>,
    /// bbox 리덕션 바인드 (mask_lo[p] — EMA 이후 마스크 소비) [parity]
    bbox_bind: Vec<wgpu::BindGroup>,
}

pub struct VideoPipeline {
    pub state: EffectsState,
    pre: Preprocess,
    ingest: MaskIngest,
    up: MaskUpsample,
    refine: MaskRefine,
    blur: BgBlur,
    comp: Compose,
    bbox: BboxStage,
    framing: Framing,
    /// 최신 bbox 스캔 결과 — 리드백 사이엔 직전 값 유지 (v-ai 스냅샷 재스캔 등가)
    last_bbox: Option<BBox>,
    /// 게이트/디버그: 크롭 강제 고정 (스무딩 우회)
    framing_override: Option<(f32, f32, f32)>,
    epoch: Instant,
    sampler: wgpu::Sampler,
    /// v-ai 파리티: 저해상 마스크(segmentationTexture)·JBF 출력(personMask)은
    /// NEAREST다 — 마스크를 최근접으로 읽는 패스가 이 샘플러를 쓴다
    sampler_near: wgpu::Sampler,
    res: Option<Res>,
    bg_img: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    bg_dirty: bool,
    parity: usize,
}

impl VideoPipeline {
    /// target_format = 최종 서피스/텍스처 포맷
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        VideoPipeline {
            state: EffectsState::default(),
            pre: Preprocess::new(ctx),
            ingest: MaskIngest::new(ctx),
            up: MaskUpsample::new(ctx),
            refine: MaskRefine::new(ctx),
            blur: BgBlur::new(ctx),
            comp: Compose::new(ctx, target_format),
            bbox: BboxStage::new(ctx),
            framing: Framing::default(),
            last_bbox: None,
            framing_override: None,
            epoch: Instant::now(),
            sampler: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("video"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            sampler_near: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("video-near"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            res: None,
            bg_img: None,
            bg_dirty: false,
            parity: 0,
        }
    }

    /// EffectsPatch JSON 적용 (없음=유지 / null=해제 / 값=설정)
    pub fn apply_json(&mut self, json: &str) -> Result<(), String> {
        self.state.apply_json(json)
    }

    /// 배경 이미지 업로드 (RGBA8) — "image" 배경 모드가 이 텍스처를 쓴다
    pub fn set_background_image(&mut self, ctx: &GpuContext, rgba: &[u8], w: u32, h: u32) {
        let (tex, view) = tex2d(
            ctx,
            "video-bg-image",
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_DST,
        );
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.bg_img = Some((tex, view, w, h));
        self.bg_dirty = true; // comp_bind가 이 뷰를 물고 있다 — 재생성 필요
    }

    /// 카메라 프레임 텍스처 확보 — 채우기(외부 이미지 임포트 등)는 호출자 몫
    pub fn with_frame_texture<R>(
        &mut self,
        ctx: &GpuContext,
        seg: &GpuSession,
        w: u32,
        h: u32,
        f: impl FnOnce(&wgpu::Texture) -> R,
    ) -> Result<R, TaskError> {
        self.ensure(ctx, seg, w, h)?;
        Ok(f(&self.res.as_ref().unwrap().frame_tex))
    }

    /// 매팅 합성(fgr) 사용 중인가 — 진단·게이트용
    pub fn uses_fgr(&self) -> bool {
        self.res.as_ref().map(|r| r.fgr.is_some()).unwrap_or(false)
    }

    /// 세션 교체 시 리소스 폐기 (모델 버퍼가 바인드그룹에 박혀 있다).
    /// ⚠ 프레이밍 스무딩 상태는 유지 — v-ai 규율: 리셋하면 옵션 조작마다 줌이
    /// 1x로 튕겼다 재수렴한다("띡띡"). 리셋은 스트림 파기에서만.
    pub fn invalidate(&mut self) {
        self.res = None;
    }

    /// 게이트/디버그: 크롭 강제 고정 (bbox·스무딩 우회). None = 정상 경로
    pub fn set_framing_override(&mut self, v: Option<(f32, f32, f32)>) {
        self.framing_override = v;
    }

    /// 현재 프레이밍 크롭 (scale, cx, cy) — HUD·테스트
    pub fn framing_current(&self) -> (f32, f32, f32) {
        self.framing.current()
    }

    /// 최신 bbox 스캔 결과 — 테스트·진단 (None = 인물 없음/미도착)
    pub fn last_bbox(&self) -> Option<BBox> {
        self.last_bbox
    }

    /// 프레임 시작 프레이밍 틱: 리드백 회수 → 스무딩 → 이번 프레임 발행 슬롯.
    /// 프레이밍 off면 발행 없음 (비용 0)
    fn framing_tick(&mut self, ctx: &GpuContext) -> Option<usize> {
        let (mw, mh) = {
            let r = self.res.as_ref().unwrap();
            (r.mw, r.mh)
        };
        if let Some(scan) = self.bbox.pump(mw, mh) {
            self.last_bbox = scan;
        }
        let now = self.epoch.elapsed().as_secs_f64() * 1e3;
        self.framing.update(self.state.framing.as_ref(), self.last_bbox, now);
        if self.state.framing.is_some() {
            self.bbox.prepare(ctx)
        } else {
            None
        }
    }

    fn ensure(
        &mut self,
        ctx: &GpuContext,
        seg: &GpuSession,
        fw: u32,
        fh: u32,
    ) -> Result<(), TaskError> {
        if !self.bg_dirty {
            if let Some(r) = &self.res {
                if r.fw == fw && r.fh == fh {
                    return Ok(());
                }
            }
        }
        let model = seg.model();
        let in_name = model.sw.tensors[model.sw.inputs[0] as usize].name.clone();
        let (out_name, out_desc) = mask_output(model)?;
        let fgr_out = fgr_output(model, &out_name);
        let (in_buf, in_desc) = model
            .input_storage(&in_name)
            .ok_or_else(|| TaskError::Other("세그 입력 버퍼 없음".into()))?;
        let (mw, mh, out_cg) = (out_desc.w, out_desc.h, out_desc.cg());
        // dtype 인지: fp16 모델이면 레인이 8B (rgba16float 스테이징)
        let texel_bytes = out_desc.dt.vec4_bytes() as u32;
        let mask_bytes_per_row = mw * out_cg * texel_bytes;
        if mask_bytes_per_row % 256 != 0 {
            return Err(TaskError::Other(format!(
                "마스크 행 {mask_bytes_per_row}B가 256 정렬 아님 (W={mw} cg={out_cg})"
            )));
        }
        let (frame_tex, frame_view) = tex2d(
            ctx,
            "video-frame",
            fw,
            fh,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (staging, staging_view) = tex2d(
            ctx,
            "video-mask-staging",
            mw * out_cg,
            mh,
            if texel_bytes == 8 {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::Rgba32Float
            },
            wgpu::TextureUsages::COPY_DST,
        );
        let mask_lo: Vec<wgpu::TextureView> = (0..2)
            .map(|i| {
                tex2d(
                    ctx,
                    &format!("video-mask-lo{i}"),
                    mw,
                    mh,
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
                .1
            })
            .collect();
        let mask_hi = tex2d(
            ctx,
            "video-mask-hi",
            fw,
            fh,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .1;
        let refine_v: Vec<wgpu::TextureView> = (0..3)
            .map(|i| {
                tex2d(
                    ctx,
                    &format!("vb-refine{i}"),
                    fw,
                    fh,
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
                .1
            })
            .collect();
        let (bw, bh) = ((fw / BLUR_SCALE).max(1), (fh / BLUR_SCALE).max(1));
        let blur_v: Vec<wgpu::TextureView> = (0..2)
            .map(|i| {
                tex2d(
                    ctx,
                    &format!("video-blur{i}"),
                    bw,
                    bh,
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
                .1
            })
            .collect();

        use wgpu::BindGroupEntry as E;
        use wgpu::BindingResource::{Sampler as BSampler, TextureView as BTex};
        let bind = |label: &str, layout, entries: &[E]| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
        };
        let pre_bind = bind(
            "video-pre",
            &self.pre.bgl,
            &[
                E { binding: 0, resource: BTex(&frame_view) },
                E { binding: 1, resource: BSampler(&self.sampler) },
                E { binding: 2, resource: in_buf.as_entire_binding() },
                E { binding: 3, resource: self.pre.params.as_entire_binding() },
            ],
        );
        let ingest_bind: Vec<wgpu::BindGroup> = (0..2)
            .map(|p| {
                bind(
                    "video-ingest",
                    &self.ingest.fs.bgl,
                    &[
                        E { binding: 0, resource: BTex(&staging_view) },
                        E { binding: 1, resource: BTex(&mask_lo[1 - p]) },
                        E { binding: 2, resource: self.ingest.params.as_entire_binding() },
                    ],
                )
            })
            .collect();
        let up_bind: Vec<wgpu::BindGroup> = (0..2)
            .map(|p| {
                bind(
                    "video-up",
                    &self.up.fs.bgl,
                    &[
                        E { binding: 0, resource: BTex(&frame_view) },
                        E { binding: 1, resource: BTex(&mask_lo[p]) },
                        E { binding: 2, resource: BSampler(&self.sampler) },
                        E { binding: 3, resource: self.up.params.as_entire_binding() },
                        E { binding: 4, resource: BSampler(&self.sampler_near) },
                    ],
                )
            })
            .collect();
        // 정제 3패스: mask_hi → tmp(h) → blurred(v) → refined (raw=mask_hi).
        // 첫 blur(h) 소스는 mask_hi(v-ai personMask=NEAREST) — 최근접 샘플러.
        // 나머지 소스(tmp/blurred)는 v-ai가 LINEAR 텍스처라 선형 유지.
        let refine_binds: Vec<wgpu::BindGroup> = [
            (&mask_hi, &mask_hi, 0usize),
            (&refine_v[0], &mask_hi, 1),
            (&refine_v[1], &mask_hi, 2),
        ]
        .into_iter()
        .map(|(src, raw, i)| {
            bind(
                "vb-refine",
                if i == 2 { &self.refine.refine.bgl } else { &self.refine.blur.bgl },
                &[
                    E { binding: 0, resource: BTex(src) },
                    E { binding: 1, resource: BTex(raw) },
                    E {
                        binding: 2,
                        resource: BSampler(if i == 0 { &self.sampler_near } else { &self.sampler }),
                    },
                    E { binding: 3, resource: self.refine.params[i].as_entire_binding() },
                ],
            )
        })
        .collect();
        // 블러 체인 (v-ai 순서: **세로 먼저** — V,H ×6):
        // v(frame→b0) → h(b0→b1) → v(b1→b0) → … 마지막은 항상 b1
        let mut blur_binds = Vec::with_capacity(BLUR_ITERS * 2);
        for i in 0..BLUR_ITERS * 2 {
            let (src, dir, tgt) = if i == 0 {
                (&frame_view, 1usize, 0usize)
            } else if i % 2 == 1 {
                (&blur_v[0], 0, 1)
            } else {
                (&blur_v[1], 1, 0)
            };
            blur_binds.push((
                bind(
                    "video-blur",
                    &self.blur.fs.bgl,
                    &[
                        E { binding: 0, resource: BTex(src) },
                        E { binding: 1, resource: BTex(&refine_v[2]) },
                        E { binding: 2, resource: BSampler(&self.sampler) },
                        E { binding: 3, resource: self.blur.params[dir].as_entire_binding() },
                    ],
                ),
                tgt,
            ));
        }
        // RVM 전경색 스테이징 (마스크와 같은 dtype 규약)
        let fgr = fgr_out.map(|(n, d)| {
            let tb = d.dt.vec4_bytes() as u32;
            let (t, _) = tex2d(
                ctx,
                "vb-fgr-staging",
                d.w * d.cg(),
                d.h,
                if tb == 8 {
                    wgpu::TextureFormat::Rgba16Float
                } else {
                    wgpu::TextureFormat::Rgba32Float
                },
                wgpu::TextureUsages::COPY_DST,
            );
            (n, t, d.w * d.cg() * tb)
        });
        let bbox_bind: Vec<wgpu::BindGroup> = (0..2)
            .map(|p| {
                bind(
                    "vb-bbox",
                    &self.bbox.bgl,
                    &[
                        E { binding: 0, resource: BTex(&mask_lo[p]) },
                        E { binding: 1, resource: self.bbox.storage().as_entire_binding() },
                    ],
                )
            })
            .collect();
        let fgr_view = fgr.as_ref().map(|(_, t, _)| t.create_view(&Default::default()));
        // 배경 이미지 없으면 블러 텍스처를 더미로 (해당 분기는 안 읽힌다)
        let bg_view = self.bg_img.as_ref().map(|b| &b.1).unwrap_or(&blur_v[1]);
        let comp_bind = bind(
            "video-compose",
            &self.comp.fs.bgl,
            &[
                E { binding: 0, resource: BTex(&frame_view) },
                E { binding: 1, resource: BTex(&refine_v[2]) },
                E { binding: 2, resource: BTex(&blur_v[1]) },
                E { binding: 3, resource: BTex(bg_view) },
                E { binding: 4, resource: BSampler(&self.sampler) },
                E { binding: 5, resource: self.comp.params.as_entire_binding() },
                E {
                    binding: 6,
                    resource: BTex(fgr_view.as_ref().unwrap_or(&frame_view)),
                },
            ],
        );
        self.res = Some(Res {
            fw,
            fh,
            frame_tex,
            staging,
            mask_lo,
            mask_hi,
            refine_v,
            refine_binds,
            blur_v,
            bw,
            bh,
            pre_bind,
            ingest_bind,
            up_bind,
            blur_binds,
            comp_bind,
            mask_bytes_per_row,
            mw,
            mh,
            out_cg,
            out_name,
            fgr,
            mask_kind_alpha: out_desc.c == 1,
            in_f16: in_desc.dt.vec4_bytes() == 8,
            in_w: in_desc.w,
            in_h: in_desc.h,
            in_cg: in_desc.cg(),
            ext_mask: None,
            bbox_bind,
        });
        self.bg_dirty = false;
        Ok(())
    }

    /// 프레임 1장: 전처리→추론→마스크 스택→합성 (제출 포함).
    /// 프레임 텍스처는 미리 `with_frame_texture`로 채워져 있어야 한다.
    /// kind_alpha: 모델이 알파 직출(RVM pha)이면 true, 2ch 로짓이면 false.
    pub async fn process_gpu(
        &mut self,
        ctx: &GpuContext,
        seg: &mut GpuSession,
        fw: u32,
        fh: u32,
        target: &wgpu::TextureView,
    ) -> Result<(), TaskError> {
        self.ensure(ctx, seg, fw, fh)?;
        let p = self.parity;
        let bbox_slot = self.framing_tick(ctx);
        {
            let r = self.res.as_ref().unwrap();
            self.pre.write_params(ctx, r.in_w, r.in_h, r.in_cg);
            self.write_stack_params(
                ctx,
                r,
                if r.mask_kind_alpha { MaskKind::Alpha } else { MaskKind::Logits2 },
                r.fgr.is_some(),
            );
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("video-pre"),
            });
            self.pre.encode(&mut enc, &r.pre_bind, r.in_w, r.in_h, r.in_f16);
            ctx.queue.submit([enc.finish()]);
        }
        // 추론 — 자체 인코더·제출 (같은 큐 = 순서 보장)
        seg.infer(ctx).await?;
        let r = self.res.as_ref().unwrap();
        let model = seg.model();
        let (out_buf, _) = model
            .output_storage(&r.out_name)
            .ok_or_else(|| TaskError::Other("세그 출력 버퍼 없음".into()))?;
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("video-fx") });
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(r.mask_bytes_per_row),
                    rows_per_image: Some(r.mh),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &r.staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: r.mw * r.out_cg, height: r.mh, depth_or_array_layers: 1 },
        );
        if let Some((fgr_name, fgr_tex, fgr_bpr)) = &r.fgr {
            let (fgr_buf, fd) = model
                .output_storage(fgr_name)
                .ok_or_else(|| TaskError::Other("fgr 버퍼 없음".into()))?;
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: fgr_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(*fgr_bpr),
                        rows_per_image: Some(fd.h),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: fgr_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: fd.w * fd.cg(),
                    height: fd.h,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.encode_stack(&mut enc, r, p, &r.ingest_bind[p], bbox_slot, target);
        ctx.queue.submit([enc.finish()]);
        if let Some(s) = bbox_slot {
            self.bbox.map(s);
        }
        self.parity = 1 - p;
        Ok(())
    }

    /// 외부 마스크 주입 경로 (픽셀 diff 게이트 · P2 B티어 진입점): 추론을 건너뛰고
    /// 모델 마스크 해상도의 f32 마스크를 직접 업로드해 이펙트 스택만 돌린다.
    /// ch=1 알파 직출(RVM pha 등가, EMA 0.6/0.9), ch=2 로짓 [bg, person]
    /// (softmax 경로, EMA 0.03/0.9). ema=false면 시간 상태를 끊는다(게이트 재현성).
    /// fgr(매팅)은 쓰지 않는다 — v-ai 파리티 기준.
    pub fn process_gpu_mask(
        &mut self,
        ctx: &GpuContext,
        seg: &GpuSession,
        mask: &[f32],
        ch: u32,
        ema: bool,
        fw: u32,
        fh: u32,
        target: &wgpu::TextureView,
    ) -> Result<(), TaskError> {
        self.ensure(ctx, seg, fw, fh)?;
        let p = self.parity;
        let bbox_slot = self.framing_tick(ctx);
        {
            let r = self.res.as_mut().unwrap();
            if !(ch == 1 || ch == 2) || mask.len() != (r.mw * r.mh * ch) as usize {
                return Err(TaskError::Other(format!(
                    "외부 마스크 불일치: len {} ≠ {}×{}×{ch}",
                    mask.len(),
                    r.mw,
                    r.mh
                )));
            }
            if r.ext_mask.as_ref().map(|(c, ..)| *c != ch).unwrap_or(true) {
                let fmt = if ch == 1 {
                    wgpu::TextureFormat::R32Float
                } else {
                    wgpu::TextureFormat::Rg32Float
                };
                let (tex, view) =
                    tex2d(ctx, "vb-ext-mask", r.mw, r.mh, fmt, wgpu::TextureUsages::COPY_DST);
                let binds = (0..2)
                    .map(|pp| {
                        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("vb-ext-ingest"),
                            layout: &self.ingest.fs.bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &r.mask_lo[1 - pp],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: self.ingest.params.as_entire_binding(),
                                },
                            ],
                        })
                    })
                    .collect();
                r.ext_mask = Some((ch, tex, binds));
            }
            let (_, tex, _) = r.ext_mask.as_ref().unwrap();
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(mask),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(r.mw * 4 * ch),
                    rows_per_image: Some(r.mh),
                },
                wgpu::Extent3d { width: r.mw, height: r.mh, depth_or_array_layers: 1 },
            );
        }
        let r = self.res.as_ref().unwrap();
        self.write_stack_params_ema(
            ctx,
            r,
            if ch == 1 { MaskKind::Alpha } else { MaskKind::Logits2 },
            false,
            ema,
        );
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("video-ext") });
        let (_, _, binds) = r.ext_mask.as_ref().unwrap();
        self.encode_stack(&mut enc, r, p, &binds[p], bbox_slot, target);
        ctx.queue.submit([enc.finish()]);
        if let Some(s) = bbox_slot {
            self.bbox.map(s);
        }
        self.parity = 1 - p;
        Ok(())
    }

    /// 이펙트 스택 uniform 일괄 기록 — process_gpu / process_gpu_mask 공유
    fn write_stack_params(&self, ctx: &GpuContext, r: &Res, kind: MaskKind, use_fgr: bool) {
        self.write_stack_params_ema(ctx, r, kind, use_fgr, true);
    }

    fn write_stack_params_ema(
        &self,
        ctx: &GpuContext,
        r: &Res,
        kind: MaskKind,
        use_fgr: bool,
        ema: bool,
    ) {
        self.ingest.write_params(ctx, kind, ema);
        let d = self.state.derived();
        self.up.write_params(ctx, d.sigma_space, d.sigma_color, (r.fw, r.fh), (r.mw, r.mh));
        self.blur.write_params(ctx, r.bw, r.bh);
        // v-ai `imageBackground = !!background` — 단색도 이미지급 정제 상수
        self.refine.write_params(
            ctx,
            r.fw,
            r.fh,
            matches!(self.state.background, Background::Image | Background::Color(_)),
        );
        self.comp.write_params(
            ctx,
            &self.state,
            (r.fw, r.fh),
            self.bg_img.as_ref().map(|b| (b.2, b.3)),
            use_fgr,
            self.framing_override.unwrap_or_else(|| self.framing.current()),
        );
    }

    /// 마스크 스테이징 이후 공통 인코딩: ingest → (bbox 리덕션) → upsample →
    /// refine ×3 → (bg blur) → compose — process_gpu / process_gpu_mask 공유
    fn encode_stack(
        &self,
        enc: &mut wgpu::CommandEncoder,
        r: &Res,
        p: usize,
        ingest_bind: &wgpu::BindGroup,
        bbox_slot: Option<usize>,
        target: &wgpu::TextureView,
    ) {
        self.ingest.fs.pass(enc, "mask-ingest", &r.mask_lo[p], ingest_bind);
        if let Some(slot) = bbox_slot {
            // EMA 이후 마스크(mask_lo[p])에서 인물 bbox — 20B 리드백만 내려간다
            self.bbox.encode(enc, &r.bbox_bind[p], r.mw, r.mh, slot);
        }
        self.up.fs.pass(enc, "mask-upsample", &r.mask_hi, &r.up_bind[p]);
        self.refine.blur.pass(enc, "mask-refine-h", &r.refine_v[0], &r.refine_binds[0]);
        self.refine.blur.pass(enc, "mask-refine-v", &r.refine_v[1], &r.refine_binds[1]);
        self.refine.refine.pass(enc, "mask-refine", &r.refine_v[2], &r.refine_binds[2]);
        let need_blur = self.state.blur > 0.0 && matches!(self.state.background, Background::None);
        if need_blur {
            for (bind, tgt) in &r.blur_binds {
                self.blur.fs.pass(enc, "bg-blur", &r.blur_v[*tgt], bind);
            }
        }
        self.comp.fs.pass(enc, "compose", target, &r.comp_bind);
    }
}

/// 세그 마스크로 쓸 출력 선택: c==1(알파 직출, RVM pha) 우선, 다음 c==2(로짓),
/// 없으면 첫 출력. RVM처럼 fgr(3ch)+pha(1ch)가 섞인 모델에서 pha를 집는다.
fn mask_output(
    model: &ai_gpu_runtime::Model,
) -> Result<(String, ai_core::TensorDesc), TaskError> {
    let names: Vec<String> = model
        .sw
        .outputs
        .iter()
        .map(|&o| model.sw.tensors[o as usize].name.clone())
        .collect();
    let mut cands: Vec<(String, ai_core::TensorDesc)> = names
        .into_iter()
        .filter_map(|n| model.output_storage(&n).map(|(_, d)| (n, d)))
        .collect();
    cands.sort_by_key(|(_, d)| match d.c {
        1 => 0u32,
        2 => 1,
        c => 2 + c,
    });
    cands.into_iter().next().ok_or_else(|| TaskError::Other("세그 출력 없음".into()))
}

/// RVM류 전경색 출력 (c==3, 마스크와 다른 텐서) — 256정렬 안 되면 미사용
fn fgr_output(
    model: &ai_gpu_runtime::Model,
    mask_name: &str,
) -> Option<(String, ai_core::TensorDesc)> {
    let names: Vec<String> = model
        .sw
        .outputs
        .iter()
        .map(|&o| model.sw.tensors[o as usize].name.clone())
        .collect();
    names
        .into_iter()
        .filter(|n| n != mask_name)
        .filter_map(|n| model.output_storage(&n).map(|(_, d)| (n, d)))
        .find(|(_, d)| d.c == 3 && (d.w * d.cg() * d.dt.vec4_bytes() as u32) % 256 == 0)
}
