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

use crate::error::TaskError;
use crate::session::gpu::GpuSession;
pub use super::params::{Background, EffectsPatch, EffectsState};
use super::stages::tex2d;
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
}

pub struct VideoPipeline {
    pub state: EffectsState,
    pre: Preprocess,
    ingest: MaskIngest,
    up: MaskUpsample,
    refine: MaskRefine,
    blur: BgBlur,
    comp: Compose,
    sampler: wgpu::Sampler,
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
            sampler: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("video"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
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

    /// 세션 교체 시 리소스 폐기 (모델 버퍼가 바인드그룹에 박혀 있다)
    pub fn invalidate(&mut self) {
        self.res = None;
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
                    ],
                )
            })
            .collect();
        // 정제 3패스: mask_hi → tmp(h) → blurred(v) → refined (raw=mask_hi)
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
                    E { binding: 2, resource: BSampler(&self.sampler) },
                    E { binding: 3, resource: self.refine.params[i].as_entire_binding() },
                ],
            )
        })
        .collect();
        // 블러 체인: h(frame→b0) → v(b0→b1) → h(b1→b0) → … 마지막은 항상 b1
        let mut blur_binds = Vec::with_capacity(BLUR_ITERS * 2);
        for i in 0..BLUR_ITERS * 2 {
            let (src, dir, tgt) = if i == 0 {
                (&frame_view, 0usize, 0usize)
            } else if i % 2 == 1 {
                (&blur_v[0], 1, 1)
            } else {
                (&blur_v[1], 0, 0)
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
        {
            let r = self.res.as_ref().unwrap();
            self.pre.write_params(ctx, r.in_w, r.in_h, r.in_cg);
            self.ingest.write_params(
                ctx,
                if r.mask_kind_alpha { MaskKind::Alpha } else { MaskKind::Logits2 },
            );
            let d = self.state.derived();
            self.up.write_params(ctx, d.sigma_space, d.sigma_color, (r.fw, r.fh), (r.mw, r.mh));
            self.blur.write_params(ctx, r.bw, r.bh);
            self.refine.write_params(
                ctx,
                r.fw,
                r.fh,
                matches!(self.state.background, Background::Image),
            );
            self.comp.write_params(
                ctx,
                &self.state,
                (r.fw, r.fh),
                self.bg_img.as_ref().map(|b| (b.2, b.3)),
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
        self.ingest.fs.pass(&mut enc, "mask-ingest", &r.mask_lo[p], &r.ingest_bind[p]);
        self.up.fs.pass(&mut enc, "mask-upsample", &r.mask_hi, &r.up_bind[p]);
        self.refine.blur.pass(&mut enc, "mask-refine-h", &r.refine_v[0], &r.refine_binds[0]);
        self.refine.blur.pass(&mut enc, "mask-refine-v", &r.refine_v[1], &r.refine_binds[1]);
        self.refine.refine.pass(&mut enc, "mask-refine", &r.refine_v[2], &r.refine_binds[2]);
        let need_blur = self.state.blur > 0.0 && matches!(self.state.background, Background::None);
        if need_blur {
            for (bind, tgt) in &r.blur_binds {
                self.blur.fs.pass(&mut enc, "bg-blur", &r.blur_v[*tgt], bind);
            }
        }
        self.comp.fs.pass(&mut enc, "compose", target, &r.comp_bind);
        ctx.queue.submit([enc.finish()]);
        self.parity = 1 - p;
        Ok(())
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
