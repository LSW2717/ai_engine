//! 픽셀 diff 게이트 하네스 — 프레임(RGBA8)·마스크(f32)를 주입해 이펙트 스택만
//! 돌리고 최종 RGBA를 리드백한다. v-ai GLSL 스택과의 파리티 게이트가 소비
//! (web/demo/vb-diff.html — P1 완료 조건). 네이티브·wasm 공용.
//!
//! 추론은 돌지 않는다 — 세션은 마스크 해상도·리소스 확보에만 쓰인다.
//! 시간 상태(EMA)는 `reset()`으로 초기화 (프로토콜 단계 재현성).

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::error::TaskError;
use crate::session::gpu::GpuSession;
use super::pipeline::VideoPipeline;

struct Target {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    buf: wgpu::Buffer,
    bpr: u32,
    w: u32,
    h: u32,
}

pub struct GateHarness {
    pub pipeline: VideoPipeline,
    tgt: Option<Target>,
}

impl GateHarness {
    pub fn new(ctx: &GpuContext) -> Self {
        GateHarness {
            pipeline: VideoPipeline::new(ctx, wgpu::TextureFormat::Rgba8Unorm),
            tgt: None,
        }
    }

    /// 시간 상태(EMA 핑퐁) 초기화 — 리소스 재생성 (새 텍스처는 0으로 보장)
    pub fn reset(&mut self) {
        self.pipeline.invalidate();
    }

    /// 프레임 1장 주입 → 이펙트 스택 → 최종 RGBA (행 패딩 제거된 fw×fh×4).
    /// ema=false면 시간 상태(EMA)를 끊는다 — 공간 스택만 결정적으로 게이트.
    pub async fn frame(
        &mut self,
        ctx: &GpuContext,
        seg: &GpuSession,
        frame_rgba: &[u8],
        fw: u32,
        fh: u32,
        mask: &[f32],
        ch: u32,
        ema: bool,
    ) -> Result<Vec<u8>, TaskError> {
        if frame_rgba.len() != (fw * fh * 4) as usize {
            return Err(TaskError::Other(format!(
                "프레임 크기 불일치: {} ≠ {fw}×{fh}×4",
                frame_rgba.len()
            )));
        }
        if self.tgt.as_ref().map(|t| t.w != fw || t.h != fh).unwrap_or(true) {
            let bpr = (fw * 4 + 255) & !255;
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vb-gate-target"),
                size: wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vb-gate-read"),
                size: (bpr * fh) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.tgt = Some(Target { tex, view, buf, bpr, w: fw, h: fh });
        }
        self.pipeline.with_frame_texture(ctx, seg, fw, fh, |tex| {
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                frame_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(fw * 4),
                    rows_per_image: Some(fh),
                },
                wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
            );
        })?;
        let t = self.tgt.as_ref().unwrap();
        self.pipeline.process_gpu_mask(ctx, seg, mask, ch, ema, fw, fh, &t.view)?;
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vb-gate") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &t.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &t.buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(t.bpr),
                    rows_per_image: Some(fh),
                },
            },
            wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
        );
        ctx.queue.submit([enc.finish()]);
        let data = ai_gpu::readback::read_buffers(ctx, &[&t.buf])
            .await
            .map_err(TaskError::Other)?
            .remove(0);
        // 행 패딩 제거
        let row = (fw * 4) as usize;
        let mut out = Vec::with_capacity(row * fh as usize);
        for y in 0..fh as usize {
            let s = y * t.bpr as usize;
            out.extend_from_slice(&data[s..s + row]);
        }
        Ok(out)
    }
}
