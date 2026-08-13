//! 마스크 업샘플 — joint bilateral (프레임 가이드)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::stages::{ubo, Bind, Fullscreen, FULLSCREEN_VS};

const SRC: &str = include_str!("../shaders/mask_upsample.wgsl");

pub(crate) struct MaskUpsample {
    pub fs: Fullscreen,
    pub params: wgpu::Buffer,
}

impl MaskUpsample {
    pub fn new(ctx: &GpuContext) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let fs = Fullscreen::new(
            ctx,
            "video-mask-upsample",
            &src,
            wgpu::TextureFormat::R8Unorm,
            // 마지막 Sampler = binding 4 (마스크 최근접 샘플러 — v-ai NEAREST 파리티)
            &[Bind::Tex, Bind::Tex, Bind::Sampler, Bind::Uniform, Bind::Sampler],
        );
        MaskUpsample { fs, params: ubo(ctx, "video-mask-upsample", 32) }
    }

    /// 웹 updateSigmaSpace 등가 — σ를 업샘플 배율로 스케일하고 스텝/반경을 유도.
    /// 계산은 f64 (웹 JS와 비트 정합 — step이 1 ULP 다르면 루프 탭 수가 달라져
    /// 픽셀 diff 게이트가 흔들린다).
    pub fn write_params(
        &self,
        ctx: &GpuContext,
        sigma_space: f32,
        sigma_color: f32,
        (fw, fh): (u32, u32),
        (mw, mh): (u32, u32),
    ) {
        let sigma = sigma_space as f64 * (fw as f64 / mw as f64).max(fh as f64 / mh as f64);
        let step = (sigma.sqrt() * 0.66).max(1.0);
        let offset = if step > 1.0 { step * 0.5 } else { 0.0 };
        let (tx, ty) = (1.0 / fw as f64, 1.0 / fh as f64);
        // wgsl 필드 순서: texel, step, radius, offset, sigma_texel, sigma_color
        let p: [f32; 8] = [
            tx as f32,
            ty as f32,
            step as f32,
            sigma as f32,
            offset as f32,
            (tx.max(ty) * sigma) as f32,
            sigma_color,
            0.0,
        ];
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_valid() {
        let src = format!("{}\n{}", super::FULLSCREEN_VS, super::SRC);
        let m = naga::front::wgsl::parse_str(&src).unwrap();
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap();
    }
}
