//! 마스크 정제 — 분리 블러 h/v + 엣지 인지 재혼합 (v-ai 파리티:
//! maskBlurPx 1.1/1.2, edgeBlend 0.36/0.4, edgeGamma 0.98, edgeFeather 0.54/0.58)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use super::{ubo, Bind, Fullscreen, FULLSCREEN_VS};

const SRC: &str = include_str!("../shaders/mask_refine.wgsl");

pub(crate) struct MaskRefine {
    pub blur: Fullscreen,
    pub refine: Fullscreen,
    /// [0]=blur h, [1]=blur v, [2]=refine
    pub params: [wgpu::Buffer; 3],
}

impl MaskRefine {
    pub fn new(ctx: &GpuContext) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let binds = [Bind::Tex, Bind::Tex, Bind::Sampler, Bind::Uniform];
        MaskRefine {
            blur: Fullscreen::new_entry(
                ctx, "vb-mask-refine-blur", &src, wgpu::TextureFormat::R8Unorm, &binds, "fs_blur",
            ),
            refine: Fullscreen::new_entry(
                ctx, "vb-mask-refine", &src, wgpu::TextureFormat::R8Unorm, &binds, "fs_refine",
            ),
            params: [
                ubo(ctx, "vb-refine-h", 32),
                ubo(ctx, "vb-refine-v", 32),
                ubo(ctx, "vb-refine-r", 32),
            ],
        }
    }

    /// image_bg: 이미지 배경이면 정제 상수가 살짝 강해진다 (웹 규약)
    pub fn write_params(&self, ctx: &GpuContext, fw: u32, fh: u32, image_bg: bool) {
        let (blur_px, blend, feather) =
            if image_bg { (1.2f32, 0.4f32, 0.58f32) } else { (1.1, 0.36, 0.54) };
        let s = blur_px.max(1.0);
        let h: [f32; 8] = [s / fw as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v: [f32; 8] = [0.0, s / fh as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r: [f32; 8] =
            [1.0 / fw as f32, 1.0 / fh as f32, blend, 0.98, feather, 0.0, 0.0, 0.0];
        ctx.queue.write_buffer(&self.params[0], 0, bytemuck::cast_slice(&h));
        ctx.queue.write_buffer(&self.params[1], 0, bytemuck::cast_slice(&v));
        ctx.queue.write_buffer(&self.params[2], 0, bytemuck::cast_slice(&r));
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
