//! 배경 블러 — 1/5 해상도 분리 가우시안 ×반복 (인물 제외 가중)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::stages::{ubo, Bind, Fullscreen, FULLSCREEN_VS};

const SRC: &str = include_str!("../shaders/bg_blur.wgsl");

/// 웹 buildBlurPass와 동일 구조 (scale 0.2, 7탭, 6회 반복)
pub(crate) const BLUR_SCALE: u32 = 5;
pub(crate) const BLUR_ITERS: usize = 6;

pub(crate) struct BgBlur {
    pub fs: Fullscreen,
    /// [0]=수평, [1]=수직 — 방향은 바인드그룹이 아니라 UBO 2개로 고정
    pub params: [wgpu::Buffer; 2],
}

impl BgBlur {
    pub fn new(ctx: &GpuContext) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let fs = Fullscreen::new(
            ctx,
            "video-bg-blur",
            &src,
            wgpu::TextureFormat::Rgba8Unorm,
            &[Bind::Tex, Bind::Tex, Bind::Sampler, Bind::Uniform],
        );
        BgBlur {
            fs,
            params: [ubo(ctx, "video-bg-blur-h", 16), ubo(ctx, "video-bg-blur-v", 16)],
        }
    }

    pub fn write_params(&self, ctx: &GpuContext, bw: u32, bh: u32) {
        let h: [f32; 4] = [1.0 / bw as f32, 0.0, 0.0, 0.0];
        let v: [f32; 4] = [0.0, 1.0 / bh as f32, 0.0, 0.0];
        ctx.queue.write_buffer(&self.params[0], 0, bytemuck::cast_slice(&h));
        ctx.queue.write_buffer(&self.params[1], 0, bytemuck::cast_slice(&v));
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
