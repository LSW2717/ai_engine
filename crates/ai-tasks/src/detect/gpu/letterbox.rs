//! 레터박스 커널 — `letterbox_u8_rgb`의 GPU 짝 (셰이더: shaders/letterbox.wgsl).
//! 기하·수학은 CPU 판이 기준: 여기와 wgsl은 그걸 옮겨 적기만 한다.

use ai_core::TensorDesc;
use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use super::KernelPair;
use crate::error::TaskError;

const SRC: &str = include_str!("shaders/letterbox.wgsl");

pub(crate) struct LetterboxKernel {
    k: KernelPair,
}

impl LetterboxKernel {
    pub fn new(ctx: &GpuContext) -> Self {
        LetterboxKernel { k: KernelPair::new(ctx, SRC, "det-letterbox", 32) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ctx: &GpuContext,
        frame: &wgpu::TextureView,
        src_w: u32,
        src_h: u32,
        out: &wgpu::Buffer,
        desc: &TensorDesc,
        lo: f32,
        hi: f32,
    ) -> Result<(), TaskError> {
        let mut params = [0u8; 32];
        params[..24].copy_from_slice(bytemuck::cast_slice(&[
            desc.w,
            desc.h,
            desc.cg(),
            src_w,
            src_h,
            0,
        ]));
        params[24..].copy_from_slice(bytemuck::cast_slice(&[lo, hi]));
        ctx.queue.write_buffer(&self.k.params, 0, &params);
        self.k.dispatch(ctx, frame, out, desc, desc.w, desc.h, "det-letterbox")
    }
}

#[cfg(test)]
mod tests {
    fn validate(src: &str) {
        let m = naga::front::wgsl::parse_str(src).unwrap();
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap();
    }

    #[test]
    fn wgsl_valid() {
        validate(super::SRC);
        validate(&super::super::src_f16(super::SRC));
    }
}
