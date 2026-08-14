//! 회전 ROI 크롭 커널 — `crop_u8_rgb`의 GPU 짝 (셰이더: shaders/crop.wgsl).
//! 기하·수학은 CPU 판이 기준: 여기와 wgsl은 그걸 옮겨 적기만 한다.

use ai_core::TensorDesc;
use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use super::KernelPair;
use crate::detect::roi::Roi;
use crate::error::TaskError;

const SRC: &str = include_str!("shaders/crop.wgsl");

pub(crate) struct CropKernel {
    k: KernelPair,
}

impl CropKernel {
    pub fn new(ctx: &GpuContext) -> Self {
        CropKernel { k: KernelPair::new(ctx, SRC, "det-crop", 48) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ctx: &GpuContext,
        frame: &wgpu::TextureView,
        src_w: u32,
        src_h: u32,
        roi: &Roi,
        out: &wgpu::Buffer,
        desc: &TensorDesc,
        lo: f32,
        hi: f32,
    ) -> Result<(), TaskError> {
        if desc.w != desc.h {
            return Err(TaskError::Other(format!(
                "크롭은 정사각 입력 전제 (모델 입력 {}×{})",
                desc.w, desc.h
            )));
        }
        let (sinr, cosr) = roi.rotation.sin_cos();
        let mut params = [0u8; 48];
        params[..16].copy_from_slice(bytemuck::cast_slice(&[desc.w, desc.cg(), src_w, src_h]));
        params[16..].copy_from_slice(bytemuck::cast_slice(&[
            roi.cx, roi.cy, roi.w, roi.h, sinr, cosr, lo, hi,
        ]));
        ctx.queue.write_buffer(&self.k.params, 0, &params);
        self.k.dispatch(ctx, frame, out, desc, desc.w, desc.h, "det-crop")
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
