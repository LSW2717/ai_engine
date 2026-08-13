//! GPU 전처리 — 프레임 텍스처 → 모델 입력 버퍼 (컴퓨트, CPU 픽셀 0)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::stages::ubo;

const SRC: &str = include_str!("../shaders/preprocess.wgsl");

pub(crate) struct Preprocess {
    pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub params: wgpu::Buffer,
}

impl Preprocess {
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video-preprocess"),
            source: wgpu::ShaderSource::Wgsl(SRC.into()),
        });
        let pipeline =
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("video-preprocess"),
                layout: None, // 자동 레이아웃 — 바인딩이 셰이더에 전부 선언돼 있다
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bgl = pipeline.get_bind_group_layout(0);
        Preprocess { pipeline, bgl, params: ubo(ctx, "video-preprocess", 16) }
    }

    pub fn write_params(&self, ctx: &GpuContext, w: u32, h: u32, cg: u32) {
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[w, h, cg, 0]));
    }

    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, bind: &wgpu::BindGroup, w: u32, h: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_valid() {
        let m = naga::front::wgsl::parse_str(super::SRC).unwrap();
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap();
    }
}
