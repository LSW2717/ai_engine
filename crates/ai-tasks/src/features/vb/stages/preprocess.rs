//! GPU 전처리 — 프레임 텍스처 → 모델 입력 버퍼 (컴퓨트, CPU 픽셀 0)
//!
//! ⚠ f32/f16 파이프라인이 **명시적 공유 레이아웃**을 쓴다: auto-layout(`layout:
//! None`)은 파이프라인마다 별개 정체성이라 한쪽 레이아웃으로 만든 바인드그룹을
//! 다른 파이프라인에 물리면 패스가 조용히 무효화된다 — 전처리가 아무것도 안 써서
//! 마스크가 통째로 죽었던 사고의 원인. f16 변형은 디바이스가 SHADER_F16을
//! 실제로 켰을 때만 생성한다(adapter 지원 여부와 다르다 — 미요청 기기에서
//! 생성 시도만으로 검증 에러 → 디바이스 오염).

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use super::ubo;

const SRC: &str = include_str!("../shaders/preprocess.wgsl");

pub(crate) struct Preprocess {
    pipeline: wgpu::ComputePipeline,
    pipeline_f16: Option<wgpu::ComputePipeline>,
    pub bgl: wgpu::BindGroupLayout,
    pub params: wgpu::Buffer,
}

/// f32 소스 → f16 스토리지 변형 (입력 버퍼가 f16 레인일 때)
fn src_f16() -> String {
    format!(
        "enable f16;\n{}",
        SRC.replace("array<vec4f>", "array<vec4<f16>>").replace(
            "out[(gid.y * p.w + gid.x) * p.cg] = vec4f(rgb, 0.0);",
            "out[(gid.y * p.w + gid.x) * p.cg] = vec4<f16>(vec4f(rgb, 0.0));"
        )
    )
}

impl Preprocess {
    pub fn new(ctx: &GpuContext) -> Self {
        let dev = &ctx.device;
        let entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vb-preprocess"),
            entries: &entries,
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vb-preprocess"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let make = |src: &str, label: &str| {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = make(SRC, "vb-preprocess");
        // 디바이스가 실제로 켠 기능 기준 (adapter caps 아님!)
        let pipeline_f16 = ctx
            .device
            .features()
            .contains(wgpu::Features::SHADER_F16)
            .then(|| make(&src_f16(), "vb-preprocess-f16"));
        Preprocess { pipeline, pipeline_f16, bgl, params: ubo(ctx, "vb-preprocess", 16) }
    }

    pub fn write_params(&self, ctx: &GpuContext, w: u32, h: u32, cg: u32) {
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[w, h, cg, 0]));
    }

    /// f16: 입력 버퍼가 f16 레인이면 true (desc.dt.vec4_bytes()==8)
    pub fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        w: u32,
        h: u32,
        f16: bool,
    ) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(if f16 {
            self.pipeline_f16.as_ref().expect("f16 모델인데 SHADER_F16 미가동 디바이스")
        } else {
            &self.pipeline
        });
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
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
        validate(&super::src_f16());
    }
}
