//! GPU 입력 전처리 — 프레임 텍스처에서 디텍터/랜드마크 **모델 입력 버퍼로 직결**
//! (`input_storage`), 프레임당 CPU 픽셀 왕복 0.
//!
//! CPU 짝(`letterbox_u8_rgb`/`crop_u8_rgb`)과 같은 수학을 컴퓨트 커널로 옮긴 것 —
//! 좌표 규약(레터박스=픽셀 중심, 크롭=warpPerspective 코너 정합)이 서로 다르므로
//! 커널도 2개다. 파리티는 tests/tex_input.rs(커널 단독, CPU 대조)와
//! tests/face_task_tex.rs(태스크 e2e)가 지킨다.
//!
//! 저사양 타깃 근거(target-hardware-lowend): getImageData 리드백(720p ~3.7MB/프레임)
//! + JS 재패킹 + wasm 복사가 구형 iGPU·저가 노트북에서 실비용이다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

pub mod crop;
pub mod frame;
pub mod letterbox;

pub use frame::FrameTex;

use crate::detect::roi::Roi;
use crate::error::TaskError;
use ai_core::TensorDesc;

/// f32 소스 → f16 스토리지 변형 (입력 버퍼가 f16 레인일 때).
/// vb preprocess와 같은 규약 — 스토리지 배열 타입과 최종 store만 바꾼다.
fn src_f16(src: &str) -> String {
    format!(
        "enable f16;\n{}",
        src.replace("array<vec4f>", "array<vec4<f16>>")
            .replace("= vec4f(val, 0.0);", "= vec4<f16>(vec4f(val, 0.0));")
    )
}

/// 컴퓨트 커널 한 벌 (f32 + 옵션 f16) — 두 커널의 공통 뼈대.
///
/// ⚠ f32/f16이 **명시적 공유 레이아웃**을 쓴다 — auto-layout 바인드그룹 재사용이
/// 패스를 조용히 무효화한 vb preprocess 사고(NEXT.md "마스크 전멸")의 재발 방지.
pub(crate) struct KernelPair {
    pub bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pipeline_f16: Option<wgpu::ComputePipeline>,
    pub params: wgpu::Buffer,
}

impl KernelPair {
    pub fn new(ctx: &GpuContext, src: &str, label: &str, params_size: u64) -> Self {
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
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
            label: Some(label),
            entries: &entries,
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
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
        let pipeline = make(src, label);
        // 디바이스가 실제로 켠 기능 기준 (adapter caps 아님 — 미요청 기기에서
        // 생성 시도만으로 검증 에러)
        let pipeline_f16 = dev
            .features()
            .contains(wgpu::Features::SHADER_F16)
            .then(|| make(&src_f16(src), &format!("{label}-f16")));
        let params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        KernelPair { bgl, pipeline, pipeline_f16, params }
    }

    /// 바인드그룹 + 디스패치 + 제출. params는 호출자가 먼저 write_buffer.
    ///
    /// 바인드그룹은 캐시하지 않고 매번 만든다: 프레임 텍스처(리사이즈)·세션(입력
    /// 버퍼)이 호출 사이에 바뀔 수 있고, 낡은 바인딩은 **이전 모델 버퍼에 조용히
    /// 쓴다** (studio_invalidate 사고와 같은 급). 프레임당 ≤2회라 비용은 µs대 —
    /// 문제가 되면 세대(gen) 키 캐시로.
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        frame: &wgpu::TextureView,
        out: &wgpu::Buffer,
        desc: &TensorDesc,
        gx: u32,
        gy: u32,
        label: &str,
    ) -> Result<(), TaskError> {
        let f16 = desc.dt.vec4_bytes() == 8;
        let pipeline = if f16 {
            self.pipeline_f16.as_ref().ok_or_else(|| {
                TaskError::Other("f16 입력 모델인데 SHADER_F16 미가동 디바이스".into())
            })?
        } else {
            &self.pipeline
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame),
                },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(gx.div_ceil(8), gy.div_ceil(8), 1);
        }
        // 자체 제출 — 같은 큐라 이후 infer 제출보다 먼저 실행됨이 보장된다
        // (vb preprocess가 seg.infer 앞에 별도 제출하는 것과 같은 규약)
        ctx.queue.submit([enc.finish()]);
        Ok(())
    }
}

/// c=3(RGB) 입력 전제 검증 — 커널이 픽셀당 vec4 하나([r,g,b,0])만 쓴다
fn ensure_rgb(desc: &TensorDesc) -> Result<(), TaskError> {
    if desc.c != 3 {
        return Err(TaskError::Other(format!(
            "GPU 입력 전처리는 c=3 전제 (모델 입력 c={})",
            desc.c
        )));
    }
    Ok(())
}

/// GPU 입력 전처리 한 벌 — 레터박스 + 회전 크롭 커널과 (스탠드얼론용) 프레임
/// 텍스처. 태스크 상태(FaceTask 등)와 분리해 호스트/바인딩이 소유한다 —
/// 커널은 세션·태스크 어느 쪽에도 묶이지 않는 무상태 리소스라서
/// (face·hand가 같은 커널을 쓰고, 프레임 텍스처는 studio처럼 밖에서 올 수 있다).
pub struct GpuPre {
    /// 스탠드얼론 프레임 텍스처 (studio는 파이프라인 텍스처를 대신 넘긴다)
    pub frame: FrameTex,
    letterbox: letterbox::LetterboxKernel,
    crop: crop::CropKernel,
}

impl GpuPre {
    pub fn new(ctx: &GpuContext) -> Self {
        GpuPre {
            frame: FrameTex::new(),
            letterbox: letterbox::LetterboxKernel::new(ctx),
            crop: crop::CropKernel::new(ctx),
        }
    }

    /// 프레임(src_w×src_h) → keep_aspect 레터박스 → `out`(디텍터 입력 버퍼).
    /// [lo,hi]는 `DetectorPost::input_range()` — face [-1,1] / palm [0,1].
    #[allow(clippy::too_many_arguments)]
    pub fn letterbox_into(
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
        ensure_rgb(desc)?;
        self.letterbox.run(ctx, frame, src_w, src_h, out, desc, lo, hi)
    }

    /// 프레임에서 회전 ROI를 desc.h(=desc.w) 정사각으로 warp → `out`(랜드마크
    /// 입력 버퍼). 랜드마크 규약은 [0,1] (crop_u8_rgb와 동일).
    #[allow(clippy::too_many_arguments)]
    pub fn crop_into(
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
        ensure_rgb(desc)?;
        self.crop.run(ctx, frame, src_w, src_h, roi, out, desc, lo, hi)
    }
}
