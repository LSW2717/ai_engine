//! 커널 추상화 — KernelSpec(codegen 입력) → CompiledKernel(파이프라인) → OpDispatch(실행 단위).
//!
//! 바인딩 규약(모든 커널 공통):
//! - `@binding(0)`: params uniform (dynamic offset, op당 256B 슬롯)
//! - `@binding(1..)`: storage 버퍼들 (spec.bindings() 순서, 마지막이 출력이 관례)
//!
//! `OpDispatch`의 bind group은 로드 시 한 번 생성된다 — 프레임 루프에서의
//! bind group/버퍼 생성은 이 엔진의 금지 사항이다.

use std::sync::Arc;

use crate::context::{DeviceCaps, GpuContext};

/// storage 바인딩의 접근 방향 (Uniform = Metal constant 주소공간 — 브로드캐스트
/// 최적 경로. 64KB 이하 read-only 상수 데이터 전용)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageDir {
    Read,
    ReadWrite,
    Uniform,
}

/// 커널 계열마다 하나씩 구현하는 codegen 명세.
/// 구현 파일 규약은 ARCHITECTURE.md 참조 (Spec + 변형 정책 + impl + naga 테스트).
/// Send+Sync: 로드 시 병렬 컴파일 팬아웃을 위해 (spec은 순수 데이터).
pub trait KernelSpec: Send + Sync {
    /// codegen 입력 전체를 담은 canonical 키. 같은 키 ⟺ 같은 WGSL.
    /// 프로파일러 라벨로도 쓰이므로 사람이 읽을 수 있게.
    fn cache_key(&self, caps: &DeviceCaps) -> String;
    /// 완전히 특수화된 WGSL 소스 (shape은 리터럴 상수)
    fn wgsl(&self, caps: &DeviceCaps) -> String;
    /// binding 1부터의 storage 바인딩 방향 목록
    fn bindings(&self) -> Vec<StorageDir>;
    /// 디스패치 워크그룹 수
    fn workgroups(&self) -> [u32; 3];
}

pub struct CompiledKernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub key: String,
}

/// 실행 준비가 끝난 op 하나. Phase 2 executor는 그래프를 `Vec<OpDispatch>`로 lowering한다.
/// bind_group이 Arc인 이유: 상태 ping-pong의 even/odd 디스패치 리스트 2벌이
/// 상태 무관 op의 bind group을 공유하기 위해서다.
pub struct OpDispatch {
    pub kernel: Arc<CompiledKernel>,
    pub bind_group: Arc<wgpu::BindGroup>,
    pub groups: [u32; 3],
    /// params 슬롯의 dynamic offset (바이트, 256 배수)
    pub param_offset: u32,
    pub label: String,
}

/// 2단계 컴파일의 중간 산출물 — 모듈 생성(비싼 셰이더 번역이 백엔드에서 비동기로
/// 시작됨)과 파이프라인 생성(모듈 완료 대기)을 분리해, wasm에서 모듈들을 먼저 전부
/// 만들어 브라우저 내부 스레드풀 병렬 컴파일을 유도한다.
pub struct PendingKernel {
    module: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    key: String,
}

/// 콜드 컴파일 측정용 소스 salt. `AI_WGSL_SALT`로 빌드하면 주석 한 줄이 붙어
/// 소스 해시가 달라지고, OS/브라우저 셰이더 캐시를 확실히 우회한다.
/// (이게 없으면 macOS Metal 캐시 때문에 같은 셰이더가 0.3ms~200ms로 널뛰어
/// "무엇이 비싼가"를 측정할 수 없다 — 실제로 한 번 틀린 결론을 낸 적이 있다.)
fn salted(src: String) -> String {
    match option_env!("AI_WGSL_SALT") {
        Some(s) => format!("// salt {s}\n{src}"),
        None => src,
    }
}

/// 1단계: WGSL codegen + 모듈/BGL 생성 (error scope 없음 — 병렬 팬아웃용.
/// 오류는 on_uncaptured_error 로그 + 워밍업 디스패치에서 드러난다.)
pub fn begin_compile(ctx: &GpuContext, spec: &dyn KernelSpec) -> PendingKernel {
    let key = spec.cache_key(&ctx.caps);
    let src = salted(spec.wgsl(&ctx.caps));
    let entries = bgl_entries(&spec.bindings());
    let device = &ctx.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&key),
        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&key),
        entries: &entries,
    });
    PendingKernel { module, bgl, key }
}

/// 2단계: 파이프라인 생성 (백엔드 셰이더 컴파일 완료 대기 지점)
pub fn finish_compile(ctx: &GpuContext, p: PendingKernel) -> CompiledKernel {
    let device = &ctx.device;
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&p.key),
        bind_group_layouts: &[Some(&p.bgl)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&p.key),
        layout: Some(&layout),
        module: &p.module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    CompiledKernel { pipeline, bgl: p.bgl, key: p.key }
}

/// error scope 없이 일괄 컴파일 — 병렬 팬아웃용
pub fn compile_unscoped(ctx: &GpuContext, spec: &dyn KernelSpec) -> CompiledKernel {
    finish_compile(ctx, begin_compile(ctx, spec))
}

fn bgl_entries(storage: &[StorageDir]) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }];
    for (i, dir) in storage.iter().enumerate() {
        let ty = match dir {
            StorageDir::Uniform => wgpu::BufferBindingType::Uniform,
            d => wgpu::BufferBindingType::Storage { read_only: *d == StorageDir::Read },
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (i + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        });
    }
    entries
}

/// KernelSpec → CompiledKernel. validation error scope로 감싸 오류를 Result로.
pub async fn compile(ctx: &GpuContext, spec: &dyn KernelSpec) -> Result<CompiledKernel, String> {
    let key = spec.cache_key(&ctx.caps);
    let src = spec.wgsl(&ctx.caps);
    let entries = bgl_entries(&spec.bindings());

    ctx.with_validation(|device| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&key),
            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&key),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&key),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&key),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        CompiledKernel { pipeline, bgl, key }
    })
    .await
    .map_err(|e| format!("커널 컴파일 실패 [{}]: {e}", spec.cache_key(&ctx.caps)))
}

/// 디스패치 목록을 단일 컴퓨트 패스로 기록.
/// (wgpu가 storage 버퍼 해저드 배리어를 자동 삽입한다)
pub fn record(encoder: &mut wgpu::CommandEncoder, ops: &[OpDispatch]) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("ai-engine"),
        timestamp_writes: None,
    });
    for op in ops {
        pass.set_pipeline(&op.kernel.pipeline);
        pass.set_bind_group(0, op.bind_group.as_ref(), &[op.param_offset]);
        pass.dispatch_workgroups(op.groups[0], op.groups[1], op.groups[2]);
    }
}
