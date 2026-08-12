//! ai-gpu — wgpu 30 기반 커널 실행 계층.
//!
//! GpuContext(디바이스/caps), arena(정적 메모리 플랜), params(유니폼 슬롯),
//! KernelSpec→CompiledKernel→OpDispatch 파이프라인, 커널별 WGSL 템플릿
//! (`kernels/<이름>.rs` ↔ `kernels/shaders/<이름>.wgsl`), 파이프라인 캐시,
//! 타임스탬프 프로파일러, 공유 정확도 스위트·벤치 코어.

pub mod arena;
pub mod bench;
pub mod cache;
pub mod context;
pub mod kernel;
pub mod kernels;
pub mod params;
pub mod profile;
pub mod readback;
pub mod testsuite;

pub use context::{DeviceCaps, GpuContext, InitError};

/// 하위 크레이트(ai-runtime 등)가 wgpu 타입을 쓸 수 있게 재수출
/// (타겟별 feature 구성은 ai-gpu가 소유 — 중복 선언 방지)
pub use wgpu;

/// 단위 테스트 공용 유틸 (naga 검증은 GPU 없이 동작)
#[cfg(test)]
pub(crate) mod test_util {
    use crate::context::DeviceCaps;

    /// WGSL 소스를 naga로 파싱+검증. 실패 시 소스 전문과 함께 panic.
    pub fn validate_wgsl(src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("WGSL 파싱 실패: {e}\n--- 소스 ---\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL 검증 실패: {e:?}\n--- 소스 ---\n{src}"));
    }

    /// GPU 없이 codegen을 돌리기 위한 가짜 caps
    pub fn fake_caps() -> DeviceCaps {
        DeviceCaps {
            f16: true,
            subgroups: false,
            timestamps: false,
            limits: wgpu::Limits::default(),
            info: wgpu::AdapterInfo {
                name: "fake".into(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Other,
                device_pci_bus_id: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Noop,
                subgroup_min_size: 0,
                subgroup_max_size: 0,
                transient_saves_memory: None,
                limit_bucket: None,
            },
            storage_align: 256,
        }
    }
}
