//! GPU 디바이스 초기화와 capability 스냅샷.
//!
//! wgpu 버전 민감 API(request_adapter/request_device/poll/error scope)는
//! 이 파일과 readback.rs에만 존재한다 — 버전 업 시 수정 지점을 격리.
//!
//! limits는 `Limits::default()`(WebGPU 이식성 기본값)만 요청한다. adapter limits를
//! 요청하면 네이티브에서는 되고 웹에서 깨지는 커널이 침묵 속에 만들어진다.

use thiserror::Error;

/// init 실패의 구조화된 이유 — 호스트가 폴백 티어 결정에 사용한다.
#[derive(Error, Debug)]
pub enum InitError {
    /// WebGPU/adapter 자체가 없음 (웹: navigator.gpu 부재 포함)
    #[error("GPU adapter 없음: {0}")]
    NoAdapter(String),
    /// adapter는 있으나 디바이스 요청 실패
    #[error("디바이스 요청 실패: {0}")]
    RequestDevice(String),
}

/// 디바이스 capability 스냅샷. 커널 변형 선택(f16), 프로파일러(timestamps),
/// arena 정렬(storage_align), 기기별 티어 판정(info)의 근거.
#[derive(Debug)]
pub struct DeviceCaps {
    pub f16: bool,
    /// subgroup 연산 지원 (WGSL enable subgroups)
    pub subgroups: bool,
    pub timestamps: bool,
    pub limits: wgpu::Limits,
    pub info: wgpu::AdapterInfo,
    /// min_storage_buffer_offset_alignment (보통 256)
    pub storage_align: u32,
}

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: DeviceCaps,
    /// 프로세스 수명 파이프라인 캐시 (cache_key → 컴파일 완료 커널).
    /// 재로드·다중 모델이 같은 shape 커널을 공유한다 — 로드 시간의 지배 항목이
    /// 파이프라인 컴파일이라 2회차 로드가 버퍼 비용만 남는다.
    pub kernel_cache: std::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<crate::kernel::CompiledKernel>>,
    >,
}

impl GpuContext {
    /// 실행기 무관 비동기 초기화 — 네이티브·wasm 공용.
    pub async fn new() -> Result<Self, InitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| InitError::NoAdapter(e.to_string()))?;

        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        for f in [
            wgpu::Features::SHADER_F16,
            wgpu::Features::TIMESTAMP_QUERY,
            wgpu::Features::SUBGROUP,
        ] {
            if adapter_features.contains(f) {
                required_features |= f;
            }
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ai-engine"),
                required_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError::RequestDevice(e.to_string()))?;

        // error scope 밖에서 새는 오류도 침묵하지 않도록.
        // 단 wasm에서는 등록하지 않는다: wgpu 30 웹 백엔드의 Error::from_js가
        // Validation/OOM 외 타입(GPUInternalError 등)에서 panic!("Unexpected error")로
        // 죽는 업스트림 이슈가 있어, 웹은 브라우저 콘솔의 기본 에러 표시에 맡긴다.
        #[cfg(not(target_arch = "wasm32"))]
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            log::error!("[ai-gpu] uncaptured GPU error: {e}");
        }));

        let limits = device.limits();
        let caps = DeviceCaps {
            f16: required_features.contains(wgpu::Features::SHADER_F16),
            subgroups: required_features.contains(wgpu::Features::SUBGROUP),
            timestamps: required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
            storage_align: limits.min_storage_buffer_offset_alignment,
            limits,
            info: adapter.get_info(),
        };
        log::info!(
            "[ai-gpu] adapter: {} ({:?}), f16={}, timestamps={}",
            caps.info.name,
            caps.info.backend,
            caps.f16,
            caps.timestamps
        );

        Ok(Self { instance, adapter, device, queue, caps, kernel_cache: Default::default() })
    }

    /// 네이티브 전용 블로킹 래퍼. wasm에는 존재하지 않는다(블로킹 poll 불가).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self, InitError> {
        pollster::block_on(Self::new())
    }

    /// validation error scope로 감싸 셰이더/파이프라인 생성 오류를 Result로 변환.
    /// (드라이버가 침묵으로 무시하는 실패를 진단 가능한 에러로 바꾼다)
    ///
    /// wasm에서는 scope를 쓰지 않는다 — wgpu 30 웹 백엔드의 pop 경로가
    /// Validation/OOM 외 에러 타입에서 panic하는 업스트림 이슈(webgpu.rs:85).
    /// 웹 검증 에러는 브라우저 콘솔에 그대로 표시된다. (Phase 2: wgpu 패치/업스트림 추적)
    pub async fn with_validation<T>(
        &self,
        f: impl FnOnce(&wgpu::Device) -> T,
    ) -> Result<T, String> {
        #[cfg(target_arch = "wasm32")]
        {
            Ok(f(&self.device))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let out = f(&self.device);
            match guard.pop().await {
                None => Ok(out),
                Some(e) => Err(e.to_string()),
            }
        }
    }
}
