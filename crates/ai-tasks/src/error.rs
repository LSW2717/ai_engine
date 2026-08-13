//! 태스크 레이어 오류 — **호스트가 폴백 티어를 결정하는 근거**라서 구조화한다.
//!
//! 문자열 하나로 뭉개면 바인딩마다 다르게 해석하게 되고, 웹은 강등하는데
//! 모바일은 안 하는 식으로 갈라진다.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TaskError {
    /// GPU 자체를 못 잡았다 — CPU 백엔드로 강등해야 하는 신호
    #[error("GPU 초기화 실패: {0}")]
    NoGpu(String),
    /// 실행 중 디바이스를 잃었다 (탭 백그라운드·드라이버 리셋·GPU 프로세스 크래시)
    #[error("디바이스 유실: {0}")]
    DeviceLost(String),
    /// 모델 로드/실행 실패
    #[error("런타임: {0}")]
    Runtime(#[from] ai_gpu_runtime::RuntimeError),
    /// GPU 계층 오류 (컴파일·리드백 등)
    #[error("GPU: {0}")]
    Gpu(String),
    /// CPU 폴백 계층 오류 — 이것마저 실패하면 호스트는 "호환 안 됨" 판정
    #[error("CPU: {0}")]
    Cpu(String),
    #[error("{0}")]
    Other(String),
}
