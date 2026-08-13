//! ai-gpu-runtime — .sw 그래프 executor.
//!
//! 로드: 컨테이너 파싱 → 가중치 단일 버퍼 업로드(memcpy) → 뷰 실체화 계획 →
//! liveness 슬롯 배정 → 병렬 파이프라인 컴파일 → bind group 사전 생성(even/odd) → 워밍업.
//! 추론: 인코더 1개, 단일 컴퓨트 패스, 상태 ping-pong은 리스트 교대 — 프레임 루프 할당 0.

pub mod compile;
pub mod error;
pub mod lowering;
pub mod model;
pub mod plan;

pub use error::RuntimeError;
pub use model::Model;
