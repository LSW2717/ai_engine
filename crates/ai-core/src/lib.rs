//! ai-core — 엔진의 GPU 무의존 코어.
//!
//! 텐서 기술자(NHWC-C4 레이아웃 수학), 텐서/가중치 패킹, 활성화 정의,
//! CPU 레퍼런스 구현(정확도 오라클), 그리고 후속 Phase의 그래프 IR·컨테이너
//! 포맷 타입이 여기에 산다. wgpu에 의존하지 않으므로 변환기(ai-convert)와
//! 런타임이 같은 계약을 공유한다.

pub mod activation;
pub mod format;
pub mod ops;
pub mod pack;
pub mod reference;
pub mod rng;
pub mod tensor;

pub use activation::Activation;
pub use tensor::{DType, TensorDesc};
