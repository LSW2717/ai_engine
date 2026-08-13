//! ai-tasks — **공개 API 본체**. 태스크 단위(세그멘테이션/랜드마크/노이즈 억제)와
//! 그 전·후처리, 합성, 폴백 정책이 여기 산다.
//!
//! `ai-wasm`(웹)과 `ai-ffi`(모바일)는 이 크레이트를 **타입 변환만** 해서 노출한다.
//! 바인딩 크레이트에 분기(`if`)가 생겼다면 그건 로직이고 여기로 내려와야 한다 —
//! 그 규칙이 깨지는 순간 웹과 모바일 동작이 갈라진다. 실제로 지금 랜드마크가
//! 웹(mediapipe-wasm)과 모바일(vcxrust_ai/ncnn) 두 벌로 갈라져 있고, 그 실수를
//! 한 층 아래에서 반복하지 않으려고 이 크레이트를 둔다.
//!
//! 플랫폼마다 **진짜** 다른 것만 바인딩에 남긴다:
//! 서피스 획득, 프레임 임포트, 스레드 모델, 모델 바이트 조달.

pub mod clock;
pub mod composite;
pub mod error;
pub mod segmenter;
pub mod segmenter_cpu;

pub use clock::Stats;
pub use composite::{CompositeOpts, Compositor};
pub use error::TaskError;
pub use segmenter::Segmenter;
pub use segmenter_cpu::CpuSegmenter;
