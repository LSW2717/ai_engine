//! ai-convert — ONNX → .sw 컨테이너 변환기 (native 전용).
//!
//! 오프라인에서 무거운 일을 전부 처리한다: BN 폴딩, 융합 마킹, NHWC-C4 사전 패킹.
//! 런타임 로드는 memcpy가 된다. 패킹 레이아웃의 단일 진실 원천은 ai-core::pack.

pub mod cli;
pub mod emit;
pub mod error;
pub mod ir;
pub mod onnx;
pub mod passes;
pub mod plan;
pub mod verify;
