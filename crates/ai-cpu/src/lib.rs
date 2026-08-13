//! ai-cpu — SIMD(NEON/SIMD128) CPU 실행 계층. GPU 미지원·저사양 기기의 폴백.
//!
//! 구조는 ai-gpu와 대칭이다:
//! - `plan`: 로드 시 전부 — 가중치 재패킹, 슬롯 liveness 계획, SwOp → PlanOp
//! - `model`: 프레임 루프 — PlanOp 디스패치 (프레임 중 할당 0)
//! - `kernels/*`: 파일 하나 = 커널 하나 + 자체 레퍼런스 대조 테스트
//! - `simd`: 아키텍처 격리 (커널은 `core::arch`를 모른다)
//!
//! 새 op 추가 절차(ARCHITECTURE.md 참조): kernels/<이름>.rs + plan.rs lowering
//! arm 하나 — GPU의 "새 커널 추가 절차"와 대칭.

pub mod kernels;
pub mod simd;
pub mod view;

mod model;
mod plan;

pub use model::Model;
pub use model::StepProf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CpuError {
    #[error("컨테이너 파싱 실패: {0}")]
    Format(String),
    #[error("미지원: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Other(String),
}
