//! 손 태스크 — HandTask(팜 검출→ROI→랜드마크, 2손 트래킹) + 제스처 판정.
//! clap은 웹 로직의 실패 모드를 수리한 개선판 (gesture.rs 헤더 참조).
//! ROI 수학은 roi.rs — MediaPipe 출하 quirk 2개(90rad target, 서브셋 인덱스)를
//! 파리티 우선으로 보존한다.

pub mod gesture;
pub mod roi;
pub mod task;

pub use task::{HandResult, HandTask};
