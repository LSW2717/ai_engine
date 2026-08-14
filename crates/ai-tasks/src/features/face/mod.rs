//! face — 얼굴 도메인: FaceTask(검출→ROI 트래킹→랜드마크) + 스무딩 +
//! 터치업/메이크업 래스터라이즈 + 3D 아이템
//! (예정: blendshape 상시 스트림, Horn 피팅 — INTEGRATION.md P3)

pub mod blendshapes;
pub mod items3d;
pub mod makeup;
pub mod smoothing;
pub mod task;
pub mod touchup;

pub use task::{FaceResult, FaceTask};
