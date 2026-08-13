//! face — 얼굴 도메인: FaceTask(검출→ROI 트래킹→랜드마크) + 스무딩
//! (예정: 화장·터치업 기하, blendshape, 3D 피팅 — INTEGRATION.md P3)

pub mod smoothing;
pub mod task;

pub use task::{FaceResult, FaceTask};
