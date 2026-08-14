//! 게이즈(집중도) 태스크 — 웹 focus-tracker 이식.
//! 크롭 규약: 478 랜드마크 bbox + margin(0.18/0.22 ×bbox), **비회전·종횡비 무시**
//! 448² (FaceTask의 MediaPipe 회전 크롭과 다르다 — L2CS 학습 규약).

pub mod one_euro;
pub mod preprocess;
pub mod state;
pub mod task;

pub use state::{FocusResult, FocusStatus, MonitorInfo, ScreenLayout};
pub use task::GazeTask;
