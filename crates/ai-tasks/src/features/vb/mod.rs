//! vb — 가상배경 파이프라인 (배경교체·블러·조명·프레이밍·밝기/흑백 = 세그
//! 마스크를 소비하는 스테이지들). 새 효과 추가 절차는 stages/mod.rs 헤더.

pub mod compositor; // (레거시) infer_and_present 전용 단순 합성기
pub mod params;
pub mod pipeline;
pub(crate) mod stages;

pub use params::{Background, EffectsPatch, EffectsState};
pub use pipeline::VideoPipeline;
