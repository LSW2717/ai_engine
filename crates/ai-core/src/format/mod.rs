//! .sw 컨테이너 포맷 — 변환기(ai-convert)와 런타임(ai-gpu-runtime)의 공유 계약.
//!
//! 단일 파일: 헤더(매직 "SW1\0") + JSON 그래프 + 256정렬 가중치 블롭.
//! 텐서별 dtype 태그로 f32/f16(향후 int8) 수용, 상태 텐서 쌍으로 순환 모델 지원,
//! h=1 규약으로 conv1d/GRU(향후 CPU 백엔드) 확장 여지를 남긴다.
//! 변환기는 webgl2 엔진 호환(plan.json + weights.bin) 출력도 지원 예정(웹 폴백 티어).

pub mod header;
pub mod model;

pub use header::{parse_container, write_container, FormatError, BLOB_ALIGN, MAGIC, VERSION};
pub use model::{
    SeFc,
    SwAlias, SwConcatPart, SwModel, SwOp, SwOperand, SwSize, SwState, SwTensor, WRef,
};
