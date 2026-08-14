//! audio — 오디오 노이즈 제거 (fastenhancer, 로드맵 #6).
//!
//! 구조: stft.rs(스트리밍 STFT/iSTFT — vcx-noise 이식) + graph.rs/ops.rs
//! (spec2spec 서브그래프 미니 실행기 — .sw가 아닌 이유는 graph.rs 헤더) +
//! enhancer.rs(스트리밍 상태 + 압축/복소 마스크/역압축).
//! 모델 산출: tools/prep_fastenhancer.py (ONNX 수술 + --verify + --export).
//! 오디오는 **CPU 고정** — AudioWorklet에 WebGPU가 없다 (워커 토폴로지 규약).

pub mod enhancer;
pub mod graph;
pub mod ops;
pub mod stft;

pub use enhancer::Enhancer;
