//! audio — 오디오 노이즈 제거 (fastenhancer, 로드맵 #6 · RNNoise — 기본 모드) + 음성 감지(Silero VAD).
//!
//! 구조: stft.rs(스트리밍 STFT/iSTFT — vcx-noise 이식) + graph.rs/ops.rs
//! (spec2spec 서브그래프 미니 실행기 — .sw가 아닌 이유는 graph.rs 헤더) +
//! enhancer.rs(스트리밍 상태 + 압축/복소 마스크/역압축).
//! denoiser.rs = RNNoise(nnnoiseless) — 가중치 내장, 48kHz 전용, 잡음 제거 기본 모드.
//! vad.rs = Silero VAD(v6.2) — 음성 확률 전용(정적 1-청크 그래프 + 16k 리샘플러).
//! 모델 산출: tools/prep_fastenhancer.py · prep_silero.py (ONNX 수술/재구성 + --verify + --export).
//! 오디오는 **CPU 고정** — AudioWorklet에 WebGPU가 없다 (워커 토폴로지 규약).

pub mod denoiser;
pub mod enhancer;
pub mod graph;
pub mod ops;
pub mod stft;
pub mod vad;

pub use denoiser::Denoiser;
pub use enhancer::Enhancer;
pub use vad::Vad;
