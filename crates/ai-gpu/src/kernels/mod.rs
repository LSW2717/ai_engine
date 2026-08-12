//! 커널 레지스트리 — 커널당 `.rs` 파일 하나 ↔ `shaders/`에 `.wgsl` 템플릿 하나.
//! 파일 규약(Spec + 변형 정책 + KernelSpec impl + naga 테스트)은 ARCHITECTURE.md 참조.

pub mod avgpool;
pub mod channel_gather;
pub mod common;
pub mod conv_dw;
pub mod conv_igemm;
pub mod elementwise;
pub mod gemm_pw;
pub mod gpool;
pub mod resize_bilinear;
pub mod se_gate;
