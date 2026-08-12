//! CPU 레퍼런스 구현 — GPU 커널의 정확도 오라클.
//!
//! 전부 논리 NHWC(`[h][w][c]`) f32, 순진한 루프. 성능은 목적이 아니다.
//! 각 함수의 에필로그 규약은 GPU 커널과 동일: bias → activation → residual(활성화 후) 순.

pub mod conv;
pub mod elementwise;
pub mod pool;
pub mod resize;
