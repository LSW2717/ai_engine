//! CPU 커널 계열 — 파일 하나 = 커널 하나, 각자 레퍼런스 대조 테스트를 소유한다
//! (ai-gpu/src/kernels와 같은 규율).

pub mod conv;
pub mod dw;
pub mod elementwise;
pub mod pool;
pub mod resize;
pub mod segate;
pub mod shape;
