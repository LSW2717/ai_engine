//! 그래프 IR의 op 타입들 — op별 파일 하나.
//!
//! Phase 1에서는 CPU 레퍼런스·커널 spec이 공유하는 속성 구조체로 쓰이고,
//! Phase 2에서 컨테이너 포맷의 그래프 노드가 이 타입들을 직렬화한다.

pub mod conv;
pub mod elementwise;
pub mod pool;
pub mod resize;
pub mod shape;

pub use conv::Conv2d;
pub use elementwise::BinaryOp;
pub use pool::AvgPool2d;
pub use resize::{CoordMode, ResizeBilinear};
