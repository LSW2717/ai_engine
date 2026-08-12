//! 변환기 내부 IR — 그래프, 텐서 메타, (P2-3에서) shape 추론·상수 평가.

pub mod eval;
pub mod graph;
pub mod shape;
pub mod tensor_info;

pub use graph::{Attr, Graph, Node};
pub use tensor_info::{OnnxDtype, TensorInfo};
