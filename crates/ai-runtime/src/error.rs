//! 런타임 오류.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("컨테이너: {0}")]
    Format(#[from] ai_core::format::FormatError),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error("{0}")]
    Other(String),
}
