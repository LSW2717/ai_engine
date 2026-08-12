//! 변환기 오류 — 사용자가 조치할 수 있는 메시지를 원칙으로 한다.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConvertError {
    #[error("파일 I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("ONNX 프로토 디코드 실패: {0}")]
    Proto(#[from] prost::DecodeError),
    #[error("ONNX 형식 오류: {0}")]
    Malformed(String),
    #[error("미지원 요소:\n{}", .0.join("\n"))]
    Unsupported(Vec<String>),
    #[error("shape 미해석: {0} — --size/--set-input으로 입력을 고정했는지 확인")]
    ShapeUnresolved(String),
    #[error("{0}")]
    Other(String),
}
