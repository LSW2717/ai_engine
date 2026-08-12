//! prost 생성 ONNX 타입 (build.rs가 OUT_DIR에 생성)

#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
