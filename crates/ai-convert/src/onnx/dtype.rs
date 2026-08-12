//! ONNX TensorProto 디코딩 — elem_type 매핑과 raw/typed 데이터 통합.

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::onnx::proto;

pub fn map_elem_type(t: i32) -> OnnxDtype {
    // onnx.TensorProto.DataType
    match t {
        1 => OnnxDtype::F32,
        7 => OnnxDtype::I64,
        6 => OnnxDtype::I32,
        10 => OnnxDtype::F16,
        9 => OnnxDtype::Bool,
        other => OnnxDtype::Unknown(other),
    }
}

/// TensorProto → TensorInfo (데이터 포함). external data는 거부.
pub fn decode_tensor(t: &proto::TensorProto) -> Result<TensorInfo, ConvertError> {
    if t.data_location() == proto::tensor_proto::DataLocation::External {
        return Err(ConvertError::Unsupported(vec![format!(
            "external data 이니셜라이저: {}",
            t.name()
        )]));
    }
    let dtype = map_elem_type(t.data_type());
    let shape: Vec<i64> = t.dims.clone();
    let elems: usize = shape.iter().product::<i64>().max(1) as usize;

    let data: Vec<u8> = if !t.raw_data().is_empty() {
        t.raw_data().to_vec()
    } else {
        // typed 필드 폴백 (proto2에서 소형 상수에 흔함)
        match dtype {
            OnnxDtype::F32 => bytemuck::cast_slice(&t.float_data).to_vec(),
            OnnxDtype::I64 => bytemuck::cast_slice(&t.int64_data).to_vec(),
            OnnxDtype::I32 => bytemuck::cast_slice(&t.int32_data).to_vec(),
            OnnxDtype::F16 => {
                // f16은 int32_data에 하위 16비트로 저장됨
                let h: Vec<u16> = t.int32_data.iter().map(|v| *v as u16).collect();
                bytemuck::cast_slice(&h).to_vec()
            }
            _ => Vec::new(),
        }
    };

    if !data.is_empty() {
        let expect = elems * dtype.byte_size().unwrap_or(1);
        if data.len() != expect {
            return Err(ConvertError::Malformed(format!(
                "이니셜라이저 {} 크기 불일치: {} != {expect}",
                t.name(),
                data.len()
            )));
        }
    }

    Ok(TensorInfo { shape: Some(shape), dtype, data: Some(Arc::new(data)) })
}
