//! ONNX ModelProto → ir::Graph.
//!
//! - 이니셜라이저 → 상수 TensorInfo
//! - `Constant` 노드 → 즉시 상수화(노드 제거)
//! - 옵션 입력 자리의 빈 문자열("") 제거 (segm 모델 Resize의 roi/scales 생략 형태)
//! - value_info/입출력 shape 힌트 수집 (심볼릭 차원은 -1)

use prost::Message;

use crate::error::ConvertError;
use crate::ir::{Attr, Graph, Node, TensorInfo};
use crate::onnx::{dtype, proto};

pub struct Imported {
    pub graph: Graph,
    pub opset: i64,
}

pub fn import(bytes: &[u8]) -> Result<Imported, ConvertError> {
    let model = proto::ModelProto::decode(bytes)?;
    let opset = model
        .opset_import
        .iter()
        .find(|o| o.domain().is_empty())
        .map(|o| o.version())
        .unwrap_or(0);
    for o in &model.opset_import {
        if !o.domain().is_empty() {
            return Err(ConvertError::Unsupported(vec![format!(
                "커스텀 opset 도메인: {}",
                o.domain()
            )]));
        }
    }
    let g_proto = model
        .graph
        .ok_or_else(|| ConvertError::Malformed("graph 없음".into()))?;

    let mut g = Graph::default();

    // 이니셜라이저
    for t in &g_proto.initializer {
        g.tensors.insert(t.name().to_string(), dtype::decode_tensor(t)?);
    }

    // shape 힌트 (input/output/value_info)
    let mut record_vi = |vi: &proto::ValueInfoProto, g: &mut Graph| {
        if let Some(proto::type_proto::Value::TensorType(tt)) =
            vi.r#type.as_ref().and_then(|t| t.value.as_ref())
        {
            let info = g.info_mut(vi.name());
            if info.data.is_none() {
                info.dtype = dtype::map_elem_type(tt.elem_type());
                if let Some(shape) = &tt.shape {
                    let dims: Vec<i64> = shape
                        .dim
                        .iter()
                        .map(|d| match &d.value {
                            Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                            _ => -1, // 심볼릭
                        })
                        .collect();
                    info.shape = Some(dims);
                }
            }
        }
    };
    for vi in g_proto.input.iter().chain(&g_proto.output).chain(&g_proto.value_info) {
        record_vi(vi, &mut g);
    }

    // 그래프 입력 (이니셜라이저 제외)
    for vi in &g_proto.input {
        if !g.tensors.get(vi.name()).is_some_and(|t| t.is_const()) {
            g.inputs.push(vi.name().to_string());
        }
    }
    g.outputs = g_proto.output.iter().map(|o| o.name().to_string()).collect();

    // 노드
    for n in &g_proto.node {
        // Constant → 상수화
        if n.op_type() == "Constant" {
            let out = n.output.first().cloned().unwrap_or_default();
            let mut info: Option<TensorInfo> = None;
            for a in &n.attribute {
                match a.name() {
                    "value" => {
                        if let Some(t) = &a.t {
                            info = Some(dtype::decode_tensor(t)?);
                        }
                    }
                    "value_float" => {
                        info = Some(TensorInfo {
                            shape: Some(vec![]),
                            dtype: crate::ir::OnnxDtype::F32,
                            data: Some(std::sync::Arc::new(a.f().to_le_bytes().to_vec())),
                        })
                    }
                    "value_int" => {
                        info = Some(TensorInfo {
                            shape: Some(vec![]),
                            dtype: crate::ir::OnnxDtype::I64,
                            data: Some(std::sync::Arc::new(a.i().to_le_bytes().to_vec())),
                        })
                    }
                    other => {
                        return Err(ConvertError::Unsupported(vec![format!(
                            "Constant attr {other}"
                        )]))
                    }
                }
            }
            let info = info
                .ok_or_else(|| ConvertError::Malformed(format!("빈 Constant: {}", n.name())))?;
            g.add_const(out, info);
            continue;
        }

        let mut attrs = std::collections::HashMap::new();
        for a in &n.attribute {
            use proto::attribute_proto::AttributeType as AT;
            let v = match a.r#type() {
                AT::Int => Attr::I(a.i()),
                AT::Ints => Attr::Is(a.ints.clone()),
                AT::Float => Attr::F(a.f()),
                AT::Floats => Attr::Fs(a.floats.clone()),
                AT::String => Attr::S(String::from_utf8_lossy(a.s()).into_owned()),
                AT::Tensor => Attr::T(
                    a.t.as_ref()
                        .map(dtype::decode_tensor)
                        .transpose()?
                        .unwrap_or_default(),
                ),
                other => {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "attr 타입 {other:?} ({}.{})",
                        n.name(),
                        a.name()
                    )]))
                }
            };
            attrs.insert(a.name().to_string(), v);
        }

        g.nodes.push(Node {
            op: n.op_type().to_string(),
            name: n.name().to_string(),
            attrs,
            inputs: n.input.iter().filter(|s| !s.is_empty()).cloned().collect(),
            outputs: n.output.clone(),
            dead: false,
        });
    }

    Ok(Imported { graph: g, opset })
}
