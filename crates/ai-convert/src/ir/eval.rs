//! 상수 평가 — 그래프의 "shape 배관"(Shape/Slice/Concat/Gather/산술)을 컴파일 타임에
//! 값으로 접는다. RVM의 Shape 12개, Resize scale 공급망, Split 파라미터가 전부
//! 여기서 정적으로 해소된다.

use std::sync::Arc;

use crate::ir::graph::{Graph, Node};
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};

fn const_of(g: &Graph, name: &str) -> Option<TensorInfo> {
    g.info(name).filter(|t| t.is_const()).cloned()
}

fn i64_tensor(vals: Vec<i64>, shape: Vec<i64>) -> TensorInfo {
    TensorInfo {
        shape: Some(shape),
        dtype: OnnxDtype::I64,
        data: Some(Arc::new(bytemuck::cast_slice(&vals).to_vec())),
    }
}

fn f32_tensor(vals: Vec<f32>, shape: Vec<i64>) -> TensorInfo {
    TensorInfo {
        shape: Some(shape),
        dtype: OnnxDtype::F32,
        data: Some(Arc::new(bytemuck::cast_slice(&vals).to_vec())),
    }
}

/// 노드를 상수로 평가 시도. 성공 시 출력별 TensorInfo 반환.
/// (Shape은 입력이 상수가 아니어도 shape만 정적이면 평가 가능)
pub fn try_eval(g: &Graph, node: &Node) -> Option<Vec<TensorInfo>> {
    match node.op.as_str() {
        "Shape" => {
            let shape = g.info(&node.inputs[0])?.static_shape()?.to_vec();
            let n = shape.len() as i64;
            Some(vec![i64_tensor(shape, vec![n])])
        }
        "Cast" => {
            let t = const_of(g, &node.inputs[0])?;
            let to = node.attr_i("to")?;
            let out = match (t.dtype, to) {
                (OnnxDtype::I64, 1) => {
                    f32_tensor(t.as_i64s()?.iter().map(|v| *v as f32).collect(), t.shape.clone()?)
                }
                (OnnxDtype::F32, 7) => {
                    i64_tensor(t.as_f32s()?.iter().map(|v| *v as i64).collect(), t.shape.clone()?)
                }
                (OnnxDtype::I64, 7) | (OnnxDtype::F32, 1) => t.clone(),
                _ => return None,
            };
            Some(vec![out])
        }
        "Concat" => {
            // 1-D 텐서 연결 (shape 배관 + Resize scale)
            let parts: Option<Vec<TensorInfo>> =
                node.inputs.iter().map(|i| const_of(g, i)).collect();
            let parts = parts?;
            if parts.iter().any(|p| p.shape.as_deref().map(|s| s.len()) != Some(1)) {
                return None;
            }
            let dtype = parts[0].dtype;
            match dtype {
                OnnxDtype::I64 => {
                    let mut v = Vec::new();
                    for p in &parts {
                        v.extend_from_slice(p.as_i64s()?);
                    }
                    let n = v.len() as i64;
                    Some(vec![i64_tensor(v, vec![n])])
                }
                OnnxDtype::F32 => {
                    let mut v = Vec::new();
                    for p in &parts {
                        v.extend_from_slice(p.as_f32s()?);
                    }
                    let n = v.len() as i64;
                    Some(vec![f32_tensor(v, vec![n])])
                }
                _ => None,
            }
        }
        "Slice" => {
            // 1-D 상수 슬라이스 (shape 배관)
            let data = const_of(g, &node.inputs[0])?;
            if data.shape.as_deref().map(|s| s.len()) != Some(1) {
                return None;
            }
            let starts = const_of(g, node.inputs.get(1)?)?.as_i64s()?.to_vec();
            let ends = const_of(g, node.inputs.get(2)?)?.as_i64s()?.to_vec();
            let len = data.shape.as_ref()?[0];
            let norm = |v: i64| v.clamp(-len, len).rem_euclid(len.max(1)).min(len);
            let (s, e) = (norm(starts[0]), if ends[0] >= len { len } else { norm(ends[0]) });
            match data.dtype {
                OnnxDtype::I64 => {
                    let v = data.as_i64s()?[s as usize..e.max(s) as usize].to_vec();
                    let n = v.len() as i64;
                    Some(vec![i64_tensor(v, vec![n])])
                }
                OnnxDtype::F32 => {
                    let v = data.as_f32s()?[s as usize..e.max(s) as usize].to_vec();
                    let n = v.len() as i64;
                    Some(vec![f32_tensor(v, vec![n])])
                }
                _ => None,
            }
        }
        "Gather" => {
            let data = const_of(g, &node.inputs[0])?;
            let idx = const_of(g, &node.inputs[1])?;
            if data.shape.as_deref().map(|s| s.len()) != Some(1) {
                return None;
            }
            let ids = idx.as_i64s()?;
            let out_shape = idx.shape.clone()?;
            match data.dtype {
                OnnxDtype::I64 => {
                    let d = data.as_i64s()?;
                    let v: Vec<i64> = ids.iter().map(|i| d[*i as usize]).collect();
                    Some(vec![i64_tensor(v, out_shape)])
                }
                _ => None,
            }
        }
        "Squeeze" | "Unsqueeze" => {
            let t = const_of(g, &node.inputs[0])?;
            // 데이터는 동일, shape만 변경 — 1-D 배관에선 shape 재계산이 단순
            let mut shape = t.shape.clone()?;
            if node.op == "Unsqueeze" {
                let axes = node
                    .attr_is("axes")
                    .map(|a| a.to_vec())
                    .or_else(|| Some(const_of(g, node.inputs.get(1)?)?.as_i64s()?.to_vec()))?;
                for &a in axes.iter() {
                    let a = if a < 0 { (shape.len() as i64 + 1 + a) as usize } else { a as usize };
                    shape.insert(a, 1);
                }
            } else {
                shape.retain(|d| *d != 1);
            }
            Some(vec![TensorInfo { shape: Some(shape), ..t.clone() }])
        }
        "Mul" | "Add" | "Sub" | "Div" => {
            let a = const_of(g, &node.inputs[0])?;
            let b = const_of(g, &node.inputs[1])?;
            // 스칼라/동형 브로드캐스트만 (배관 용도)
            let f = |x: f32, y: f32| match node.op.as_str() {
                "Mul" => x * y,
                "Add" => x + y,
                "Sub" => x - y,
                _ => x / y,
            };
            let (av, bv) = (a.as_f32s()?, b.as_f32s()?);
            let n = av.len().max(bv.len());
            if av.len() != n && av.len() != 1 || bv.len() != n && bv.len() != 1 {
                return None;
            }
            let v: Vec<f32> = (0..n)
                .map(|i| f(av[i % av.len()], bv[i % bv.len()]))
                .collect();
            let shape = if av.len() >= bv.len() { a.shape.clone()? } else { b.shape.clone()? };
            Some(vec![f32_tensor(v, shape)])
        }
        _ => None,
    }
}
