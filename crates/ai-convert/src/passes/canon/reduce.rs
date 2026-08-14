//! ReduceMean 정규화 — axes[2,3]→gpool, axes[1]→cout=1 pointwise Conv 합성
//! (RVM refiner의 채널 평균이 기존 커널로 표현된다).
//! ReduceSum — rank-3 [1,a,k] 마지막 축(작은 k) 합. rank-3은 평탄 채널벡터로
//! 내려가므로(lower desc_of) 쌍합 = 희소 가중치 pointwise Conv(cout=a, cin=a·k,
//! W[i, i·k+j]=1)로 합성한다 (blendshapes의 L2 norm Sum이 소비).

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{Attr, Graph, Node};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead || !matches!(g.nodes[idx].op.as_str(), "ReduceMean" | "ReduceSum") {
            idx += 1;
            continue;
        }
        let node = g.nodes[idx].clone();
        let keep = node.attr_i("keepdims").unwrap_or(1) == 1;
        let axes: Vec<i64> = node
            .attr_is("axes")
            .map(|a| a.to_vec())
            .or_else(|| {
                node.inputs
                    .get(1)
                    .and_then(|n| g.info(n))
                    .and_then(|t| t.as_i64s().map(|v| v.to_vec()))
            })
            .unwrap_or_default();
        let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
        let rank = x.len() as i64;
        let mut norm: Vec<i64> = axes.iter().map(|a| if *a < 0 { rank + a } else { *a }).collect();
        norm.sort_unstable();

        if !keep {
            return Err(ConvertError::Unsupported(vec![format!(
                "ReduceMean keepdims=0 ({})",
                node.name
            )]));
        }

        if node.op == "ReduceSum" {
            // rank-3 [1,a,k] 마지막 축 합 → 희소 pointwise Conv (평탄 채널벡터 위)
            let ok = x.len() == 3 && x[0] == 1 && norm == vec![2] && x[2] <= 8;
            if !ok {
                return Err(ConvertError::Unsupported(vec![format!(
                    "ReduceSum shape={x:?} axes={norm:?} ({})",
                    node.name
                )]));
            }
            let (a, k) = (x[1] as usize, x[2] as usize);
            let mut w = vec![0f32; a * a * k];
            for i in 0..a {
                for j in 0..k {
                    w[i * (a * k) + i * k + j] = 1.0;
                }
            }
            let w_name = format!("{}__pairsum_w", node.outputs[0]);
            g.add_const(
                w_name.clone(),
                TensorInfo {
                    shape: Some(vec![a as i64, (a * k) as i64, 1, 1]),
                    dtype: OnnxDtype::F32,
                    data: Some(Arc::new(bytemuck::cast_slice(&w).to_vec())),
                },
            );
            let n = &mut g.nodes[idx];
            n.op = "Conv".into();
            n.inputs = vec![n.inputs[0].clone(), w_name];
            n.attrs.clear();
            n.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
            n.attrs.insert("group".into(), Attr::I(1));
            report.rewrites += 1;
            idx += 1;
            continue;
        }

        if x.len() == 3 && x[0] == 1 && norm == vec![1] {
            // rank-3 [1,a,k] 중간축 평균 (좌표별 평균점 등) — 평탄 채널벡터 위에서
            // 희소 pointwise Conv: cout=k, cin=a·k, W[j, i·k+j] = 1/a
            let (a, k) = (x[1] as usize, x[2] as usize);
            if k > 8 {
                return Err(ConvertError::Unsupported(vec![format!(
                    "ReduceMean rank3 k={k} ({})",
                    node.name
                )]));
            }
            let scale = 1.0 / a as f32;
            let mut w = vec![0f32; k * a * k];
            for j in 0..k {
                for i in 0..a {
                    w[j * (a * k) + i * k + j] = scale;
                }
            }
            let w_name = format!("{}__midmean_w", node.outputs[0]);
            g.add_const(
                w_name.clone(),
                TensorInfo {
                    shape: Some(vec![k as i64, (a * k) as i64, 1, 1]),
                    dtype: OnnxDtype::F32,
                    data: Some(Arc::new(bytemuck::cast_slice(&w).to_vec())),
                },
            );
            let n = &mut g.nodes[idx];
            n.op = "Conv".into();
            n.inputs = vec![n.inputs[0].clone(), w_name];
            n.attrs.clear();
            n.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
            n.attrs.insert("group".into(), Attr::I(1));
            report.rewrites += 1;
            idx += 1;
            continue;
        }

        if norm == vec![2, 3] {
            let n = &mut g.nodes[idx];
            n.op = "gpool".into();
            n.inputs.truncate(1);
            n.attrs.clear();
            report.rewrites += 1;
        } else if norm == vec![1] && x.len() == 4 && x[2] == 1 {
            // h==1 채널 평균: conv 출력 [1,1,1,W]가 채널벡터로 접혀(lower desc_of)
            // conv 지오메트리와 어긋난다 → transpose(w↔c) + gpool(공간 평균)
            let node2 = g.nodes[idx].clone();
            let tname = format!("{}__t", node2.outputs[0]);
            g.tensors.insert(
                tname.clone(),
                TensorInfo {
                    shape: Some(vec![x[0], x[3], 1, x[1]]),
                    dtype: OnnxDtype::F32,
                    data: None,
                },
            );
            g.nodes.insert(
                idx,
                Node {
                    op: "transpose".into(),
                    name: format!("{}#t", node2.name),
                    attrs: Default::default(),
                    inputs: vec![node2.inputs[0].clone()],
                    outputs: vec![tname.clone()],
                    dead: false,
                },
            );
            let n = &mut g.nodes[idx + 1];
            n.op = "gpool".into();
            n.inputs = vec![tname];
            n.attrs.clear();
            report.rewrites += 1;
            idx += 2;
            continue;
        } else if norm == vec![1] {
            // 채널 평균 = 1/C 가중치 pointwise conv
            let cin = x[1];
            let w: Vec<f32> = vec![1.0 / cin as f32; cin as usize];
            let w_name = format!("{}__chmean_w", node.outputs[0]);
            g.add_const(
                w_name.clone(),
                TensorInfo {
                    shape: Some(vec![1, cin, 1, 1]),
                    dtype: OnnxDtype::F32,
                    data: Some(Arc::new(bytemuck::cast_slice(&w).to_vec())),
                },
            );
            let n = &mut g.nodes[idx];
            n.op = "Conv".into();
            n.inputs = vec![n.inputs[0].clone(), w_name];
            n.attrs.clear();
            n.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
            n.attrs.insert("group".into(), Attr::I(1));
            report.rewrites += 1;
        } else {
            return Err(ConvertError::Unsupported(vec![format!(
                "ReduceMean axes={norm:?} ({})",
                node.name
            )]));
        }
        idx += 1;
    }
    Ok(report)
}
