//! ReduceMean 정규화 — axes[2,3]→gpool, axes[1]→cout=1 pointwise Conv 합성
//! (RVM refiner의 채널 평균이 기존 커널로 표현된다).

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "ReduceMean" {
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

        if norm == vec![2, 3] {
            let n = &mut g.nodes[idx];
            n.op = "gpool".into();
            n.inputs.truncate(1);
            n.attrs.clear();
            report.rewrites += 1;
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
    }
    Ok(report)
}
