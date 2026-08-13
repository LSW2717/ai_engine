//! Gemm → 1x1 Conv 재작성 — 가중치가 initializer라 재배열이 변환 시점에 공짜.
//!
//! Gemm(A[1,K], W, bias) = A를 (1,1,K) 픽셀 하나로 보고 K→N pointwise conv.
//! transB=1이면 W[N,K] = OIHW [N,K,1,1] 그대로, transB=0이면 [K,N]을 전치.
//! (gaze MobileOne 헤드 transB=1, hand_landmarks MatMul-융합 Gemm transB=0.)

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::graph::Attr;
use crate::ir::{Graph, TensorInfo};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Gemm" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let alpha = node.attr_f("alpha").unwrap_or(1.0);
        let beta = node.attr_f("beta").unwrap_or(1.0);
        if alpha != 1.0 || beta != 1.0 || node.attr_i("transA").unwrap_or(0) == 1 {
            return Err(ConvertError::Unsupported(vec![format!(
                "Gemm alpha/beta/transA 변형 ({})",
                node.name
            )]));
        }
        let tb = node.attr_i("transB").unwrap_or(0) == 1;
        let w_name = node.inputs[1].clone();
        let wi = g
            .info(&w_name)
            .filter(|t| t.is_const())
            .ok_or_else(|| {
                ConvertError::Unsupported(vec![format!("Gemm 비상수 가중치 ({})", node.name)])
            })?
            .clone();
        let ws = wi.static_shape().unwrap().to_vec();
        let data = wi.as_f32s().unwrap();
        let (n, k) = if tb { (ws[0], ws[1]) } else { (ws[1], ws[0]) };
        let oihw: Vec<f32> = if tb {
            data.to_vec() // [N,K] 그대로
        } else {
            let (kk, nn) = (ws[0] as usize, ws[1] as usize);
            let mut t = vec![0f32; kk * nn];
            for r in 0..kk {
                for c in 0..nn {
                    t[c * kk + r] = data[r * nn + c];
                }
            }
            t
        };
        let new_w = format!("{w_name}__gemm_oihw");
        g.add_const(
            new_w.clone(),
            TensorInfo {
                shape: Some(vec![n, k, 1, 1]),
                dtype: wi.dtype,
                data: Some(Arc::new(bytemuck::cast_slice(&oihw).to_vec())),
            },
        );
        let nm = &mut g.nodes[idx];
        nm.op = "Conv".into();
        nm.inputs[1] = new_w;
        nm.attrs.clear();
        nm.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
        nm.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
        nm.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
        nm.attrs.insert("dilations".into(), Attr::Is(vec![1, 1]));
        nm.attrs.insert("group".into(), Attr::I(1));
        report.rewrites += 1;
    }
    Ok(report)
}
