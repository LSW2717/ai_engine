//! BatchNormalization → 선행 Conv에 폴딩.
//!
//! `s_c = γ_c / sqrt(σ²_c + ε)`; `W'[c,..] = W[c,..]·s_c`; `b'_c = (b_c − μ_c)·s_c + β_c`.
//! RVM/segm export에는 BN이 0개지만(이미 폴딩된 export) 범용 변환기의 필수 패스다.

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "BatchNormalization" {
            continue;
        }
        let bn = g.nodes[idx].clone();
        let x = &bn.inputs[0];
        let Some(conv_idx) = g.producer(x) else {
            return Err(ConvertError::Unsupported(vec![format!(
                "Conv 뒤가 아닌 BatchNormalization: {}",
                bn.name
            )]));
        };
        if g.nodes[conv_idx].op != "Conv" || g.consumers(x).len() != 1 {
            return Err(ConvertError::Unsupported(vec![format!(
                "폴딩 불가 BatchNormalization: {}",
                bn.name
            )]));
        }

        let get = |name: &str| -> Result<Vec<f32>, ConvertError> {
            g.info(name)
                .and_then(|t| t.as_f32s().map(|v| v.to_vec()))
                .ok_or_else(|| ConvertError::Malformed(format!("BN 파라미터 비상수: {name}")))
        };
        let gamma = get(&bn.inputs[1])?;
        let beta = get(&bn.inputs[2])?;
        let mean = get(&bn.inputs[3])?;
        let var = get(&bn.inputs[4])?;
        let eps = bn.attr_f("epsilon").unwrap_or(1e-5);

        let conv = g.nodes[conv_idx].clone();
        let w_name = conv.inputs[1].clone();
        let w_info = g.info(&w_name).cloned().ok_or_else(|| {
            ConvertError::Malformed(format!("Conv 가중치 없음: {w_name}"))
        })?;
        let w = w_info
            .as_f32s()
            .ok_or_else(|| ConvertError::Malformed(format!("Conv 가중치 비상수: {w_name}")))?
            .to_vec();
        let w_shape = w_info.shape.clone().unwrap();
        let cout = w_shape[0] as usize;
        let per_out = w.len() / cout;

        let bias: Vec<f32> = if let Some(b_name) = conv.inputs.get(2) {
            get(b_name)?
        } else {
            vec![0.0; cout]
        };

        let mut w2 = w;
        let mut b2 = vec![0f32; cout];
        for c in 0..cout {
            let s = gamma[c] / (var[c] + eps).sqrt();
            for k in 0..per_out {
                w2[c * per_out + k] *= s;
            }
            b2[c] = (bias[c] - mean[c]) * s + beta[c];
        }

        let w2_name = format!("{w_name}__bnfold");
        let b2_name = format!("{w_name}__bnfold_bias");
        g.add_const(
            w2_name.clone(),
            TensorInfo {
                shape: Some(w_shape),
                dtype: OnnxDtype::F32,
                data: Some(Arc::new(bytemuck::cast_slice(&w2).to_vec())),
            },
        );
        g.add_const(
            b2_name.clone(),
            TensorInfo {
                shape: Some(vec![cout as i64]),
                dtype: OnnxDtype::F32,
                data: Some(Arc::new(bytemuck::cast_slice(&b2).to_vec())),
            },
        );

        let cn = &mut g.nodes[conv_idx];
        cn.inputs[1] = w2_name;
        if cn.inputs.len() >= 3 {
            cn.inputs[2] = b2_name;
        } else {
            cn.inputs.push(b2_name);
        }
        let bn_out = bn.outputs[0].clone();
        let conv_out = conv.outputs[0].clone();
        g.make_alias(idx, &conv_out, &bn_out);
        report.rewrites += 1;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Attr, Node};
    use ai_core::rng::XorShift32;

    /// 골든 수학 테스트: conv+BN 수동 계산과 폴딩 결과 일치
    #[test]
    fn bn_fold_math_golden() {
        let mut g = Graph::default();
        let mut rng = XorShift32::new(9);
        let w = rng.vec_f32(4 * 2); // cout=4, cin=2, 1x1
        let mk = |v: &[f32], shape: Vec<i64>| TensorInfo {
            shape: Some(shape),
            dtype: OnnxDtype::F32,
            data: Some(Arc::new(bytemuck::cast_slice(v).to_vec())),
        };
        g.add_const("w", mk(&w, vec![4, 2, 1, 1]));
        g.add_const("gamma", mk(&[1.0, 2.0, 0.5, 1.5], vec![4]));
        g.add_const("beta", mk(&[0.1, -0.2, 0.0, 0.3], vec![4]));
        g.add_const("mean", mk(&[0.5, 0.0, -0.5, 1.0], vec![4]));
        g.add_const("var", mk(&[1.0, 4.0, 0.25, 9.0], vec![4]));
        g.nodes.push(Node {
            op: "Conv".into(),
            name: "c".into(),
            attrs: Default::default(),
            inputs: vec!["x".into(), "w".into()],
            outputs: vec!["cy".into()],
            dead: false,
        });
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("epsilon".to_string(), Attr::F(0.0));
        g.nodes.push(Node {
            op: "BatchNormalization".into(),
            name: "bn".into(),
            attrs,
            inputs: vec!["cy".into(), "gamma".into(), "beta".into(), "mean".into(), "var".into()],
            outputs: vec!["y".into()],
            dead: false,
        });
        g.outputs = vec!["y".into()];

        run(&mut g, &Ctx::default()).unwrap();

        // s = [1, 1, 1, 0.5]; W'[3] = 0.5*W[3]; b' = (0-mean)*s + beta
        let w2 = g.info("w__bnfold").unwrap().as_f32s().unwrap().to_vec();
        assert_eq!(w2[0], w[0]);
        assert!((w2[6] - w[6] * 0.5 / 0.5 * 0.5).abs() < 1e-6); // c=3: s=1.5/3=0.5
        let b2 = g.info("w__bnfold_bias").unwrap().as_f32s().unwrap().to_vec();
        assert!((b2[0] - (-0.5 + 0.1)).abs() < 1e-6); // (0-0.5)*1 + 0.1
        assert!((b2[3] - ((0.0 - 1.0) * 0.5 + 0.3)).abs() < 1e-6);
        // BN 노드 죽고 conv가 y의 별칭 원천
        assert!(g.nodes[1].dead);
        assert_eq!(g.resolve_alias("y"), "cy");
    }
}
