//! 행 브로드캐스트 Binary 정규화 (face_blendshapes LayerNorm 그리드 경로) —
//! `Binary(a [1,1,H,W], b 원소수 H)` : 행(H)별 스칼라를 W 방향 전체에.
//!
//! a의 desc가 (H·W px, c=1)이라 픽셀벡터 모드가 못 잡는다 → a를 [1,W,H,1]
//! (desc (H,1,W) = H픽셀×W채널, 논리 스트림 항등)로 relayout해 감싸면
//! b(len H)가 픽셀벡터 브로드캐스트로 정확히 떨어진다. 결과를 원 desc로 복원.

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{Graph, Node};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead || !matches!(g.nodes[idx].op.as_str(), "Mul" | "Add" | "Sub") {
            idx += 1;
            continue;
        }
        let node = g.nodes[idx].clone();
        if node.inputs.len() < 2 {
            idx += 1;
            continue;
        }
        let shape = |n: &str| g.info(n).and_then(|t| t.static_shape().map(|s| s.to_vec()));
        let (Some(s0), Some(s1)) = (shape(&node.inputs[0]), shape(&node.inputs[1])) else {
            idx += 1;
            continue;
        };
        // a = [1,1,H,W] 그리드 (H,W > 1), b = 원소수 H (게다가 a와 다른 텐서)
        let grid = |s: &[i64]| {
            (s.len() == 4 && s[0] == 1 && s[1] == 1 && s[2] > 1 && s[3] > 1)
                .then(|| (s[2], s[3]))
        };
        let n_of = |s: &[i64]| -> i64 { s.iter().product() };
        let (a_i, b_i, h, w) = match (grid(&s0), grid(&s1)) {
            (Some((h, w)), _) if n_of(&s1) == h && n_of(&s1) != n_of(&s0) => (0usize, 1usize, h, w),
            (_, Some((h, w)))
                if n_of(&s0) == h
                    && n_of(&s0) != n_of(&s1)
                    && matches!(node.op.as_str(), "Mul" | "Add") =>
            {
                (1, 0, h, w)
            }
            _ => {
                idx += 1;
                continue;
            }
        };
        let (a_name, b_name) = (node.inputs[a_i].clone(), node.inputs[b_i].clone());
        let out = node.outputs[0].clone();

        // ① a → [1,W,H,1] (desc (H,1,W)) relayout
        let a2 = format!("{out}__rowv_a");
        g.tensors.insert(
            a2.clone(),
            TensorInfo { shape: Some(vec![1, w, h, 1]), dtype: OnnxDtype::F32, data: None },
        );
        g.nodes.insert(
            idx,
            Node {
                op: "relayout".into(),
                name: format!("{}#rowv_a", node.name),
                attrs: Default::default(),
                inputs: vec![a_name],
                outputs: vec![a2.clone()],
                dead: false,
            },
        );
        // ② binary (a' ∘ b) → o' [1,W,H,1]
        let o2 = format!("{out}__rowv_o");
        g.tensors.insert(
            o2.clone(),
            TensorInfo { shape: Some(vec![1, w, h, 1]), dtype: OnnxDtype::F32, data: None },
        );
        {
            let n = &mut g.nodes[idx + 1];
            n.inputs = vec![a2, b_name];
            n.outputs = vec![o2.clone()];
        }
        // ③ o' → 원 출력 relayout (논리 항등)
        g.nodes.insert(
            idx + 2,
            Node {
                op: "relayout".into(),
                name: format!("{}#rowv_r", node.name),
                attrs: Default::default(),
                inputs: vec![o2],
                outputs: vec![out],
                dead: false,
            },
        );
        report.rewrites += 1;
        idx += 3;
    }
    Ok(report)
}
