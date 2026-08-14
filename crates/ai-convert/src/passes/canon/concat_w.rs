//! W축 Concat 정규화 (h==1, 파트 채널 동일, c%4==0) — MLP-Mixer AddExtraTokens.
//!
//! (1,w,c)의 C4 레이아웃은 [w][cg]라, c%4==0이면 플랫 채널벡터 (1,1,w·c)와
//! **바이트 동일**. 그래서 W-concat = 플랫 벡터들의 채널 concat:
//!   각 파트 → relayout(플랫) → Concat(axis=1) → relayout(원 desc 복원)
//! 파트가 작아(수 KB) relayout 복사 비용은 무시 가능 — 정확성 우선.

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{Attr, Graph, Node};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Concat" {
            idx += 1;
            continue;
        }
        let node = g.nodes[idx].clone();
        let axis = node.attr_i("axis").unwrap_or(1);
        let shapes: Option<Vec<Vec<i64>>> = node
            .inputs
            .iter()
            .map(|i| g.info(i).and_then(|t| t.static_shape().map(|s| s.to_vec())))
            .collect();
        let Some(shapes) = shapes else {
            idx += 1;
            continue;
        };
        // 표적: rank4 axis=3(W), H==1, 전 파트 채널(s[1]) 동일 && %4==0
        let ok = axis == 3
            && shapes.iter().all(|s| s.len() == 4 && s[2] == 1)
            && shapes.windows(2).all(|w| w[0][1] == w[1][1])
            && shapes[0][1] % 4 == 0;
        if !ok {
            idx += 1;
            continue;
        }
        let c = shapes[0][1];

        // 파트 → 플랫 relayout 노드들 (concat 앞에 삽입 — topo 유지)
        let mut flat_names = Vec::new();
        let mut inserted = 0usize;
        for (k, (inp, s)) in node.inputs.iter().zip(&shapes).enumerate() {
            let n_flat = c * s[3];
            let fname = format!("{}__wflat{k}", node.outputs[0]);
            g.tensors.insert(
                fname.clone(),
                TensorInfo {
                    shape: Some(vec![1, n_flat, 1, 1]),
                    dtype: OnnxDtype::F32,
                    data: None,
                },
            );
            g.nodes.insert(
                idx + inserted,
                Node {
                    op: "relayout".into(),
                    name: format!("{}#wflat{k}", node.name),
                    attrs: Default::default(),
                    inputs: vec![inp.clone()],
                    outputs: vec![fname.clone()],
                    dead: false,
                },
            );
            flat_names.push(fname);
            inserted += 1;
        }
        let cidx = idx + inserted; // 원 Concat의 현재 위치

        // Concat → 플랫 채널 concat + 출력 복원 relayout
        let out = g.nodes[cidx].outputs[0].clone();
        let total: i64 = shapes.iter().map(|s| c * s[3]).sum();
        let cat_name = format!("{out}__wcat");
        g.tensors.insert(
            cat_name.clone(),
            TensorInfo { shape: Some(vec![1, total, 1, 1]), dtype: OnnxDtype::F32, data: None },
        );
        {
            let n = &mut g.nodes[cidx];
            n.inputs = flat_names;
            n.outputs = vec![cat_name.clone()];
            n.attrs.insert("axis".into(), Attr::I(1));
        }
        g.nodes.insert(
            cidx + 1,
            Node {
                op: "relayout".into(),
                name: format!("{}#wunflat", node.name),
                attrs: Default::default(),
                inputs: vec![cat_name],
                outputs: vec![out],
                dead: false,
            },
        );
        report.rewrites += 1;
        idx = cidx + 2;
    }
    Ok(report)
}
