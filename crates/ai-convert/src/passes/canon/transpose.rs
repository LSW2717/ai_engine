//! 경계 Transpose 처리 — NHWC export(segm)의 입출력 순서 메타로 흡수.
//! 내부 물리 레이아웃은 어차피 NHWC-C4라, 호출자 데이터의 논리 순서만 기록하면 된다.
//! 내부(비경계) Transpose는 미지원.

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Transpose" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let perm = node.attr_is("perm").map(|v| v.to_vec()).unwrap_or_default();
        let src = node.inputs[0].clone();
        let out = node.outputs[0].clone();

        if perm == [0, 3, 1, 2] && g.inputs.iter().any(|i| *i == src) {
            // NHWC 입력 → 내부 NCHW: 입력 텐서의 IR shape을 NCHW로 재기록
            let nchw = g.info(&out).unwrap().static_shape().unwrap().to_vec();
            g.info_mut(&src).shape = Some(nchw);
            g.nhwc_inputs.push(src.clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
        } else if perm == [0, 2, 3, 1] && g.is_output(&out) {
            g.nhwc_outputs.push(out.clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
        } else {
            return Err(ConvertError::Unsupported(vec![format!(
                "내부 Transpose perm={perm:?} ({})",
                node.name
            )]));
        }
    }
    Ok(report)
}
