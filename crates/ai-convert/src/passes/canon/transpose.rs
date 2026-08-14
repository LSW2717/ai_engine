//! 경계 Transpose 처리 — NHWC export(segm)의 입출력 순서 메타로 흡수.
//! 내부 물리 레이아웃은 어차피 NHWC-C4라, 호출자 데이터의 논리 순서만 기록하면 된다.
//! 내부(비경계) Transpose는 미지원.

use crate::error::ConvertError;
use crate::ir::graph::Attr;
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

        let in_shape = g.info(&src).and_then(|t| t.static_shape().map(|s| s.to_vec()));
        let dim = |i: usize| in_shape.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(0);
        let rank4 = in_shape.as_ref().map(|s| s.len() == 4).unwrap_or(false);
        let boundary_in = g.inputs.iter().any(|i| *i == src);
        if perm == [0, 3, 2, 1] && rank4 && dim(2) == 1 {
            // 내부 W↔C 전치 (h==1) — MLP-Mixer 토큰↔채널 (face_blendshapes).
            // 레인 재배치가 필요한 실복사 → transpose 커널 op로 강등
            g.nodes[idx].op = "transpose".into();
            report.rewrites += 1;
        } else if !boundary_in
            && ((perm == [0, 2, 3, 1] && rank4 && dim(2) == 1)
                || (perm == [0, 3, 1, 2] && rank4 && dim(1) == 1))
        {
            // 논리 row-major 순서 보존 재배치 (tf2onnx 언팩/팩) — 데이터 전치 아님.
            // [0,2,3,1] in [1,C,1,W]→[1,1,W,C] / [0,3,1,2] in [1,1,H,W]→[1,W,1,H]
            // 둘 다 desc의 논리 스트림이 동일 → relayout(레인 재패킹)으로 강등
            g.nodes[idx].op = "relayout".into();
            report.rewrites += 1;
        } else if perm == [0, 3, 1, 2] && boundary_in {
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
        } else if perm == [0, 2, 3, 1] && {
            let cons = g.consumers(&out);
            !cons.is_empty() && cons.iter().all(|&ci| g.nodes[ci].op == "Reshape")
        } {
            // tf2onnx 디텍터 헤드: NCHW→NHWC 뒤 flatten-Reshape. 물리 레이아웃이
            // 이미 NHWC라 transpose는 항등 — 소비 Reshape에 flat_ok를 마킹해
            // reshape 패스가 안전하게 chcopy로 실체화하게 한다.
            for ci in g.consumers(&out) {
                g.nodes[ci].attrs.insert("flat_ok".into(), Attr::I(1));
            }
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
