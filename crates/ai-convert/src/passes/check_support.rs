//! 최종 지원성 검사 — 남은 모든 노드가 lowering 가능한지, 미지원은 전체 목록으로 보고.

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

const LOWERABLE: &[&str] = &[
    "Conv", "Mul", "Add", "Sub", "Relu", "Sigmoid", "Tanh", "HardSigmoid", "hswish", "act",
    "Concat", "chview", "chcopy", "resize", "avgpool", "gpool", "maxpool", "mix", "segate",
    "PRelu", "Sqrt", "Neg", "Reciprocal", "transpose", "relayout",
];

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut problems = Vec::new();
    for (_, n) in g.live_nodes() {
        if !LOWERABLE.contains(&n.op.as_str()) {
            problems.push(format!("{} ({})", n.op, n.name));
            continue;
        }
        if n.op == "Conv" {
            let d = n.attr_is("dilations").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
            if d.len() != 2 || d[0] != d[1] || d[0] < 1 {
                problems.push(format!("비등방 Conv dilation {d:?} ({})", n.name));
            }
            let group = n.attr_i("group").unwrap_or(1);
            if group != 1 {
                // depthwise(g==cin==cout)만 허용 — 가중치 shape [cout, cin/g, kh, kw]
                let w = g
                    .info(&n.inputs[1])
                    .and_then(|t| t.static_shape().map(|s| s.to_vec()))
                    .unwrap_or_default();
                let (cout, cin_g) = (w[0], w[1]);
                if !(cin_g == 1 && cout == group) {
                    problems.push(format!(
                        "grouped Conv g={group} w={w:?} ({}) — depthwise만 지원",
                        n.name
                    ));
                }
            }
            let ks = n.attr_is("kernel_shape").map(|v| v.to_vec()).unwrap_or_default();
            if ks.len() == 2 && ks[0] != ks[1] {
                problems.push(format!("비정방 커널 {ks:?} ({})", n.name));
            }
        }
    }
    if problems.is_empty() {
        Ok(PassReport::default())
    } else {
        Err(ConvertError::Unsupported(problems))
    }
}
