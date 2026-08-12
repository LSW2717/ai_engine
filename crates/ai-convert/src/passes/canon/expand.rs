//! 잔여 Expand — 동일 shape이면 alias, 아니면 미지원.
//! (상태 입력의 Expand는 resolve_static에서 이미 해소됨)

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Expand" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let xs = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
        let os = g.info(&node.outputs[0]).unwrap().static_shape().unwrap().to_vec();
        if xs == os {
            let (src, out) = (node.inputs[0].clone(), node.outputs[0].clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
        } else {
            return Err(ConvertError::Unsupported(vec![format!(
                "브로드캐스트 Expand {xs:?}→{os:?} ({})",
                node.name
            )]));
        }
    }
    Ok(report)
}
