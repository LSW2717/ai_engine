//! Clip(x, 0, 1) → act clamp01. 다른 경계(Relu6 등)는 명시적 미지원.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Clip" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let scalar = |i: usize, attr: &str| -> Option<f32> {
            node.inputs
                .get(i)
                .and_then(|n| g.info(n))
                .and_then(|t| t.as_scalar_f32())
                .or_else(|| node.attr_f(attr))
        };
        let (min, max) = (scalar(1, "min"), scalar(2, "max"));
        match (min, max) {
            (Some(lo), Some(hi)) if lo == 0.0 && (hi == 1.0 || hi == 6.0) => {
                let tag = if hi == 1.0 { "clamp01" } else { "relu6" };
                let n = &mut g.nodes[idx];
                n.op = "act".into();
                n.inputs.truncate(1);
                n.attrs.clear();
                n.attrs.insert("act".into(), Attr::S(tag.into()));
                report.rewrites += 1;
            }
            other => {
                return Err(ConvertError::Unsupported(vec![format!(
                    "Clip 경계 {other:?} ({}) — clamp01/relu6만 지원",
                    node.name
                )]))
            }
        }
    }
    Ok(report)
}
