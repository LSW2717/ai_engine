//! Identity/Dropout(추론) → alias.

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead {
            continue;
        }
        if matches!(g.nodes[idx].op.as_str(), "Identity" | "Dropout") {
            let (src, out) = (g.nodes[idx].inputs[0].clone(), g.nodes[idx].outputs[0].clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
        }
    }
    Ok(report)
}
