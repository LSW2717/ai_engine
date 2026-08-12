//! Split→Concat 상쇄: `Concat(Split(x)의 전 출력, 원순서, 같은 축)` → x의 alias.
//! (RVM refiner의 Split_295/Concat_300 쌍이 여기서 사라진다)

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Concat" {
            continue;
        }
        let concat = g.nodes[idx].clone();
        let Some(split_idx) = g.producer(&concat.inputs[0]) else { continue };
        if g.nodes[split_idx].op != "Split" {
            continue;
        }
        let split = g.nodes[split_idx].clone();
        let same_axis = {
            let ca = concat.attr_i("axis").unwrap_or(0);
            let sa = split.attr_i("axis").unwrap_or(0);
            ca == sa
        };
        if same_axis && split.outputs == concat.inputs {
            let (src, out) = (split.inputs[0].clone(), concat.outputs[0].clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
        }
    }
    Ok(report)
}
