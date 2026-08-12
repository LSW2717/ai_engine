//! HardSwish 재합성 — opset 12는 HardSwish op이 없어 `Mul(x, HardSigmoid(x, α=1/6))`로
//! 분해돼 있다(RVM에 20개). 단일 소비자 조건에서 `hswish(x)` 단항으로 되돌린다.
//! (SE 게이트의 Mul(x, HardSigmoid(se_branch))는 입력이 달라 매칭되지 않는다)

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

const ALPHA_SIXTH: f32 = 1.0 / 6.0;

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Mul" {
            continue;
        }
        let node = g.nodes[idx].clone();
        if node.inputs.len() != 2 {
            continue;
        }
        // (x, hsig(x)) 또는 (hsig(x), x)
        for (xi, hi) in [(0usize, 1usize), (1, 0)] {
            let x = &node.inputs[xi];
            let h = &node.inputs[hi];
            let Some(hidx) = g.producer(h) else { continue };
            let hs = &g.nodes[hidx];
            if hs.op != "HardSigmoid" || &hs.inputs[0] != x {
                continue;
            }
            let alpha = hs.attr_f("alpha").unwrap_or(0.2);
            let beta = hs.attr_f("beta").unwrap_or(0.5);
            if (alpha - ALPHA_SIXTH).abs() > 1e-4 || (beta - 0.5).abs() > 1e-4 {
                continue;
            }
            if g.consumers(h).len() != 1 {
                continue;
            }
            let x = x.clone();
            g.nodes[hidx].dead = true;
            let n = &mut g.nodes[idx];
            n.op = "hswish".into();
            n.inputs = vec![x];
            report.rewrites += 1;
            break;
        }
    }
    Ok(report)
}
