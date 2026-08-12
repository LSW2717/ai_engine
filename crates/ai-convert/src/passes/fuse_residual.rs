//! residual 융합 — WebGL2 conv-tail 규칙.
//!
//! `Add(a, b)`에서 한 피연산자의 생산자가 Conv(단일 소비자, res 미보유)이고
//! **다른 피연산자가 topo상 그 Conv보다 먼저** 계산돼 있으면(생존 보장 + 무순환)
//! Conv의 res 에필로그로 흡수한다. 둘 다 후보면 더 깊은(뒤쪽) Conv를 택한다.
//! 에필로그 순서는 bias→act→residual — CPU 레퍼런스·GPU 커널과 동일 규약.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Add" {
            continue;
        }
        let node = g.nodes[idx].clone();
        if node.inputs.len() != 2 || node.attr_s("act").is_some() {
            continue; // act가 이미 융합된 Add는 대상 아님 (에필로그에 post-res act 슬롯 없음)
        }

        // 각 피연산자의 conv 후보 평가
        let mut best: Option<(usize, usize, usize)> = None; // (conv_idx, conv_operand, other_operand)
        for (ci, oi) in [(0usize, 1usize), (1, 0)] {
            let cn = &node.inputs[ci];
            let other = &node.inputs[oi];
            if g.is_const(other) {
                continue;
            }
            let Some(pidx) = g.producer(cn) else { continue };
            if g.nodes[pidx].op != "Conv"
                || g.nodes[pidx].attrs.contains_key("res")
                || g.consumers(cn).len() != 1
            {
                continue;
            }
            // 안전 규칙: other가 conv보다 topo상 먼저 (그래프 입력이면 무조건 먼저)
            let other_idx = g.producer(other);
            let ok = match other_idx {
                Some(oidx) => oidx < pidx,
                None => g.inputs.iter().any(|i| i == other),
            };
            if !ok {
                continue;
            }
            // 더 깊은 conv 선택
            if best.map(|(b, _, _)| pidx > b).unwrap_or(true) {
                best = Some((pidx, ci, oi));
            }
        }

        if let Some((pidx, _, oi)) = best {
            let other = node.inputs[oi].clone();
            g.nodes[pidx].attrs.insert("res".into(), Attr::S(other));
            let conv_out = g.nodes[pidx].outputs[0].clone();
            let add_out = node.outputs[0].clone();
            // conv_out과 그것을 가리키던 기존 별칭들(act 융합 산물)은 이제 res 포함 값
            g.semantic_changed.push(conv_out.clone());
            let stale: Vec<String> = g
                .alias_of
                .iter()
                .filter(|(a, _)| **a != add_out && g.resolve_alias(a) == conv_out)
                .map(|(a, _)| a.clone())
                .collect();
            g.semantic_changed.extend(stale);
            g.make_alias(idx, &conv_out, &add_out);
            report.rewrites += 1;
        }
    }
    Ok(report)
}
