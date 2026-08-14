//! 활성화 융합 — 단항 활성화의 생산자가 (act 없는) Conv/elementwise이고
//! 그 출력의 소비자가 이 활성화 하나뿐이면 에필로그로 흡수한다.
//! 융합 안 된 활성화는 standalone으로 남아 elementwise Unary로 lowering된다.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

const SIXTH: f32 = 1.0 / 6.0;

/// 단항 활성화 노드 → 엔진 act 태그
fn act_tag(g: &Graph, idx: usize) -> Result<Option<&'static str>, ConvertError> {
    let n = &g.nodes[idx];
    Ok(match n.op.as_str() {
        "Relu" => Some("relu"),
        "Sigmoid" => Some("sigmoid"),
        "Tanh" => Some("tanh"),
        "hswish" => Some("hswish"),
        "Sqrt" => Some("sqrt"),
        "Neg" => Some("neg"),
        "Reciprocal" => Some("recip"),
        "HardSigmoid" => {
            let a = n.attr_f("alpha").unwrap_or(0.2);
            let b = n.attr_f("beta").unwrap_or(0.5);
            if (a - SIXTH).abs() > 1e-4 || (b - 0.5).abs() > 1e-4 {
                return Err(ConvertError::Unsupported(vec![format!(
                    "HardSigmoid alpha={a} beta={b} ({}) — 1/6, 0.5만 지원",
                    n.name
                )]));
            }
            Some("hsigmoid")
        }
        // ⚠ 알 수 없는 태그를 "none"으로 접으면 활성화가 조용히 증발한다
        // (relu6가 실제로 그렇게 사라져 hand_landmarks 출력이 1e9까지 폭주했다).
        // 모르는 태그는 융합하지 않고 standalone으로 남긴다.
        "act" => n.attr_s("act").and_then(|s| match s {
            "clamp01" => Some("clamp01"),
            "relu6" => Some("relu6"),
            "relu" => Some("relu"),
            "sigmoid" => Some("sigmoid"),
            "tanh" => Some("tanh"),
            "hswish" => Some("hswish"),
            "hsigmoid" => Some("hsigmoid"),
            _ => None,
        }),
        _ => None,
    })
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead {
            continue;
        }
        let Some(tag) = act_tag(g, idx)? else { continue };
        let node = g.nodes[idx].clone();
        let src = &node.inputs[0];
        let Some(pidx) = g.producer(src) else { continue };
        let p = &g.nodes[pidx];
        let fusable_producer =
            matches!(p.op.as_str(), "Conv" | "Mul" | "Add" | "Sub") && p.attr_s("act").is_none();
        if !fusable_producer || g.consumers(src).len() != 1 {
            continue;
        }
        g.nodes[pidx].attrs.insert("act".into(), Attr::S(tag.into()));
        let (src, out) = (src.clone(), node.outputs[0].clone());
        // src 이름은 이제 활성화 "후" 값을 담는다 — 오라클의 같은 이름(활성화 전)과 다름
        g.semantic_changed.push(src.clone());
        g.make_alias(idx, &src, &out);
        report.rewrites += 1;
    }
    Ok(report)
}
