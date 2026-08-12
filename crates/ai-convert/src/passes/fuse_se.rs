//! SE 게이트 융합 — gpool → 1×1 conv(act) [→ 1×1 conv(act)] 체인을 segate 1-op로.
//!
//! RVM/MNv 계열의 SE 블록은 M=1 벡터 경로라 op당 일감이 극소한데 디스패치+배리어
//! 비용은 그대로 낸다 (체인 8~9개 × 2~3 디스패치). 워크그룹 하나가 평균→FC1→FC2를
//! 공유메모리로 이어 한 디스패치로 끝낸다. 후속 cvec-mul은 건드리지 않는다.
//!
//! 조건: gpool 출력이 단일 소비 1×1 Conv(groups=1, res 없음), (선택) 그 출력도
//! 단일 소비 1×1 Conv. 가중치·bias는 상수여야 한다 (lower가 패킹).

use crate::error::ConvertError;
use crate::ir::{Attr, Graph, Node};
use crate::passes::{Ctx, PassReport};

/// 단일 소비 1×1 conv(groups=1, res 없음)이면 (weight, bias, cout, act) 반환
fn as_fusable_pw(g: &Graph, idx: usize) -> Option<(String, Option<String>, i64, String)> {
    let n = &g.nodes[idx];
    if n.op != "Conv" || n.attr_i("group").unwrap_or(1) != 1 || n.attrs.contains_key("res") {
        return None;
    }
    if n.attrs.contains_key("src_cs") {
        return None;
    }
    let w = n.inputs.get(1)?;
    let ws = g.info(w).and_then(|t| t.static_shape().map(|s| s.to_vec()))?;
    if ws.len() != 4 || ws[2] != 1 || ws[3] != 1 || !g.is_const(w) {
        return None;
    }
    let b = n.inputs.get(2).cloned();
    if let Some(bn) = &b {
        if !g.is_const(bn) {
            return None;
        }
    }
    let act = n.attr_s("act").unwrap_or("none").to_string();
    Some((w.clone(), b, ws[0], act))
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "gpool" {
            continue;
        }
        let gp = g.nodes[idx].clone();
        let gp_out = gp.outputs[0].clone();
        let users = g.consumers(&gp_out);
        if users.len() != 1 || g.is_output(&gp_out) {
            continue;
        }
        let c1 = users[0];
        let Some((w1, b1, c_mid, act1)) = as_fusable_pw(g, c1) else { continue };
        let Some(b1) = b1 else { continue }; // bias 필수 (RVM은 항상 있음)
        let c1_out = g.nodes[c1].outputs[0].clone();
        if g.is_output(&c1_out) {
            continue;
        }

        // 두 번째 FC (선택)
        let u2 = g.consumers(&c1_out);
        let fc2 = if u2.len() == 1 {
            as_fusable_pw(g, u2[0]).and_then(|(w2, b2, c_out, act2)| {
                let out2 = g.nodes[u2[0]].outputs[0].clone();
                if g.is_output(&out2) {
                    return None;
                }
                b2.map(|b2| (u2[0], w2, b2, c_out, act2, out2))
            })
        } else {
            None
        };

        let mut node = Node {
            op: "segate".into(),
            name: format!("{}_segate", gp.name),
            attrs: Default::default(),
            inputs: vec![gp.inputs[0].clone(), w1, b1],
            outputs: vec![],
            dead: false,
        };
        node.attrs.insert("c_mid".into(), Attr::I(c_mid));
        node.attrs.insert("act1".into(), Attr::S(act1));
        if let Some((c2_idx, w2, b2, c_out, act2, out2)) = fc2 {
            node.inputs.push(w2);
            node.inputs.push(b2);
            node.attrs.insert("c_out".into(), Attr::I(c_out));
            node.attrs.insert("act2".into(), Attr::S(act2));
            node.outputs = vec![out2];
            g.nodes[c2_idx].dead = true;
        } else {
            node.outputs = vec![c1_out.clone()];
        }
        g.nodes[c1].dead = true;
        g.nodes[idx] = node;
        report.rewrites += 1;
    }
    Ok(report)
}
