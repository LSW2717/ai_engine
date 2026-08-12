//! GRU 갱신 융합 — `(1-z)*a + z*b` 4-op 체인을 mix 1-op로.
//!
//! 패턴 (RVM ConvGRU 상태 갱신, 상태 4개 × 4디스패치 = 16 → 4):
//!   Sub:  s  = 1 - z      (scalar_first, v=1.0 — canon이 attrs로 옮겨둠)
//!   MulA: mA = s * a
//!   MulB: mB = z * b
//!   Add:  out = mA + mB
//! → mix(a, b, z) = a + z*(b-a). s/mA/mB가 각각 단일 소비자일 때만 융합
//! (다른 소비자가 있으면 중간값이 필요하므로 놔둔다).
//!
//! 수치 주의: (1-z)a+zb와 a+z(b-a)는 부동소수점상 비트 동일이 아니다 —
//! 오라클 대조는 허용오차 기반이라 통과한다.

use crate::error::ConvertError;
use crate::ir::{Graph, Node};
use crate::passes::{Ctx, PassReport};

/// node가 `1 - x` 형태의 Sub인가 (canon 후: inputs=[x], attrs{scalar:1.0, scalar_first:1})
fn is_one_minus(n: &Node) -> bool {
    n.op == "Sub"
        && n.attr_f("scalar") == Some(1.0)
        && n.attr_i("scalar_first") == Some(1)
        && n.attr_s("act").is_none()
        && n.inputs.len() == 1
}

/// 순수 tensor∘tensor Mul인가 (scalar/cvec/act 미부착)
fn is_plain_mul(n: &Node) -> bool {
    n.op == "Mul"
        && n.inputs.len() == 2
        && n.attr_s("act").is_none()
        && !n.attrs.contains_key("scalar")
        && !n.attrs.contains_key("cvec")
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        let n = &g.nodes[idx];
        if n.dead || n.op != "Add" || n.inputs.len() != 2 || n.attr_s("act").is_some() {
            continue;
        }
        let add = g.nodes[idx].clone();

        // 두 피연산자 모두 단일 소비자(이 Add)의 순수 Mul이어야 한다
        let Some(ma_idx) = g.producer(&add.inputs[0]) else { continue };
        let Some(mb_idx) = g.producer(&add.inputs[1]) else { continue };
        if !is_plain_mul(&g.nodes[ma_idx])
            || !is_plain_mul(&g.nodes[mb_idx])
            || g.consumers(&add.inputs[0]).len() != 1
            || g.consumers(&add.inputs[1]).len() != 1
        {
            continue;
        }

        // 한쪽 Mul의 피연산자에 (1-z) Sub가 있는지 — 대칭이므로 양쪽 순서 모두 시도
        let mut found: Option<(usize, String, String, String)> = None; // (sub_idx, a, b, z)
        'outer: for (sub_side_idx, other_idx) in [(ma_idx, mb_idx), (mb_idx, ma_idx)] {
            let sub_side = g.nodes[sub_side_idx].clone();
            for si in 0..2 {
                let s_name = &sub_side.inputs[si];
                let Some(sub_idx) = g.producer(s_name) else { continue };
                if !is_one_minus(&g.nodes[sub_idx]) || g.consumers(s_name).len() != 1 {
                    continue;
                }
                let z = g.nodes[sub_idx].inputs[0].clone();
                let a = sub_side.inputs[1 - si].clone();
                // 반대쪽 Mul이 z를 읽어야 완전한 GRU 패턴
                let other = &g.nodes[other_idx];
                let Some(zi) = other.inputs.iter().position(|i| *i == z) else { continue };
                let b = other.inputs[1 - zi].clone();
                found = Some((sub_idx, a, b, z));
                break 'outer;
            }
        }
        let Some((sub_idx, a, b, z)) = found else { continue };

        // Add 자리를 mix로 교체 (a·b·z 모두 Add보다 먼저 계산돼 있어 topo 안전)
        g.nodes[idx] = Node {
            op: "mix".into(),
            name: format!("{}_mix", add.name),
            attrs: Default::default(),
            inputs: vec![a, b, z],
            outputs: add.outputs.clone(),
            dead: false,
        };
        g.nodes[ma_idx].dead = true;
        g.nodes[mb_idx].dead = true;
        g.nodes[sub_idx].dead = true;
        report.rewrites += 1;
    }
    Ok(report)
}
