//! concat-into-conv 융합 — webgl2 타일판의 srcs 멀티소스 아이디어를 컴퓨트로.
//!
//! 채널 Concat의 모든 소비자가 일반 conv(k>1, groups=1)이고 파트 경계가 전부
//! 채널그룹(4ch) 정렬이면, conv가 파트 텐서들을 직접 읽게 하고 Concat을 지운다 —
//! concat 텐서의 쓰기+읽기 왕복(144×256c35 기준 ~5.2MB)과 디스패치가 사라진다.
//! RVM 디코더의 UNet 스킵 concat 12개가 대상 (fgr+pha 3+1은 비정렬이라 제외,
//! resize/pw 소비자는 후속 확장).
//!
//! 표현: 소비자 conv의 inputs[0] = 첫 파트, 나머지 파트는 inputs **끝에** 덧붙이고
//! (w/bias 인덱스 보존 + dce가 파트 생존을 보게), attrs["src_cs"] = 파트 채널 수
//! 목록. lowering이 SwOp::Conv.srcs로 옮긴다.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

/// 최대 파트 수 — 커널 바인딩 슬롯과 일치 (RVM 최대 3)
pub const MAX_PARTS: usize = 3;

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        let n = &g.nodes[idx];
        if n.dead || n.op != "Concat" {
            continue;
        }
        let axis = n.attr_i("axis").unwrap_or(1);
        if axis != 1 && axis != -3 {
            continue; // 채널 축 아님 — lowering이 거부할 영역
        }
        let concat = g.nodes[idx].clone();
        if concat.inputs.len() < 2 || concat.inputs.len() > MAX_PARTS {
            continue;
        }

        // 파트 채널 수 + 그룹 정렬 검사 (마지막 파트 빼고 누적합이 4의 배수)
        let mut cs: Vec<i64> = Vec::new();
        let mut ok = true;
        for i in &concat.inputs {
            // NCHW 논리 shape [1, C, H, W]
            match g.info(i).and_then(|t| t.static_shape().map(|s| s.to_vec())) {
                Some(s) if s.len() == 4 => cs.push(s[1]),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let mut sum = 0i64;
        for c in &cs[..cs.len() - 1] {
            sum += c;
            if sum % 4 != 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // 모든 소비자가 융합 가능한 conv여야 concat을 지울 수 있다
        let out = concat.outputs[0].clone();
        let users = g.consumers(&out);
        if users.is_empty() || g.is_output(&out) || g.states.iter().any(|(_, o)| *o == out) {
            continue;
        }
        let fusable = users.iter().all(|&u| {
            let un = &g.nodes[u];
            if un.attrs.contains_key("src_cs") || un.inputs[0] != out {
                return false;
            }
            // resize도 멀티소스 판독 가능 (cg 범위로 파트 선택)
            if un.op == "resize" {
                return true;
            }
            // k>1은 가중치 shape [cout, cin, kh, kw]로 판별 (kernel_shape attr는 선택적)
            let k_gt1 = un
                .inputs
                .get(1)
                .and_then(|w| g.info(w))
                .and_then(|t| t.static_shape().map(|s| s.len() == 4 && s[2] > 1))
                .unwrap_or(false);
            un.op == "Conv"
                && un.attr_i("group").unwrap_or(1) == 1
                && k_gt1
                // res 피연산자로도 concat을 읽으면 실체화가 여전히 필요
                && un.attr_s("res") != Some(&out)
        });
        if !fusable {
            continue;
        }

        for u in users {
            let un = &mut g.nodes[u];
            un.inputs[0] = concat.inputs[0].clone();
            un.inputs.extend(concat.inputs[1..].iter().cloned());
            un.attrs.insert("src_cs".into(), Attr::Is(cs.clone()));
        }
        g.nodes[idx].dead = true;
        report.rewrites += 1;
    }
    Ok(report)
}
