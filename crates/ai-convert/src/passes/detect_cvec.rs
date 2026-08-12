//! binary 피연산자 분류 — 스칼라 상수 / [1,C,1,1] 채널벡터(상수·런타임) / 동형 텐서.
//!
//! 결과는 attr 마킹으로 남고 lowering이 SwOperand로 옮긴다:
//! - `scalar`(F) + `scalar_first`(I): RVM의 `Sub(1, z)` 같은 상수 스칼라
//! - `cvec`(I=1): inputs[1]이 [1,C,1,1] — 상수면 블롭 cvec, 런타임이면 CvecTensor
//! 비가환 op(Sub)에서 벡터/스칼라가 첫 피연산자인 경우: 스칼라는 scalar_first로
//! 지원, cvec은 미지원 에러(현 모델들엔 없음).

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

fn is_cvec_shape(s: &[i64]) -> bool {
    // [1,C,1,1] 또는 [C,1,1] (rank-3 브로드캐스트 — RVM의 mean/std 상수)
    (s.len() == 4 && s[0] == 1 && s[2] == 1 && s[3] == 1)
        || (s.len() == 3 && s[1] == 1 && s[2] == 1)
}

fn is_scalar_shape(s: &[i64]) -> bool {
    s.iter().product::<i64>() == 1
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || !matches!(g.nodes[idx].op.as_str(), "Mul" | "Add" | "Sub") {
            continue;
        }
        let node = g.nodes[idx].clone();
        let shape = |n: &str| g.info(n).and_then(|t| t.static_shape().map(|s| s.to_vec()));
        let (Some(sa), Some(sb)) = (shape(&node.inputs[0]), shape(&node.inputs[1])) else {
            continue;
        };

        // 스칼라 상수
        for (i, s) in [(0usize, &sa), (1, &sb)] {
            if is_scalar_shape(s) && g.is_const(&node.inputs[i]) {
                let v = g
                    .info(&node.inputs[i])
                    .and_then(|t| t.as_f32s().map(|f| f[0]))
                    .ok_or_else(|| {
                        ConvertError::Malformed(format!("스칼라 상수 아님 ({})", node.name))
                    })?;
                let n = &mut g.nodes[idx];
                n.attrs.insert("scalar".into(), Attr::F(v));
                n.attrs.insert("scalar_first".into(), Attr::I((i == 0) as i64));
                // 스칼라 피연산자를 입력 목록에서 제거 (텐서 입력만 남김)
                n.inputs.remove(i);
                report.rewrites += 1;
                break;
            }
        }
        if g.nodes[idx].attrs.contains_key("scalar") {
            continue;
        }

        // 채널 벡터
        let a_vec = is_cvec_shape(&sa) && !is_cvec_shape(&sb);
        let b_vec = is_cvec_shape(&sb) && !is_cvec_shape(&sa);
        if a_vec || b_vec {
            if a_vec {
                if node.op == "Sub" {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "Sub(cvec, tensor) ({})",
                        node.name
                    )]));
                }
                g.nodes[idx].inputs.swap(0, 1); // 가환 → 벡터를 둘째로
            }
            g.nodes[idx].attrs.insert("cvec".into(), Attr::I(1));
            report.rewrites += 1;
        }
    }
    Ok(report)
}
