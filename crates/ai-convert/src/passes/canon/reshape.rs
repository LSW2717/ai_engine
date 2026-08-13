//! Reshape/Squeeze → chcopy 실체화 (flatten 계열만).
//!
//! 허용 조건 — 물리 NHWC 평탄화와 ONNX 논리 평탄화가 일치할 때:
//!   ① transpose 패스가 `flat_ok`로 마킹 (NCHW→NHWC 직후 flatten — tf2onnx 헤드)
//!   ② 실효 1차원 입력 (배치 제외 유닛 아닌 차원 ≤ 1 — 순서가 의미 없음)
//!   ③ Squeeze (원소 재배열이 없어 항상 안전)
//!
//! alias가 아니라 chcopy(복사)인 이유: 소비자(디텍터 헤드의 Concat 등)가 새
//! desc로 텐서를 봐야 하는데 alias는 채널그룹 뷰만 표현한다. 복사량은 KB급
//! (평탄화는 헤드/출력 직전에만 나온다). 일반(순서 바꾸는) Reshape는 미지원.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

/// desc_of 규약(lower.rs)에 따른 채널 수 — chcopy의 n은 "픽셀당 채널"이다
fn chan_of(shape: &[i64]) -> i64 {
    match shape.len() {
        4 => shape[1],
        3 => shape[0], // [C,1,1] 채널벡터
        2 => shape[1],
        _ => shape[0],
    }
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead {
            continue;
        }
        let op = g.nodes[idx].op.clone();
        match op.as_str() {
            "Reshape" => {
                let node = g.nodes[idx].clone();
                let x = g
                    .info(&node.inputs[0])
                    .and_then(|t| t.static_shape())
                    .map(|s| s.to_vec());
                let flat_ok = node.attr_i("flat_ok") == Some(1)
                    || match &x {
                        Some(s) => s[1..].iter().filter(|d| **d > 1).count() <= 1,
                        None => false,
                    };
                if !flat_ok {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "일반 Reshape (재배열 가능성, {})",
                        node.name
                    )]));
                }
                let c = chan_of(&x.unwrap());
                let n = &mut g.nodes[idx];
                n.op = "chcopy".into();
                n.inputs.truncate(1);
                n.attrs.clear();
                n.attrs.insert("src_c".into(), Attr::I(0));
                n.attrs.insert("n".into(), Attr::I(c));
                report.rewrites += 1;
            }
            "Squeeze" => {
                let node = g.nodes[idx].clone();
                let Some(x) = g
                    .info(&node.inputs[0])
                    .and_then(|t| t.static_shape())
                    .map(|s| s.to_vec())
                else {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "Squeeze 입력 shape 미해석 ({})",
                        node.name
                    )]));
                };
                let c = chan_of(&x);
                let n = &mut g.nodes[idx];
                n.op = "chcopy".into();
                n.inputs.truncate(1);
                n.attrs.clear();
                n.attrs.insert("src_c".into(), Attr::I(0));
                n.attrs.insert("n".into(), Attr::I(c));
                report.rewrites += 1;
            }
            _ => {}
        }
    }
    Ok(report)
}
