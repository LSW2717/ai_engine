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

/// desc_of 규약(lower.rs)에 따른 채널 수 — chcopy의 n은 "픽셀당 채널"이다.
/// ⚠ 폴딩 규칙과 반드시 일치해야 한다: [1,1,1,N]은 채널벡터 (N) — s[1]로 읽으면
/// n=1이 되어 채널 1개만 복사된다 (blendshapes 항등 Reshape에서 실제로 당함)
/// lower desc_of 미러 — (h,w,c) 트리플 (chcopy 가능성 판정용)
fn desc3(shape: &[i64]) -> (i64, i64, i64) {
    match shape.len() {
        4 if shape[1] == 1 && shape[2] == 1 => (1, 1, shape[3]),
        4 => (shape[2], shape[3], shape[1]),
        3 if shape[1] == 1 && shape[2] == 1 => (1, 1, shape[0]),
        3 if shape[0] == 1 => (1, 1, shape[1] * shape[2]),
        3 => (1, 1, shape[0]),
        2 => (1, shape[0], shape[1]),
        _ => (1, 1, shape.first().copied().unwrap_or(1)),
    }
}

fn chan_of(shape: &[i64]) -> i64 {
    match shape.len() {
        4 if shape[1] == 1 && shape[2] == 1 => shape[3], // 채널벡터 폴딩 (desc_of 동일)
        4 => shape[1],
        3 if shape[1] == 1 && shape[2] == 1 => shape[0], // [C,1,1] 채널벡터
        3 if shape[0] == 1 => shape[1] * shape[2],       // [1,a,b] 평탄
        3 => shape[0],
        2 => shape[1],
        _ => shape[0],
    }
}

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead {
            idx += 1;
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
                    // Reshape는 ONNX **NCHW row-major 평탄 순서**를 보존한다. 엔진
                    // relayout은 **desc(NHWC) 스트림** 항등이라, 채널-major 스트림
                    // (rank4에 C>1 && H·W>1)이 낀 쪽은 transpose로 보정해야 한다
                    // (blendshapes [1,1,146,2]→[1,146,1,2]에서 실제로 당함).
                    let out_shape = g
                        .info(&node.outputs[0])
                        .and_then(|t| t.static_shape().map(|s| s.to_vec()));
                    let cmajor = |s: &Option<Vec<i64>>| {
                        s.as_ref()
                            .map(|s| s.len() == 4 && s[1] > 1 && s[2] * s[3] > 1)
                            .unwrap_or(false)
                    };
                    let (in_cm, out_cm) = (cmajor(&x), cmajor(&out_shape));
                    if !in_cm && !out_cm {
                        // 양쪽 다 desc==row-major → 순수 relayout
                        g.nodes[idx].op = "relayout".into();
                        g.nodes[idx].inputs.truncate(1);
                        report.rewrites += 1;
                        idx += 1;
                        continue;
                    }
                    if !in_cm && out_cm {
                        let so = out_shape.as_ref().unwrap();
                        if so[2] != 1 {
                            return Err(ConvertError::Unsupported(vec![format!(
                                "채널-major Reshape H>1 ({})",
                                node.name
                            )]));
                        }
                        // relayout(in → mid [1, W_o, 1, C_o]) + transpose(mid → out)
                        let mid = format!("{}__rsh_m", node.outputs[0]);
                        g.tensors.insert(
                            mid.clone(),
                            crate::ir::tensor_info::TensorInfo {
                                shape: Some(vec![1, so[3], 1, so[1]]),
                                dtype: crate::ir::tensor_info::OnnxDtype::F32,
                                data: None,
                            },
                        );
                        let out_name = node.outputs[0].clone();
                        {
                            let n = &mut g.nodes[idx];
                            n.op = "relayout".into();
                            n.inputs.truncate(1);
                            n.outputs = vec![mid.clone()];
                        }
                        g.nodes.insert(
                            idx + 1,
                            crate::ir::Node {
                                op: "transpose".into(),
                                name: format!("{}#rsh_t", node.name),
                                attrs: Default::default(),
                                inputs: vec![mid],
                                outputs: vec![out_name],
                                dead: false,
                            },
                        );
                        report.rewrites += 1;
                        idx += 2;
                        continue;
                    }
                    if in_cm && !out_cm {
                        let si = x.as_ref().unwrap();
                        if si[2] != 1 {
                            return Err(ConvertError::Unsupported(vec![format!(
                                "채널-major Reshape H>1 ({})",
                                node.name
                            )]));
                        }
                        // transpose(in (1,W,C) → mid (1,C,W)) + relayout(mid → out)
                        let mid = format!("{}__rsh_m", node.outputs[0]);
                        g.tensors.insert(
                            mid.clone(),
                            crate::ir::tensor_info::TensorInfo {
                                shape: Some(vec![1, si[3], 1, si[1]]),
                                dtype: crate::ir::tensor_info::OnnxDtype::F32,
                                data: None,
                            },
                        );
                        let out_name = node.outputs[0].clone();
                        let in_name = node.inputs[0].clone();
                        {
                            let n = &mut g.nodes[idx];
                            n.op = "transpose".into();
                            n.inputs = vec![in_name];
                            n.attrs.clear();
                            n.outputs = vec![mid.clone()];
                        }
                        g.nodes.insert(
                            idx + 1,
                            crate::ir::Node {
                                op: "relayout".into(),
                                name: format!("{}#rsh_r", node.name),
                                attrs: Default::default(),
                                inputs: vec![mid],
                                outputs: vec![out_name],
                                dead: false,
                            },
                        );
                        report.rewrites += 1;
                        idx += 2;
                        continue;
                    }
                    return Err(ConvertError::Unsupported(vec![format!(
                        "양쪽 채널-major Reshape ({})",
                        node.name
                    )]));
                }
                // desc가 다르면 chcopy(픽셀당 채널 복사)가 지오메트리를 깨뜨린다 —
                // (1,1,97)→(97,1,1) 같은 폴딩 차이는 relayout으로 (실제로 당함:
                // chcopy가 GPU flatten 경로로 새어 16곳 오배선)
                let xs = x.unwrap();
                let os = g
                    .info(&node.outputs[0])
                    .and_then(|t| t.static_shape().map(|s| s.to_vec()))
                    .unwrap_or_else(|| xs.clone());
                let n = &mut g.nodes[idx];
                if desc3(&xs) != desc3(&os) {
                    n.op = "relayout".into();
                    n.inputs.truncate(1);
                    n.attrs.clear();
                } else {
                    let c = chan_of(&xs);
                    n.op = "chcopy".into();
                    n.inputs.truncate(1);
                    n.attrs.clear();
                    n.attrs.insert("src_c".into(), Attr::I(0));
                    n.attrs.insert("n".into(), Attr::I(c));
                }
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
                let os = g
                    .info(&node.outputs[0])
                    .and_then(|t| t.static_shape().map(|s| s.to_vec()))
                    .unwrap_or_else(|| x.clone());
                let n = &mut g.nodes[idx];
                if desc3(&x) != desc3(&os) {
                    n.op = "relayout".into();
                    n.inputs.truncate(1);
                    n.attrs.clear();
                } else {
                    let c = chan_of(&x);
                    n.op = "chcopy".into();
                    n.inputs.truncate(1);
                    n.attrs.clear();
                    n.attrs.insert("src_c".into(), Attr::I(0));
                    n.attrs.insert("n".into(), Attr::I(c));
                }
                report.rewrites += 1;
            }
            _ => {}
        }
        idx += 1;
    }
    Ok(report)
}
