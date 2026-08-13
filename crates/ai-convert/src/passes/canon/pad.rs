//! 채널 끝-제로패딩 Pad를 생산자에 접는다 — BlazeFace 잔차 블록의
//! "MaxPool→Pad(C)→Add" / "Conv(+Relu)→Pad(C)→Add" 패턴 (런타임 비용 0).
//!
//! pads = [0,0,0,0, 0,N,0,0] (C 끝에만 N)일 때:
//!   생산자 maxpool  → pad_c attr += N (커널이 패딩 채널을 0으로 채운다)
//!   생산자 Conv(→Relu) → 가중치 O축을 제로행으로 N만큼 확장 (act(0)=0 전제 —
//!                       Relu/PRelu만 허용). bias도 0 확장.
//! 접은 뒤 생산자의 출력 텐서를 Pad의 출력으로 바꿔치기해 desc가 자연히 커진다
//! (생산자→Pad 사이가 단독 소비여야 한다).

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::{Graph, TensorInfo};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Pad" {
            continue;
        }
        let node = g.nodes[idx].clone();
        // pads: attr(구버전) 또는 두 번째 입력 상수(opset 11+)
        let pads: Vec<i64> = if let Some(p) = node.attr_is("pads") {
            p.to_vec()
        } else if let Some(p) = node
            .inputs
            .get(1)
            .and_then(|n| g.info(n))
            .filter(|t| t.is_const())
            .and_then(|t| t.as_i64s().map(|v| v.to_vec()))
        {
            p
        } else {
            return Err(ConvertError::Unsupported(vec![format!(
                "Pad pads 미상수 ({})",
                node.name
            )]));
        };
        let c_end_only = pads.len() == 8
            && pads[..5].iter().all(|v| *v == 0)
            && pads[6] == 0
            && pads[7] == 0
            && pads[5] > 0;
        if !c_end_only {
            return Err(ConvertError::Unsupported(vec![format!(
                "일반 Pad pads={pads:?} ({})",
                node.name
            )]));
        }
        let extra = pads[5];
        let src = node.inputs[0].clone();
        let out = node.outputs[0].clone();

        // 접을 수 없으면(다중 소비/생산자 미상) k1s1 maxpool(pad_c)로 실체화 —
        // 항등 복사 + 제로필. BlazeFace의 "x가 dw분기와 pad에 갈라지는" 포크가 이 경우.
        let mut materialize = |g: &mut Graph| {
            use crate::ir::Attr;
            let n = &mut g.nodes[idx];
            n.op = "maxpool".into();
            n.inputs.truncate(1);
            n.attrs.clear();
            n.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
            n.attrs.insert("pad_c".into(), Attr::I(extra));
        };

        let sole_producer = g.producer(&src).filter(|_| g.consumers(&src).len() == 1);
        let Some(pi) = sole_producer else {
            materialize(g);
            report.rewrites += 1;
            continue;
        };

        match g.nodes[pi].op.as_str() {
            "maxpool" | "MaxPool" => {
                let prev = g.nodes[pi].attr_i("pad_c").unwrap_or(0);
                g.nodes[pi]
                    .attrs
                    .insert("pad_c".into(), crate::ir::graph::Attr::I(prev + extra));
                g.nodes[pi].outputs[0] = out;
                g.nodes[idx].dead = true;
                report.rewrites += 1;
            }
            "Relu" | "Conv" => {
                // Conv(→Relu)의 cout을 제로 가중치 행으로 확장. Relu(0)=0이라 무해.
                let (conv_i, relu_i) = if g.nodes[pi].op == "Relu" {
                    let rsrc = g.nodes[pi].inputs[0].clone();
                    let ci = g.producer(&rsrc);
                    match ci {
                        Some(ci)
                            if g.nodes[ci].op == "Conv" && g.consumers(&rsrc).len() == 1 =>
                        {
                            (ci, Some(pi))
                        }
                        _ => {
                            materialize(g);
                            report.rewrites += 1;
                            continue;
                        }
                    }
                } else {
                    (pi, None)
                };
                // 가중치 [O,I,kh,kw] → O+extra (제로행), bias [O] → O+extra
                let w_name = g.nodes[conv_i].inputs[1].clone();
                let wi = g.info(&w_name).filter(|t| t.is_const()).unwrap().clone();
                let ws = wi.static_shape().unwrap().to_vec();
                let row = (ws[1] * ws[2] * ws[3]) as usize;
                let mut wd = wi.as_f32s().unwrap().to_vec();
                wd.resize(wd.len() + extra as usize * row, 0.0);
                let new_w = format!("{w_name}__padded{extra}");
                g.add_const(
                    new_w.clone(),
                    TensorInfo {
                        shape: Some(vec![ws[0] + extra, ws[1], ws[2], ws[3]]),
                        dtype: wi.dtype,
                        data: Some(Arc::new(bytemuck::cast_slice(&wd).to_vec())),
                    },
                );
                g.nodes[conv_i].inputs[1] = new_w;
                if let Some(b_name) = g.nodes[conv_i].inputs.get(2).cloned() {
                    let bi = g.info(&b_name).filter(|t| t.is_const()).unwrap().clone();
                    let mut bd = bi.as_f32s().unwrap().to_vec();
                    bd.resize(bd.len() + extra as usize, 0.0);
                    let new_b = format!("{b_name}__padded{extra}");
                    g.add_const(
                        new_b.clone(),
                        TensorInfo {
                            shape: Some(vec![ws[0] + extra]),
                            dtype: bi.dtype,
                            data: Some(Arc::new(bytemuck::cast_slice(&bd).to_vec())),
                        },
                    );
                    g.nodes[conv_i].inputs[2] = new_b;
                }
                // 중간 텐서 shape의 C를 키우고, 체인 끝 출력명을 Pad 출력으로
                let conv_out = g.nodes[conv_i].outputs[0].clone();
                let bump = |g: &mut Graph, name: &str, extra: i64| {
                    if let Some(info) = g.info(name) {
                        if let Some(mut s) = info.shape.clone() {
                            s[1] += extra;
                            g.info_mut(name).shape = Some(s);
                        }
                    }
                };
                match relu_i {
                    Some(ri) => {
                        bump(g, &conv_out, extra);
                        g.nodes[ri].outputs[0] = out;
                    }
                    None => {
                        g.nodes[conv_i].outputs[0] = out;
                    }
                }
                g.nodes[idx].dead = true;
                report.rewrites += 1;
            }
            _ => {
                materialize(g);
                report.rewrites += 1;
            }
        }
    }
    Ok(report)
}
