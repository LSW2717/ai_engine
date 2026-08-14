//! 외적 브로드캐스트 Mul 정규화 (tf batchnorm 분해 산물, face_blendshapes) —
//! `Mul(pvec [1,1,1,W] 런타임, cvec 상수 [C])` → out [1,C,1,W] 외적.
//!
//! 표현: transpose(pvec → [1,1,W,1] = W픽셀×1ch) → 1×1 Conv(cin=1, cout=C,
//! 가중치 = cvec) → relayout([1,C,W,1] → [1,C,1,W], 논리 항등·물리 동일).
//! Conv가 곧 외적: out[p,ch] = a[p] · w[ch].

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{Attr, Graph, Node};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Mul" {
            idx += 1;
            continue;
        }
        let node = g.nodes[idx].clone();
        let shape =
            |n: &str| g.info(n).and_then(|t| t.static_shape().map(|s| s.to_vec()));
        let (Some(s0), Some(s1), Some(so)) =
            (shape(&node.inputs[0]), shape(&node.inputs[1]), shape(&node.outputs[0]))
        else {
            idx += 1;
            continue;
        };
        // 벡터 형태 판정: [1,1,1,W](채널벡터 — transpose 필요) / [1,1,W,1](이미 픽셀벡터)
        let vec_w = |s: &[i64]| -> Option<(i64, bool)> {
            if s.len() != 4 || s[0] != 1 || s[1] != 1 {
                return None;
            }
            match (s[2], s[3]) {
                (1, w) if w > 1 => Some((w, true)),  // [1,1,1,W] — transpose 필요
                (w, 1) if w > 1 => Some((w, false)), // [1,1,W,1] — 그대로
                _ => None,
            }
        };
        // (런타임 벡터, 상수 벡터) 순서 정규화
        let (pv_i, cv_i, w_px, need_t) = match (vec_w(&s0), vec_w(&s1)) {
            (Some((w, t)), _) if g.is_const(&node.inputs[1]) => (0usize, 1usize, w, t),
            (_, Some((w, t))) if g.is_const(&node.inputs[0]) => (1, 0, w, t),
            _ => {
                idx += 1;
                continue;
            }
        };
        let out_n: i64 = so.iter().product();
        let cvals = match g.info(&node.inputs[cv_i]).and_then(|t| t.as_f32s().map(|v| v.to_vec()))
        {
            Some(v) => v,
            None => {
                idx += 1;
                continue;
            }
        };
        let c = cvals.len() as i64;
        // 외적 out은 desc 스트림이 픽셀(y) 우선이어야 한다: [1,C,1,W] / [1,1,W,C] 둘 다
        // n = y·C + ch (검증 완료). 그 외 배열은 미지원 — 건너뛰고 lower가 거절하게 둔다.
        let stream_ok = so.len() == 4
            && out_n == w_px * c
            && ((so[1] == c && so[2] == 1 && so[3] == w_px)
                || (so[1] == 1 && so[2] == w_px && so[3] == c));
        if c < 2 || !stream_ok {
            idx += 1;
            continue;
        }

        let pv = node.inputs[pv_i].clone();
        let out = node.outputs[0].clone();
        // ① (필요 시) transpose: [1,1,1,W](채널벡터) → [1,1,W,1](W픽셀×1ch)
        let mut ins = 0usize;
        let conv_in = if need_t {
            let t_name = format!("{out}__outer_t");
            g.tensors.insert(
                t_name.clone(),
                TensorInfo { shape: Some(vec![1, 1, w_px, 1]), dtype: OnnxDtype::F32, data: None },
            );
            g.nodes.insert(
                idx,
                Node {
                    op: "transpose".into(),
                    name: format!("{}#outer_t", node.name),
                    attrs: Default::default(),
                    inputs: vec![pv],
                    outputs: vec![t_name.clone()],
                    dead: false,
                },
            );
            ins = 1;
            t_name
        } else {
            pv
        };
        // ② 1×1 Conv (cin=1, cout=C, 가중치 = cvec) — 외적 본체
        let w_name = format!("{out}__outer_w");
        g.add_const(
            w_name.clone(),
            TensorInfo {
                shape: Some(vec![c, 1, 1, 1]),
                dtype: OnnxDtype::F32,
                data: Some(Arc::new(bytemuck::cast_slice(&cvals).to_vec())),
            },
        );
        let tmp_name = format!("{out}__outer_c");
        g.tensors.insert(
            tmp_name.clone(),
            TensorInfo { shape: Some(vec![1, c, w_px, 1]), dtype: OnnxDtype::F32, data: None },
        );
        {
            let n = &mut g.nodes[idx + ins];
            n.op = "Conv".into();
            n.inputs = vec![conv_in, w_name];
            n.outputs = vec![tmp_name.clone()];
            n.attrs.clear();
            n.attrs.insert("kernel_shape".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("strides".into(), Attr::Is(vec![1, 1]));
            n.attrs.insert("pads".into(), Attr::Is(vec![0, 0, 0, 0]));
            n.attrs.insert("group".into(), Attr::I(1));
        }
        // ③ relayout: [1,C,W,1] → 원 출력 (desc 스트림 항등 — 위 stream_ok 검증)
        g.nodes.insert(
            idx + ins + 1,
            Node {
                op: "relayout".into(),
                name: format!("{}#outer_r", node.name),
                attrs: Default::default(),
                inputs: vec![tmp_name],
                outputs: vec![out],
                dead: false,
            },
        );
        report.rewrites += 1;
        idx += ins + 2;
    }
    Ok(report)
}
