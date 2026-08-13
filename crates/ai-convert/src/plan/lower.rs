//! IR → SwModel + 블롭.
//!
//! - NCHW `[1,C,H,W]` → `SwTensor{h,w,c}` (물리 레이아웃은 런타임의 NHWC-C4)
//! - 가중치는 여기서 ai-core::pack으로 **최종 커널 레이아웃으로 사전 패킹** — 런타임 로드 = memcpy
//! - chview 노드/alias_of 항목 → SwTensor.alias (op 아님)
//! - last_use 1스캔 계산 (뷰가 백킹 수명 연장), 그래프 출력·상태 출력은 끝까지 생존

use std::collections::HashMap;

use ai_core::format::{SeFc, SwAlias, SwConcatPart, SwModel, SwOp, SwOperand, SwSize, SwState, SwTensor};
use ai_core::ops::{BinaryOp, CoordMode};
use ai_core::{pack, Activation, DType, TensorDesc};

use crate::emit::blob::BlobBuilder;
use crate::error::ConvertError;
use crate::ir::{Graph, Node};
use crate::passes::Ctx;

fn parse_act(s: Option<&str>) -> Result<Activation, ConvertError> {
    Ok(match s {
        None | Some("none") => Activation::None,
        Some("relu") => Activation::Relu,
        Some("sigmoid") => Activation::Sigmoid,
        Some("tanh") => Activation::Tanh,
        Some("hswish") => Activation::Hardswish,
        Some("hsigmoid") => Activation::Hardsigmoid,
        Some("clamp01") => Activation::Clamp01,
        Some(other) => return Err(ConvertError::Malformed(format!("알 수 없는 act: {other}"))),
    })
}

/// 단독 unary op명 → act
fn unary_act(op: &str) -> Option<Activation> {
    Some(match op {
        "Relu" => Activation::Relu,
        "Sigmoid" => Activation::Sigmoid,
        "Tanh" => Activation::Tanh,
        "hswish" => Activation::Hardswish,
        "HardSigmoid" => Activation::Hardsigmoid,
        _ => return None,
    })
}

struct Lowerer<'a> {
    g: &'a Graph,
    /// 활성화 dtype
    dt: DType,
    /// std conv 가중치 dtype (dw 가중치·bias는 `dt`를 쓴다 — BN 접힘 레인지)
    wdt: DType,
    tids: HashMap<String, u32>,
    tensors: Vec<SwTensor>,
    blob: BlobBuilder,
}

impl<'a> Lowerer<'a> {
    fn desc_of(&self, name: &str) -> Result<(u32, u32, u32), ConvertError> {
        let s = self
            .g
            .info(name)
            .and_then(|t| t.static_shape())
            .ok_or_else(|| ConvertError::ShapeUnresolved(name.into()))?;
        match s.len() {
            4 => Ok((s[2] as u32, s[3] as u32, s[1] as u32)),
            3 if s[1] == 1 && s[2] == 1 => Ok((1, 1, s[0] as u32)), // [C,1,1] 채널벡터
            2 => Ok((1, 1, s[1] as u32)),
            1 => Ok((1, 1, s[0] as u32)),
            _ => Err(ConvertError::Malformed(format!("텐서 rank {} 미지원: {name}", s.len()))),
        }
    }

    fn tid(&mut self, name: &str) -> Result<u32, ConvertError> {
        if let Some(t) = self.tids.get(name) {
            return Ok(*t);
        }
        let (h, w, c) = self.desc_of(name)?;
        let t = self.tensors.len() as u32;
        self.tensors.push(SwTensor {
            name: name.to_string(),
            h,
            w,
            c,
            dt: self.dt,
            alias: None,
            last_use: 0,
        });
        self.tids.insert(name.to_string(), t);
        Ok(t)
    }

    fn const_f32s(&self, name: &str) -> Result<Vec<f32>, ConvertError> {
        self.g
            .info(name)
            .and_then(|t| t.as_f32s().map(|v| v.to_vec()))
            .ok_or_else(|| ConvertError::Malformed(format!("상수 f32 아님: {name}")))
    }

    fn lower_conv(&mut self, node: &Node) -> Result<SwOp, ConvertError> {
        let x = node.inputs[0].clone();
        // concat-into-conv 융합: 추가 파트는 inputs 끝에 붙어 있다 (fuse_concat 참조)
        let src_cs = node.attr_is("src_cs").map(|v| v.to_vec());
        let n_extra = src_cs.as_ref().map(|v| v.len() - 1).unwrap_or(0);
        let base_len = node.inputs.len() - n_extra;
        let mut srcs = Vec::new();
        if let Some(cs) = &src_cs {
            srcs.push(SwConcatPart { input: self.tid(&x)?, c: cs[0] as u32 });
            for (i, c) in cs[1..].iter().enumerate() {
                let name = node.inputs[base_len + i].clone();
                srcs.push(SwConcatPart { input: self.tid(&name)?, c: *c as u32 });
            }
        }
        let w_name = &node.inputs[1];
        let w_shape = self
            .g
            .info(w_name)
            .and_then(|t| t.static_shape().map(|s| s.to_vec()))
            .ok_or_else(|| ConvertError::Malformed(format!("가중치 shape 없음: {w_name}")))?;
        let (cout, cin_g, kh, kw) =
            (w_shape[0] as u32, w_shape[1] as u32, w_shape[2] as u32, w_shape[3] as u32);
        let group = node.attr_i("group").unwrap_or(1) as u32;
        let strides = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
        let pads = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0; 4]);
        let dil = node.attr_is("dilations").map(|v| v[0]).unwrap_or(1) as u32;
        let act = parse_act(node.attr_s("act"))?;

        let wvals = self.const_f32s(w_name)?;
        let is_dw = group > 1 && cin_g == 1 && cout == group;
        let (wbytes, kg_pad, cin) = if is_dw {
            (pack::pack_weights_dw(&wvals, cout, kh, kw, self.dt), 0, cout)
        } else if group == 1 {
            let cin = cin_g;
            let (b, kg) = pack::pack_weights_conv(&wvals, cout, cin, kh, kw, 4, self.wdt);
            (b, kg, cin)
        } else {
            return Err(ConvertError::Unsupported(vec![format!(
                "grouped conv g={group} ({})",
                node.name
            )]));
        };
        let w = self.blob.push(&wbytes);

        let bias = match (base_len >= 3).then(|| &node.inputs[2]) {
            Some(b) => self.const_f32s(b)?,
            None => vec![0.0; cout as usize],
        };
        let b = self.blob.push(&pack::pack_bias(&bias, cout, self.dt));

        let res = match node.attr_s("res") {
            Some(r) => {
                let r = r.to_string();
                Some(self.tid(&r)?)
            }
            None => None,
        };

        Ok(SwOp::Conv {
            input: self.tid(&x)?,
            out: self.tid(&node.outputs[0])?,
            srcs,
            res,
            cin,
            cout,
            kh,
            kw,
            sh: strides[0] as u32,
            sw: strides[1] as u32,
            pad: [pads[0] as u32, pads[1] as u32, pads[2] as u32, pads[3] as u32],
            d: dil,
            groups: if is_dw { cout } else { 1 },
            act,
            w,
            b,
            kg_pad,
        })
    }

    fn lower_binary(&mut self, node: &Node) -> Result<SwOp, ConvertError> {
        let op = match node.op.as_str() {
            "Mul" => BinaryOp::Mul,
            "Add" => BinaryOp::Add,
            "Sub" => BinaryOp::Sub,
            other => return Err(ConvertError::Malformed(format!("binary 아님: {other}"))),
        };
        let act = parse_act(node.attr_s("act"))?;
        let a_name = node.inputs[0].clone();
        let a = self.tid(&a_name)?;

        let b = if let Some(v) = node.attr_f("scalar") {
            SwOperand::Scalar { v, first: node.attr_i("scalar_first").unwrap_or(0) == 1 }
        } else if node.attrs.contains_key("cvec") {
            let bn = node.inputs[1].clone();
            let (_, _, c) = self.desc_of(&bn)?;
            if self.g.is_const(&bn) {
                let vals = self.const_f32s(&bn)?;
                let packed = pack::pack_nhwc(&vals, &TensorDesc::new(1, 1, c, self.dt));
                SwOperand::Cvec { w: self.blob.push(&packed), c }
            } else {
                SwOperand::CvecTensor { tid: self.tid(&bn)? }
            }
        } else {
            let bn = node.inputs[1].clone();
            SwOperand::Tensor { tid: self.tid(&bn)? }
        };

        Ok(SwOp::Binary { a, b, out: self.tid(&node.outputs[0])?, op, act })
    }
}

/// 패스 완료된 그래프 → (.sw 모델, 블롭)
pub fn lower(g: &Graph, ctx: &Ctx, name: &str) -> Result<(SwModel, Vec<u8>), ConvertError> {
    let dt = if ctx.fp16 { DType::F16 } else { DType::F32 };
    // 가중치만 f16: 활성화 f32를 유지하면서 std conv 가중치 트래픽만 절반으로
    let wdt = if ctx.fp16 || ctx.fp16_weights { DType::F16 } else { DType::F32 };
    let mut lw =
        Lowerer { g, dt, wdt, tids: HashMap::new(), tensors: Vec::new(), blob: BlobBuilder::new() };

    // 그래프 입력 먼저 등록 (tid 안정성)
    for i in &g.inputs {
        lw.tid(i)?;
    }

    let mut ops: Vec<SwOp> = Vec::new();
    for (_, node) in g.live_nodes() {
        match node.op.as_str() {
            "chview" => {
                // 뷰 = 텐서 속성
                let src = lw.tid(&node.inputs[0])?;
                let cg_off = node.attr_i("cg_off").unwrap_or(0) as u32;
                let out = lw.tid(&node.outputs[0])?;
                lw.tensors[out as usize].alias = Some(SwAlias { of: src, cg_off });
            }
            "Conv" => ops.push(lw.lower_conv(node)?),
            "Mul" | "Add" | "Sub" => ops.push(lw.lower_binary(node)?),
            "mix" => ops.push(SwOp::Mix {
                a: lw.tid(&node.inputs[0])?,
                b: lw.tid(&node.inputs[1])?,
                z: lw.tid(&node.inputs[2])?,
                out: lw.tid(&node.outputs[0])?,
            }),
            "gpool" => ops.push(SwOp::Gpool {
                input: lw.tid(&node.inputs[0])?,
                out: lw.tid(&node.outputs[0])?,
            }),
            "avgpool" => {
                let k = node.attr_is("kernel_shape").unwrap().to_vec();
                let s = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
                let p = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0; 4]);
                ops.push(SwOp::Avgpool {
                    input: lw.tid(&node.inputs[0])?,
                    out: lw.tid(&node.outputs[0])?,
                    kh: k[0] as u32,
                    kw: k[1] as u32,
                    sh: s[0] as u32,
                    sw: s[1] as u32,
                    pad: [p[0] as u32, p[1] as u32, p[2] as u32, p[3] as u32],
                });
            }
            "resize" => {
                let mode = match node.attr_s("mode") {
                    Some("half_pixel") => CoordMode::HalfPixel,
                    Some("asymmetric") => CoordMode::Asymmetric,
                    other => {
                        return Err(ConvertError::Malformed(format!("resize mode {other:?}")))
                    }
                };
                let mut srcs = Vec::new();
                if let Some(cs) = node.attr_is("src_cs") {
                    let cs = cs.to_vec();
                    srcs.push(SwConcatPart { input: lw.tid(&node.inputs[0])?, c: cs[0] as u32 });
                    for (i, c) in cs[1..].iter().enumerate() {
                        let name = node.inputs[1 + i].clone();
                        srcs.push(SwConcatPart { input: lw.tid(&name)?, c: *c as u32 });
                    }
                }
                ops.push(SwOp::Resize {
                    input: lw.tid(&node.inputs[0])?,
                    out: lw.tid(&node.outputs[0])?,
                    srcs,
                    oh: node.attr_i("oh").unwrap() as u32,
                    ow: node.attr_i("ow").unwrap() as u32,
                    mode,
                });
            }
            "Concat" => {
                let axis = node.attr_i("axis").unwrap_or(1);
                if axis != 1 && axis != -3 {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "채널 축 아닌 Concat axis={axis} ({})",
                        node.name
                    )]));
                }
                let mut parts = Vec::new();
                for i in &node.inputs {
                    let (_, _, c) = lw.desc_of(i)?;
                    parts.push(SwConcatPart { input: lw.tid(i)?, c });
                }
                ops.push(SwOp::Concat { out: lw.tid(&node.outputs[0])?, parts });
            }
            "segate" => {
                let x = node.inputs[0].clone();
                let (_, _, cin) = lw.desc_of(&x)?;
                let c_mid = node.attr_i("c_mid").unwrap() as u32;
                let act1 = parse_act(node.attr_s("act1"))?;
                let w1v = lw.const_f32s(&node.inputs[1])?;
                let (w1b, _) = pack::pack_weights_conv(&w1v, c_mid, cin, 1, 1, 4, lw.wdt);
                let w1 = lw.blob.push(&w1b);
                let b1v = lw.const_f32s(&node.inputs[2])?;
                let b1 = lw.blob.push(&pack::pack_bias(&b1v, c_mid, lw.dt));
                let fc2 = match node.attr_i("c_out") {
                    Some(c_out) => {
                        let c_out = c_out as u32;
                        let act2 = parse_act(node.attr_s("act2"))?;
                        let w2v = lw.const_f32s(&node.inputs[3])?;
                        let (w2b, _) = pack::pack_weights_conv(&w2v, c_out, c_mid, 1, 1, 4, lw.wdt);
                        let w2 = lw.blob.push(&w2b);
                        let b2v = lw.const_f32s(&node.inputs[4])?;
                        let b2 = lw.blob.push(&pack::pack_bias(&b2v, c_out, lw.dt));
                        Some(SeFc { c_out, act: act2, w: w2, b: b2 })
                    }
                    None => None,
                };
                ops.push(SwOp::SeGate {
                    input: lw.tid(&x)?,
                    out: lw.tid(&node.outputs[0])?,
                    c_mid,
                    act1,
                    w1,
                    b1,
                    fc2,
                });
            }
            "chcopy" => ops.push(SwOp::Chcopy {
                input: lw.tid(&node.inputs[0])?,
                out: lw.tid(&node.outputs[0])?,
                src_c: node.attr_i("src_c").unwrap() as u32,
                n: node.attr_i("n").unwrap() as u32,
            }),
            "act" => ops.push(SwOp::Act {
                input: lw.tid(&node.inputs[0])?,
                out: lw.tid(&node.outputs[0])?,
                act: parse_act(node.attr_s("act"))?,
            }),
            other => {
                if let Some(act) = unary_act(other) {
                    ops.push(SwOp::Act {
                        input: lw.tid(&node.inputs[0])?,
                        out: lw.tid(&node.outputs[0])?,
                        act,
                    });
                } else {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "lowering 불가 op {other} ({})",
                        node.name
                    )]));
                }
            }
        }
    }

    // alias_of 항목 (그래프 출력 이름 보존용 순수 별칭)
    let alias_entries: Vec<(String, String)> =
        g.alias_of.iter().map(|(a, s)| (a.clone(), s.clone())).collect();
    for (alias_name, _) in &alias_entries {
        // desc는 alias 이름 기준 (IR에 shape 있음)
        if lw.tids.contains_key(alias_name.as_str()) {
            continue;
        }
        let src = g.resolve_alias(alias_name).to_string();
        let src_tid = lw.tid(&src)?;
        // desc는 루트 기준 — 경계 Transpose alias의 IR shape은 호출자 레이아웃(NHWC)이라
        // 내부 기하가 아니다 (이걸 쓰면 pure-rename이 뷰로 오판된다)
        let (h, w, c) = lw.desc_of(&src).or_else(|_| lw.desc_of(alias_name))?;
        let t = lw.tensors.len() as u32;
        lw.tensors.push(SwTensor {
            name: alias_name.clone(),
            h,
            w,
            c,
            dt,
            alias: Some(SwAlias { of: src_tid, cg_off: 0 }),
            last_use: 0,
        });
        lw.tids.insert(alias_name.clone(), t);
    }

    // 입출력·상태 tid
    let inputs: Vec<u32> = g.inputs.iter().map(|i| lw.tids[i.as_str()]).collect();
    let outputs: Result<Vec<u32>, ConvertError> =
        g.outputs.iter().map(|o| lw.tid(o)).collect();
    let outputs = outputs?;
    let states: Vec<SwState> = g
        .states
        .iter()
        .map(|(i, o)| SwState { input: lw.tids[i.as_str()], output: lw.tids[o.as_str()] })
        .collect();

    // last_use 계산
    let mut tensors = lw.tensors;
    let op_inputs = |op: &SwOp| -> Vec<u32> {
        match op {
            SwOp::Conv { input, srcs, res, .. } => {
                let mut v = if srcs.is_empty() {
                    vec![*input]
                } else {
                    srcs.iter().map(|p| p.input).collect()
                };
                if let Some(r) = res {
                    v.push(*r);
                }
                v
            }
            SwOp::Binary { a, b, .. } => {
                let mut v = vec![*a];
                match b {
                    SwOperand::Tensor { tid } | SwOperand::CvecTensor { tid } => v.push(*tid),
                    _ => {}
                }
                v
            }
            SwOp::Resize { input, srcs, .. } => {
                if srcs.is_empty() {
                    vec![*input]
                } else {
                    srcs.iter().map(|p| p.input).collect()
                }
            }
            SwOp::Gpool { input, .. }
            | SwOp::Avgpool { input, .. }
            | SwOp::Chcopy { input, .. }
            | SwOp::Act { input, .. } => vec![*input],
            SwOp::Concat { parts, .. } => parts.iter().map(|p| p.input).collect(),
            SwOp::SeGate { input, .. } => vec![*input],
            SwOp::Mix { z, a, b, .. } => vec![*z, *a, *b],
        }
    };
    for (i, op) in ops.iter().enumerate() {
        for t in op_inputs(op) {
            tensors[t as usize].last_use = tensors[t as usize].last_use.max(i as u32);
        }
    }
    // 출력·상태출력은 끝까지 생존
    let end = ops.len() as u32;
    for &t in outputs.iter() {
        tensors[t as usize].last_use = end;
    }
    // 뷰 수명을 백킹으로 전파
    let alias_pairs: Vec<(usize, u32)> = tensors
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.alias.map(|a| (i, a.of)))
        .collect();
    for (view, mut of) in alias_pairs {
        let lu = tensors[view].last_use;
        loop {
            tensors[of as usize].last_use = tensors[of as usize].last_use.max(lu);
            match tensors[of as usize].alias {
                Some(a) => of = a.of,
                None => break,
            }
        }
    }

    // 이미지 입력 크기
    let size = inputs
        .first()
        .map(|&t| SwSize { h: tensors[t as usize].h, w: tensors[t as usize].w })
        .unwrap_or(SwSize { h: 0, w: 0 });

    let model = SwModel {
        name: name.to_string(),
        size,
        dt_default: dt,
        dt_weights: Some(wdt),
        tensors,
        inputs,
        outputs,
        states,
        ops,
    };
    Ok((model, lw.blob.finish()))
}
