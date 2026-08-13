//! .sw 모델의 CPU 레퍼런스 실행기 — GPU 런타임 없이 변환기를 완결 검증한다.
//!
//! ai-core::reference(순진 f32, 논리 NHWC)를 그대로 사용하므로 에필로그 순서
//! (bias→act→residual)가 GPU 커널과 자동으로 일치한다. 가중치는 블롭에서
//! 역패킹(unpack_weights_*)해 쓴다 — pack/unpack 왕복이 여기서 실전 검증된다.

pub mod compare;
pub mod npy;

use std::collections::HashMap;

use ai_core::format::{SwModel, SwOp, SwOperand, WRef};
use ai_core::ops::{AvgPool2d, Conv2d, ResizeBilinear};
use ai_core::{pack, reference, Activation, DType};

use crate::error::ConvertError;

pub struct CpuExec<'a> {
    pub model: &'a SwModel,
    blob: &'a [u8],
    vals: Vec<Option<Vec<f32>>>,
}

impl<'a> CpuExec<'a> {
    pub fn new(model: &'a SwModel, blob: &'a [u8]) -> Self {
        Self { model, blob, vals: vec![None; model.tensors.len()] }
    }

    fn wref(&self, w: ai_core::format::WRef) -> &[u8] {
        &self.blob[w.off as usize..(w.off + w.len) as usize]
    }

    /// NHWC 논리 데이터 주입 (그래프 입력)
    pub fn set_input(&mut self, tid: u32, nhwc: Vec<f32>) {
        self.vals[tid as usize] = Some(nhwc);
    }

    /// alias 해석 포함 읽기 (뷰는 채널 슬라이스로 실체화)
    pub fn read(&self, tid: u32) -> Result<Vec<f32>, ConvertError> {
        let t = &self.model.tensors[tid as usize];
        if let Some(a) = t.alias {
            let backing = self.read(a.of)?;
            let bt = &self.model.tensors[a.of as usize];
            let (bc, c, start) = (bt.c as usize, t.c as usize, (a.cg_off * 4) as usize);
            let px = (t.h * t.w) as usize;
            let mut out = vec![0f32; px * c];
            for p in 0..px {
                out[p * c..(p + 1) * c]
                    .copy_from_slice(&backing[p * bc + start..p * bc + start + c]);
            }
            return Ok(out);
        }
        self.vals[tid as usize]
            .clone()
            .ok_or_else(|| ConvertError::Other(format!("텐서 미계산: {} ({tid})", t.name)))
    }

    fn hw(&self, tid: u32) -> (u32, u32, u32) {
        let t = &self.model.tensors[tid as usize];
        (t.h, t.w, t.c)
    }

    /// 전체 그래프 실행 (입력은 set_input으로 선주입, 상태는 0 초기화)
    pub fn run(&mut self) -> Result<(), ConvertError> {
        // 상태 0 초기화
        for s in &self.model.states {
            let t = &self.model.tensors[s.input as usize];
            if self.vals[s.input as usize].is_none() {
                self.vals[s.input as usize] = Some(vec![0f32; (t.h * t.w * t.c) as usize]);
            }
        }

        let dt = self.model.dt_default;
        assert_eq!(dt, DType::F32, "verify는 f32 모델 전용");

        for (i, op) in self.model.ops.iter().enumerate() {
            let out = self.exec_op(op).map_err(|e| {
                ConvertError::Other(format!("op[{i}] 실행 실패: {e}"))
            })?;
            let (tid, data) = out;
            self.vals[tid as usize] = Some(data);
        }
        Ok(())
    }

    fn exec_op(&self, op: &SwOp) -> Result<(u32, Vec<f32>), ConvertError> {
        Ok(match op {
            SwOp::Conv {
                input,
                out,
                srcs,
                res,
                cin,
                cout,
                kh,
                kw,
                sh,
                sw,
                pad,
                d,
                groups,
                act,
                w,
                b,
                kg_pad,
            } => {
                let (ih, iw, _) = self.hw(*input);
                // concat 융합 conv: 파트를 채널 축으로 이어붙여 단일 입력으로 평가
                let x = if srcs.is_empty() {
                    self.read(*input)?
                } else {
                    let datas: Vec<(Vec<f32>, usize)> = srcs
                        .iter()
                        .map(|p| Ok((self.read(p.input)?, p.c as usize)))
                        .collect::<Result<_, ConvertError>>()?;
                    let px = (ih * iw) as usize;
                    let ctot: usize = datas.iter().map(|(_, c)| c).sum();
                    let mut cat = vec![0f32; px * ctot];
                    for p in 0..px {
                        let mut off = 0usize;
                        for (dv, c) in &datas {
                            cat[p * ctot + off..p * ctot + off + c]
                                .copy_from_slice(&dv[p * c..(p + 1) * c]);
                            off += c;
                        }
                    }
                    cat
                };
                let weights = if *groups > 1 {
                    pack::unpack_weights_dw(self.wref(*w), *cout, *kh, *kw, DType::F32)
                } else {
                    pack::unpack_weights_conv(
                        self.wref(*w),
                        *cout,
                        *cin,
                        *kh,
                        *kw,
                        *kg_pad,
                        DType::F32,
                    )
                };
                let bias = pack::unpack_bias(self.wref(*b), *cout, DType::F32);
                let residual = match res {
                    Some(r) => Some(self.read(*r)?),
                    None => None,
                };
                let conv = Conv2d {
                    cin: *cin,
                    cout: *cout,
                    kh: *kh,
                    kw: *kw,
                    sh: *sh,
                    sw: *sw,
                    pad: *pad,
                    dil: *d,
                    groups: *groups,
                    act: *act,
                };
                let y = reference::conv::conv2d(
                    &conv,
                    ih,
                    iw,
                    &x,
                    &weights,
                    Some(&bias),
                    residual.as_deref(),
                );
                (*out, y)
            }
            SwOp::Binary { a, b, out, op: bop, act } => {
                let (h, w, c) = self.hw(*a);
                let x = self.read(*a)?;
                let y = match b {
                    SwOperand::Tensor { tid } => {
                        reference::elementwise::binary(*bop, &x, &self.read(*tid)?, *act)
                    }
                    SwOperand::Scalar { v, first } => {
                        reference::elementwise::binary_scalar(*bop, &x, *v, *first, *act)
                    }
                    SwOperand::Cvec { w: wref, c: vc } => {
                        let vals = pack::unpack_nhwc(
                            self.wref(*wref),
                            &ai_core::TensorDesc::new(1, 1, *vc, DType::F32),
                        );
                        cvec_apply(*bop, &x, &vals, h, w, c, *act)
                    }
                    SwOperand::CvecTensor { tid } => {
                        let vals = self.read(*tid)?;
                        cvec_apply(*bop, &x, &vals, h, w, c, *act)
                    }
                };
                (*out, y)
            }
            SwOp::Gpool { input, out } => {
                let (h, w, c) = self.hw(*input);
                (*out, reference::pool::global_avg_pool(&self.read(*input)?, h, w, c))
            }
            SwOp::Avgpool { input, out, kh, kw, sh, sw, pad } => {
                let (ih, iw, c) = self.hw(*input);
                let op = AvgPool2d { kh: *kh, kw: *kw, sh: *sh, sw: *sw, pad: *pad };
                (*out, reference::pool::avg_pool(&op, ih, iw, c, &self.read(*input)?))
            }
            SwOp::Maxpool { input, out, kh, kw, sh, sw, pad, pad_c } => {
                let (ih, iw, c) = self.hw(*input);
                let op = ai_core::ops::MaxPool2d {
                    kh: *kh, kw: *kw, sh: *sh, sw: *sw, pad: *pad, pad_c: *pad_c,
                };
                (*out, reference::pool::max_pool(&op, ih, iw, c, &self.read(*input)?))
            }
            SwOp::Resize { input, out, srcs, oh, ow, mode } => {
                let (ih, iw, _) = self.hw(*input);
                // concat 융합 resize: 파트를 채널 concat해 단일 입력으로 평가
                let (x, c) = if srcs.is_empty() {
                    let (_, _, c) = self.hw(*input);
                    (self.read(*input)?, c)
                } else {
                    let datas: Vec<(Vec<f32>, usize)> = srcs
                        .iter()
                        .map(|p| Ok((self.read(p.input)?, p.c as usize)))
                        .collect::<Result<_, ConvertError>>()?;
                    let px = (ih * iw) as usize;
                    let ctot: usize = datas.iter().map(|(_, c)| c).sum();
                    let mut cat = vec![0f32; px * ctot];
                    for p in 0..px {
                        let mut off = 0usize;
                        for (dv, c) in &datas {
                            cat[p * ctot + off..p * ctot + off + c]
                                .copy_from_slice(&dv[p * c..(p + 1) * c]);
                            off += c;
                        }
                    }
                    (cat, ctot as u32)
                };
                let op = ResizeBilinear { oh: *oh, ow: *ow, mode: *mode };
                (*out, reference::resize::resize_bilinear(&op, ih, iw, c, &x))
            }
            SwOp::Concat { out, parts } => {
                let (h, w, oc) = self.hw(*out);
                let px = (h * w) as usize;
                let mut y = vec![0f32; px * oc as usize];
                let datas: Result<Vec<(Vec<f32>, usize)>, ConvertError> = parts
                    .iter()
                    .map(|p| Ok((self.read(p.input)?, p.c as usize)))
                    .collect();
                let datas = datas?;
                for p in 0..px {
                    let mut off = 0usize;
                    for (d, c) in &datas {
                        y[p * oc as usize + off..p * oc as usize + off + c]
                            .copy_from_slice(&d[p * c..(p + 1) * c]);
                        off += c;
                    }
                }
                (*out, y)
            }
            SwOp::Chcopy { input, out, src_c, n } => {
                let (h, w, c) = self.hw(*input);
                let x = self.read(*input)?;
                let px = (h * w) as usize;
                let (s, nn, c) = (*src_c as usize, *n as usize, c as usize);
                let mut y = vec![0f32; px * nn];
                for p in 0..px {
                    y[p * nn..(p + 1) * nn].copy_from_slice(&x[p * c + s..p * c + s + nn]);
                }
                (*out, y)
            }
            SwOp::Act { input, out, act } => {
                let x = self.read(*input)?;
                (*out, x.iter().map(|v| act.apply(*v)).collect())
            }
            SwOp::SeGate { input, out, c_mid, act1, w1, b1, fc2 } => {
                let (h, w, c_in) = self.hw(*input);
                let x = self.read(*input)?;
                // 채널 평균
                let px = (h * w) as usize;
                let cin = c_in as usize;
                let mut mean = vec![0f32; cin];
                for p in 0..px {
                    for ch in 0..cin {
                        mean[ch] += x[p * cin + ch];
                    }
                }
                for m in &mut mean {
                    *m /= px as f32;
                }
                let fc = |xv: &[f32], w: &WRef, b: &WRef, cout: u32, act: Activation| {
                    let cin = xv.len() as u32;
                    let kg_pad = cin.div_ceil(4).next_multiple_of(4);
                    let wts = pack::unpack_weights_conv(
                        self.wref(*w),
                        cout,
                        cin,
                        1,
                        1,
                        kg_pad,
                        DType::F32,
                    );
                    let bias = pack::unpack_bias(self.wref(*b), cout, DType::F32);
                    (0..cout as usize)
                        .map(|o| {
                            let mut acc = bias[o];
                            for i in 0..cin as usize {
                                acc += wts[o * cin as usize + i] * xv[i];
                            }
                            act.apply(acc)
                        })
                        .collect::<Vec<f32>>()
                };
                let mid = fc(&mean, w1, b1, *c_mid, *act1);
                let y = match fc2 {
                    Some(f) => fc(&mid, &f.w, &f.b, f.c_out, f.act),
                    None => mid,
                };
                (*out, y)
            }
            SwOp::Mix { z, a, b, out } => {
                let (av, bv, zv) = (self.read(*a)?, self.read(*b)?, self.read(*z)?);
                let y = (0..av.len()).map(|i| av[i] + zv[i] * (bv[i] - av[i])).collect();
                (*out, y)
            }
        })
    }
}

fn cvec_apply(
    op: ai_core::ops::BinaryOp,
    x: &[f32],
    vec: &[f32],
    h: u32,
    w: u32,
    c: u32,
    act: Activation,
) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for p in 0..(h * w) as usize {
        for ch in 0..c as usize {
            let i = p * c as usize + ch;
            y[i] = act.apply(op.apply(x[i], vec[ch]));
        }
    }
    y
}
