//! 로드 시 계획 — 프레임 루프에 분석·할당이 하나도 남지 않도록 전부 여기서 한다.
//!
//! - SwOp → PlanOp lowering: alias 사전 해석(ViewRef), 가중치 재패킹, 검증
//! - 슬롯 계획: last_use liveness로 중간 텐서 버퍼를 그리디 재사용
//!   (출력 슬롯을 먼저 할당하고 그 다음 죽은 입력을 해제 — in-place 금지)
//! - dtype 규약은 GPU lowering과 동일: std conv 가중치만 `dt_weights`,
//!   dw 가중치·bias·cvec·SeGate는 `dt_default` (ai-gpu-runtime/lowering.rs 참조)
//!
//! 새 op 추가: kernels/<이름>.rs 커널 + 여기 lowering arm 하나 + exec 디스패치 arm.

use std::collections::HashSet;

use ai_core::format::{SwModel, SwOp, SwOperand};
use ai_core::ops::{AvgPool2d, BinaryOp, Conv2d, MaxPool2d, ResizeBilinear};
use ai_core::tensor::TensorDesc;
use ai_core::{pack, Activation, DType};

use crate::kernels::{conv, dw, im2row, pw_dot, segate::SeFc};
use crate::CpuError;

/// 텐서 → 슬롯 뷰 (alias 체인 해석 완료)
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewRef {
    pub slot: usize,
    pub c_off: usize,
    pub stride: usize,
    pub c: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvPartRef {
    pub view: ViewRef,
    pub ic0: usize,
    /// 벡터 루프 경계 (kernels::conv::ConvPart::c4 참조)
    pub c4: usize,
}

pub(crate) enum PlanKind {
    ConvStd {
        op: Conv2d,
        ih: u32,
        iw: u32,
        oh: u32,
        parts: Vec<ConvPartRef>,
        w: usize,
        b: usize,
        res: Option<ViewRef>,
    },
    ConvDw {
        op: Conv2d,
        ih: u32,
        iw: u32,
        oh: u32,
        input: ViewRef,
        w: usize,
        b: usize,
        res: Option<ViewRef>,
    },
    Binary {
        bop: BinaryOp,
        a: ViewRef,
        operand: Operand,
        px: usize,
        act: Activation,
    },
    /// 소채널(cin≤4) k>1 conv — im2row 패치 실체화 후 1x1 GEMM (스템)
    ConvStem {
        /// 원본 conv (im2row 기하)
        op: Conv2d,
        /// 패치 위 1x1 (cin = kh*kw*cin 논리값)
        pw: Conv2d,
        ih: u32,
        iw: u32,
        oh: u32,
        px: usize,
        k_pad: usize,
        input: ViewRef,
        w: usize,
        b: usize,
        res: Option<ViewRef>,
    },
    /// cout ≤ pw_dot::MAX_COUT 1x1 헤드 — 채널 내적 커널
    PwDot {
        op: Conv2d,
        input: ViewRef,
        px: usize,
        w: usize,
        k_pad: usize,
        b: usize,
    },
    Gpool {
        input: ViewRef,
        px: usize,
    },
    Avgpool {
        op: AvgPool2d,
        ih: u32,
        iw: u32,
        input: ViewRef,
    },
    Maxpool {
        op: MaxPool2d,
        ih: u32,
        iw: u32,
        input: ViewRef,
    },
    Resize {
        op: ResizeBilinear,
        ih: u32,
        iw: u32,
        parts: Vec<ViewRef>,
    },
    Concat {
        parts: Vec<ViewRef>,
        px: usize,
        out_stride: usize,
    },
    Chcopy {
        input: ViewRef,
        px: usize,
    },
    Act {
        input: ViewRef,
        px: usize,
        act: Activation,
    },
    Mix {
        z: ViewRef,
        a: ViewRef,
        b: ViewRef,
        px: usize,
    },
    SeGate {
        input: ViewRef,
        px: usize,
        fc1: usize,
        fc2: Option<usize>,
    },
}

pub(crate) enum Operand {
    Tensor(ViewRef),
    Scalar { v: f32, first: bool },
    /// weights 스토어의 [c] 상수 벡터
    CvecConst(usize),
    /// [1,1,C] 런타임 텐서 (SE 게이트 출력 등)
    CvecTensor(ViewRef),
}

/// 실행 한 스텝 — 커널 종류 + 출력 슬롯
pub(crate) struct Step {
    pub kind: PlanKind,
    pub out_slot: usize,
    pub out_len: usize,
}

pub(crate) struct Plan {
    pub steps: Vec<Step>,
    /// 재패킹된 가중치·bias·cvec 상수 (인덱스로 참조)
    pub weights: Vec<Vec<f32>>,
    pub fcs: Vec<SeFc>,
    /// 슬롯별 버퍼 크기 (f32 개수)
    pub slot_len: Vec<usize>,
    /// (이름, 슬롯, 길이)
    pub inputs: Vec<(String, usize, usize)>,
    /// (이름, 뷰, 채널 수, 픽셀 수)
    pub outputs: Vec<(String, ViewRef, usize, usize)>,
    /// 프레임 시작 시 swap할 슬롯 쌍 (state ping-pong)
    pub states: Vec<(usize, usize)>,
}

/// op가 읽는 tid 전부 (liveness 해제용)
fn op_reads(op: &SwOp) -> Vec<u32> {
    match op {
        SwOp::Conv { input, srcs, res, .. } => {
            let mut v: Vec<u32> = if srcs.is_empty() {
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
        SwOp::Gpool { input, .. }
        | SwOp::Avgpool { input, .. }
        | SwOp::Maxpool { input, .. }
        | SwOp::Chcopy { input, .. }
        | SwOp::Act { input, .. }
        | SwOp::SeGate { input, .. } => vec![*input],
        SwOp::Resize { input, srcs, .. } => {
            if srcs.is_empty() {
                vec![*input]
            } else {
                srcs.iter().map(|p| p.input).collect()
            }
        }
        SwOp::Concat { parts, .. } => parts.iter().map(|p| p.input).collect(),
        SwOp::Mix { z, a, b, .. } => vec![*z, *a, *b],
    }
}

fn op_out(op: &SwOp) -> u32 {
    match op {
        SwOp::Conv { out, .. }
        | SwOp::Binary { out, .. }
        | SwOp::Gpool { out, .. }
        | SwOp::Avgpool { out, .. }
        | SwOp::Maxpool { out, .. }
        | SwOp::Resize { out, .. }
        | SwOp::Concat { out, .. }
        | SwOp::Chcopy { out, .. }
        | SwOp::Act { out, .. }
        | SwOp::Mix { out, .. }
        | SwOp::SeGate { out, .. } => *out,
    }
}

struct Builder<'a> {
    sw: &'a SwModel,
    blob: &'a [u8],
    slot_of: Vec<Option<usize>>,
    slot_len: Vec<usize>,
    free: Vec<usize>,
    persistent: HashSet<u32>,
    weights: Vec<Vec<f32>>,
    fcs: Vec<SeFc>,
}

impl<'a> Builder<'a> {
    fn tensor_len(&self, tid: u32) -> usize {
        let t = &self.sw.tensors[tid as usize];
        (t.h * t.w * t.c) as usize
    }

    fn px(&self, tid: u32) -> usize {
        let t = &self.sw.tensors[tid as usize];
        (t.h * t.w) as usize
    }

    /// 백킹 슬롯 신규 확보 (free 리스트 best-fit → 없으면 새 슬롯)
    fn alloc(&mut self, tid: u32) -> usize {
        debug_assert!(self.sw.tensors[tid as usize].alias.is_none(), "alias는 슬롯이 없다");
        let need = self.tensor_len(tid);
        // best-fit: 낭비가 가장 적은 free 슬롯
        let mut best: Option<(usize, usize)> = None; // (free 인덱스, 낭비)
        for (i, &s) in self.free.iter().enumerate() {
            if self.slot_len[s] >= need {
                let waste = self.slot_len[s] - need;
                if best.is_none_or(|(_, w)| waste < w) {
                    best = Some((i, waste));
                }
            }
        }
        let slot = match best {
            Some((i, _)) => self.free.swap_remove(i),
            None => {
                self.slot_len.push(need);
                self.slot_len.len() - 1
            }
        };
        self.slot_of[tid as usize] = Some(slot);
        slot
    }

    fn view_of(&self, tid: u32) -> Result<ViewRef, CpuError> {
        let (backing, cg_off) = self.sw.resolve_alias(tid);
        let bt = &self.sw.tensors[backing as usize];
        let t = &self.sw.tensors[tid as usize];
        let slot = self.slot_of[backing as usize].ok_or_else(|| {
            CpuError::Other(format!("텐서 미생산 상태에서 읽음: {} (tid {tid})", t.name))
        })?;
        Ok(ViewRef {
            slot,
            c_off: (cg_off * 4) as usize,
            stride: bt.c as usize,
            c: t.c as usize,
        })
    }

    /// op i 실행 후 죽는 입력 슬롯 해제
    fn release_dead(&mut self, i: usize, reads: &[u32]) {
        let mut backings: Vec<u32> =
            reads.iter().map(|&t| self.sw.resolve_alias(t).0).collect();
        backings.sort_unstable();
        backings.dedup();
        for b in backings {
            if self.persistent.contains(&b) {
                continue;
            }
            if self.sw.tensors[b as usize].last_use == i as u32 {
                if let Some(s) = self.slot_of[b as usize] {
                    self.free.push(s);
                }
            }
        }
    }

    fn wref(&self, w: ai_core::format::WRef) -> &'a [u8] {
        &self.blob[w.off as usize..(w.off + w.len) as usize]
    }

    fn push_weights(&mut self, data: Vec<f32>) -> usize {
        self.weights.push(data);
        self.weights.len() - 1
    }
}

pub(crate) fn build(sw: &SwModel, blob: &[u8]) -> Result<Plan, CpuError> {
    let dt = sw.dt_default;
    let wdt = sw.dt_weights.unwrap_or(dt);

    let mut b = Builder {
        sw,
        blob,
        slot_of: vec![None; sw.tensors.len()],
        slot_len: vec![],
        free: vec![],
        persistent: HashSet::new(),
        weights: vec![],
        fcs: vec![],
    };

    // 영속 텐서: 그래프 입출력 + 상태 쌍 (백킹 기준)
    for &t in sw.inputs.iter().chain(&sw.outputs) {
        b.persistent.insert(sw.resolve_alias(t).0);
    }
    for s in &sw.states {
        b.persistent.insert(sw.resolve_alias(s.input).0);
        b.persistent.insert(sw.resolve_alias(s.output).0);
    }
    // 입력·상태 입력은 op 이전에 존재해야 한다 → 선할당
    for &t in &sw.inputs {
        let (backing, _) = sw.resolve_alias(t);
        if b.slot_of[backing as usize].is_none() {
            b.alloc(backing);
        }
    }

    let mut steps = Vec::with_capacity(sw.ops.len());
    for (i, op) in sw.ops.iter().enumerate() {
        let out_tid = op_out(op);
        let reads = op_reads(op);

        // lowering (입력 뷰는 할당 전에 — 출력이 입력 슬롯을 못 뺏도록 alloc이 뒤)
        let kind = lower(&mut b, op, dt, wdt)?;

        let (out_backing, out_cg) = sw.resolve_alias(out_tid);
        if out_cg != 0 || out_backing != out_tid {
            return Err(CpuError::Unsupported(format!(
                "op 출력이 alias: tid {out_tid}"
            )));
        }
        let out_slot = match b.slot_of[out_tid as usize] {
            Some(s) => s, // 상태 출력 등 선할당된 경우
            None => b.alloc(out_tid),
        };
        b.release_dead(i, &reads);

        steps.push(Step { kind, out_slot, out_len: b.tensor_len(out_tid) });
    }

    // 상태 쌍 → 슬롯 쌍 (크기 동일 필수)
    let mut states = Vec::with_capacity(sw.states.len());
    for s in &sw.states {
        let (bi, ci) = sw.resolve_alias(s.input);
        let (bo, co) = sw.resolve_alias(s.output);
        if ci != 0 || co != 0 {
            return Err(CpuError::Unsupported("상태 텐서가 alias".into()));
        }
        let (si, so) = (
            b.slot_of[bi as usize].ok_or_else(|| CpuError::Other("상태 입력 미할당".into()))?,
            b.slot_of[bo as usize].ok_or_else(|| CpuError::Other("상태 출력 미생산".into()))?,
        );
        if b.tensor_len(bi) != b.tensor_len(bo) {
            return Err(CpuError::Other("상태 쌍 크기 불일치".into()));
        }
        // swap은 버퍼 전체를 바꾸므로 슬롯 용량도 같아야 한다
        let cap = b.slot_len[si].max(b.slot_len[so]);
        b.slot_len[si] = cap;
        b.slot_len[so] = cap;
        states.push((si, so));
    }

    let inputs = sw
        .inputs
        .iter()
        .map(|&t| {
            let name = sw.tensors[t as usize].name.clone();
            (name, b.slot_of[t as usize].unwrap(), b.tensor_len(t))
        })
        .collect();
    let outputs = sw
        .outputs
        .iter()
        .map(|&t| {
            let name = sw.tensors[t as usize].name.clone();
            let view = b.view_of(t)?;
            Ok((name, view, view.c, b.px(t)))
        })
        .collect::<Result<Vec<_>, CpuError>>()?;

    Ok(Plan {
        steps,
        weights: b.weights,
        fcs: b.fcs,
        slot_len: b.slot_len,
        inputs,
        outputs,
        states,
    })
}

fn lower(b: &mut Builder, op: &SwOp, dt: DType, wdt: DType) -> Result<PlanKind, CpuError> {
    Ok(match op {
        SwOp::Conv {
            input, out, srcs, res, cin, cout, kh, kw, sh, sw: swi, pad, d, groups, act, w,
            b: bias, kg_pad,
        } => {
            let it = &b.sw.tensors[*input as usize];
            let ot = &b.sw.tensors[*out as usize];
            let (ih, iw) = (it.h, it.w);
            let conv2d = Conv2d {
                cin: *cin,
                cout: *cout,
                kh: *kh,
                kw: *kw,
                sh: *sh,
                sw: *swi,
                pad: *pad,
                dil: *d,
                groups: *groups,
                act: *act,
            };
            let (oh, ow) = conv2d.out_hw(ih, iw);
            if (oh, ow) != (ot.h, ot.w) {
                return Err(CpuError::Other(format!(
                    "conv 출력 크기 불일치: 계산 {oh}x{ow} vs 텐서 {}x{}",
                    ot.h, ot.w
                )));
            }
            let res_view = res.map(|r| b.view_of(r)).transpose()?;
            let bias_v = pack::unpack_bias(b.wref(*bias), *cout, dt);

            if *groups > 1 {
                if *groups != *cin || *cin != *cout {
                    return Err(CpuError::Unsupported(format!(
                        "그룹 conv(g={groups}, cin={cin}, cout={cout})"
                    )));
                }
                if !srcs.is_empty() {
                    // GPU lowering과 동일 제약 — 변환기가 방출하지 않는 조합
                    return Err(CpuError::Unsupported("dw conv concat 융합".into()));
                }
                let w_logical = pack::unpack_weights_dw(b.wref(*w), *cout, *kh, *kw, dt);
                let (wts, c_pad) = dw::repack_weights(&w_logical, *cout, *kh, *kw);
                let w_idx = b.push_weights(wts);
                let b_idx = b.push_weights(conv::pad_bias(&bias_v, c_pad));
                PlanKind::ConvDw {
                    op: conv2d,
                    ih,
                    iw,
                    oh,
                    input: b.view_of(*input)?,
                    w: w_idx,
                    b: b_idx,
                    res: res_view,
                }
            } else if srcs.is_empty()
                && (*cin as usize) <= 4
                && (*kh * *kw) > 1
                && *d == 1
                && {
                    let v = b.view_of(*input)?;
                    v.c_off == 0 && v.stride == v.c
                }
            {
                // 스템(cin≤4, k>1): im2row로 패치를 펴서 1x1 GEMM으로 —
                // tap당 오버헤드가 4레인 하나뿐인 일을 지배하던 경로 (im2row.rs)
                let v = b.view_of(*input)?;
                let w_oihw = pack::unpack_weights_conv(
                    b.wref(*w), *cout, *cin, *kh, *kw, *kg_pad, wdt,
                );
                let w_perm = im2row::permute_weights(&w_oihw, *cout, *cin, *kh, *kw);
                let k = (*kh * *kw * *cin) as usize;
                let k_pad = k.next_multiple_of(4);
                let (wts, cout_pad) =
                    conv::repack_weights(&w_perm, *cout, k as u32, k_pad, 1, 1);
                let w_idx = b.push_weights(wts);
                let b_idx = b.push_weights(conv::pad_bias(&bias_v, cout_pad));
                PlanKind::ConvStem {
                    op: conv2d,
                    pw: Conv2d::pointwise(k as u32, *cout, *act),
                    ih,
                    iw,
                    oh,
                    px: (oh * ow) as usize,
                    k_pad,
                    input: v,
                    w: w_idx,
                    b: b_idx,
                    res: res_view,
                }
            } else if *kh == 1
                && *kw == 1
                && *sh == 1
                && *swi == 1
                && srcs.is_empty()
                && res.is_none()
                && (*cout as usize) <= pw_dot::MAX_COUT
            {
                // 세그 헤드류(cout 1~4) — NR8 GEMM은 레인을 버리므로 내적 커널
                let w_oihw = pack::unpack_weights_conv(
                    b.wref(*w), *cout, *cin, 1, 1, *kg_pad, wdt,
                );
                let (wts, k_pad) = pw_dot::repack_weights(&w_oihw, *cout, *cin);
                let w_idx = b.push_weights(wts);
                let b_idx = b.push_weights(bias_v.clone());
                PlanKind::PwDot {
                    op: conv2d,
                    input: b.view_of(*input)?,
                    px: (oh * ow) as usize,
                    w: w_idx,
                    k_pad,
                    b: b_idx,
                }
            } else {
                let w_oihw = pack::unpack_weights_conv(
                    b.wref(*w), *cout, *cin, *kh, *kw, *kg_pad, wdt,
                );
                // 단일 파트 & c%4≠0 (스템 cin=3 등): K축을 4배수로 제로패딩해
                // 벡터 경로에 태운다 — 4레인 로드의 초과분은 가중치 0이 지운다
                // (마지막 픽셀의 초과 읽기는 슬롯 +4 패딩이 보장, exec.rs 참조).
                let cin_pad = if srcs.is_empty() && (*cin as usize) % 4 != 0 {
                    (*cin as usize).next_multiple_of(4)
                } else {
                    *cin as usize
                };
                let (wts, cout_pad) =
                    conv::repack_weights(&w_oihw, *cout, *cin, cin_pad, *kh, *kw);
                let w_idx = b.push_weights(wts);
                let b_idx = b.push_weights(conv::pad_bias(&bias_v, cout_pad));
                let mut parts = Vec::new();
                let mut ic0 = 0usize;
                if srcs.is_empty() {
                    let v = b.view_of(*input)?;
                    parts.push(ConvPartRef { view: v, ic0: 0, c4: cin_pad });
                } else {
                    for p in srcs {
                        let v = b.view_of(p.input)?;
                        if v.c != p.c as usize {
                            return Err(CpuError::Other("concat 파트 채널 불일치".into()));
                        }
                        parts.push(ConvPartRef { view: v, ic0, c4: v.c });
                        ic0 += v.c;
                    }
                }
                PlanKind::ConvStd {
                    op: conv2d,
                    ih,
                    iw,
                    oh,
                    parts,
                    w: w_idx,
                    b: b_idx,
                    res: res_view,
                }
            }
        }
        SwOp::Binary { a, b: operand, out, op: bop, act } => {
            let operand = match operand {
                SwOperand::Tensor { tid } => Operand::Tensor(b.view_of(*tid)?),
                SwOperand::Scalar { v, first } => Operand::Scalar { v: *v, first: *first },
                SwOperand::Cvec { w, c } => {
                    let vals = pack::unpack_nhwc(
                        b.wref(*w),
                        &TensorDesc::new(1, 1, *c, dt),
                    );
                    Operand::CvecConst(b.push_weights(vals))
                }
                SwOperand::CvecTensor { tid } => Operand::CvecTensor(b.view_of(*tid)?),
            };
            PlanKind::Binary {
                bop: *bop,
                a: b.view_of(*a)?,
                operand,
                px: b.px(*out),
                act: *act,
            }
        }
        SwOp::Gpool { input, .. } => PlanKind::Gpool {
            input: b.view_of(*input)?,
            px: b.px(*input),
        },
        SwOp::Avgpool { input, kh, kw, sh, sw: swi, pad, .. } => {
            let it = &b.sw.tensors[*input as usize];
            PlanKind::Avgpool {
                op: AvgPool2d { kh: *kh, kw: *kw, sh: *sh, sw: *swi, pad: *pad },
                ih: it.h,
                iw: it.w,
                input: b.view_of(*input)?,
            }
        }
        SwOp::Maxpool { input, kh, kw, sh, sw: swi, pad, pad_c, .. } => {
            let it = &b.sw.tensors[*input as usize];
            PlanKind::Maxpool {
                op: MaxPool2d {
                    kh: *kh, kw: *kw, sh: *sh, sw: *swi, pad: *pad, pad_c: *pad_c,
                },
                ih: it.h,
                iw: it.w,
                input: b.view_of(*input)?,
            }
        }
        SwOp::Resize { input, srcs, oh, ow, mode, .. } => {
            let it = &b.sw.tensors[*input as usize];
            let parts = if srcs.is_empty() {
                vec![b.view_of(*input)?]
            } else {
                srcs.iter().map(|p| b.view_of(p.input)).collect::<Result<_, _>>()?
            };
            PlanKind::Resize {
                op: ResizeBilinear { oh: *oh, ow: *ow, mode: *mode },
                ih: it.h,
                iw: it.w,
                parts,
            }
        }
        SwOp::Concat { out, parts } => {
            let views = parts
                .iter()
                .map(|p| b.view_of(p.input))
                .collect::<Result<Vec<_>, _>>()?;
            PlanKind::Concat {
                parts: views,
                px: b.px(*out),
                out_stride: b.sw.tensors[*out as usize].c as usize,
            }
        }
        SwOp::Chcopy { input, out, src_c, n } => {
            let mut v = b.view_of(*input)?;
            v.c_off += *src_c as usize;
            v.c = *n as usize;
            let _ = out;
            PlanKind::Chcopy { input: v, px: b.px(*input) }
        }
        SwOp::Act { input, out, act } => PlanKind::Act {
            input: b.view_of(*input)?,
            px: b.px(*out),
            act: *act,
        },
        SwOp::Mix { z, a, b: bb, out } => PlanKind::Mix {
            z: b.view_of(*z)?,
            a: b.view_of(*a)?,
            b: b.view_of(*bb)?,
            px: b.px(*out),
        },
        SwOp::SeGate { input, c_mid, act1, w1, b1, fc2, .. } => {
            let cin = b.sw.tensors[*input as usize].c;
            let mk_fc = |b: &Builder, w: ai_core::format::WRef, bias: ai_core::format::WRef,
                         cin: u32, cout: u32, act: Activation| {
                let kg_pad = cin.div_ceil(4).next_multiple_of(4);
                SeFc {
                    w: pack::unpack_weights_conv(b.wref(w), cout, cin, 1, 1, kg_pad, dt),
                    b: pack::unpack_bias(b.wref(bias), cout, dt),
                    act,
                }
            };
            let fc1 = mk_fc(b, *w1, *b1, cin, *c_mid, *act1);
            let fc2 = fc2.as_ref().map(|f| mk_fc(b, f.w, f.b, *c_mid, f.c_out, f.act));
            b.fcs.push(fc1);
            let fc1_idx = b.fcs.len() - 1;
            let fc2_idx = fc2.map(|f| {
                b.fcs.push(f);
                b.fcs.len() - 1
            });
            PlanKind::SeGate {
                input: b.view_of(*input)?,
                px: b.px(*input),
                fc1: fc1_idx,
                fc2: fc2_idx,
            }
        }
    })
}
