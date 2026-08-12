//! SwOp → KernelSpec + 바인딩 기술 — 런타임 lowering.
//!
//! 커널 선택: Conv{groups>1→dw, k==1→pw GEMM, 그 외→implicit GEMM},
//! Binary→elementwise, Act→elementwise Unary, Concat/Chcopy/뷰 실체화→channel_gather.

use ai_core::format::{SwModel, SwOp, SwOperand, WRef};
use ai_core::ops::{BinaryOp, Conv2d};
use ai_core::Activation;
use ai_gpu::kernel::KernelSpec;
use ai_gpu::kernels::avgpool::AvgPoolSpec;
use ai_gpu::kernels::channel_gather::{ChannelGatherSpec, GatherPart};
use ai_gpu::kernels::conv_dw::ConvDwSpec;
use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
use ai_gpu::kernels::elementwise::{ElementwiseSpec, EwOperand};
use ai_gpu::kernels::gemm_pw::GemmPwSpec;
use ai_gpu::kernels::gpool::GpoolSpec;
use ai_gpu::kernels::resize_bilinear::ResizeBilinearSpec;

use crate::error::RuntimeError;

/// storage 바인딩의 출처 (binding 1..)
#[derive(Clone, Debug)]
pub enum RtBinding {
    /// 활성화 텐서 슬롯 (tid — 모델 로드가 슬롯/parity 해석)
    Tensor(u32),
    /// 가중치 블롭 오프셋 바인딩
    Weights(WRef),
}

pub struct LoweredOp {
    pub spec: Box<dyn KernelSpec>,
    pub bindings: Vec<RtBinding>,
    pub params: [u8; 16],
    pub label: String,
    /// liveness용 (tid)
    pub reads: Vec<u32>,
    pub writes: Vec<u32>,
}

fn ew_params(scalar: f32, cg: u32, len_vec4: u32) -> [u8; 16] {
    let mut p = [0u8; 16];
    p[0..4].copy_from_slice(&scalar.to_le_bytes());
    p[4..8].copy_from_slice(&cg.to_le_bytes());
    p[12..16].copy_from_slice(&len_vec4.to_le_bytes());
    p
}

fn tdesc(sw: &SwModel, tid: u32) -> (u32, u32, u32) {
    let t = &sw.tensors[tid as usize];
    (t.h, t.w, t.c)
}

/// SwOp 하나를 lowering. `map` = tid → 실효 tid (순수 rename 별칭은 백킹으로).
pub fn lower_op(
    sw: &SwModel,
    op: &SwOp,
    map: &impl Fn(u32) -> u32,
) -> Result<LoweredOp, RuntimeError> {
    let dt = sw.dt_default;
    Ok(match op {
        SwOp::Conv {
            input, out, res, cin, cout, kh, kw, sh, sw: swid, pad, d, groups, act, w, b, ..
        } => {
            let (ih, iw, _) = tdesc(sw, *input);
            let conv = Conv2d {
                cin: *cin,
                cout: *cout,
                kh: *kh,
                kw: *kw,
                sh: *sh,
                sw: *swid,
                pad: *pad,
                dil: *d,
                groups: *groups,
                act: *act,
            };
            let spec: Box<dyn KernelSpec> = if *groups > 1 {
                Box::new(ConvDwSpec::from_op(&conv, ih, iw, res.is_some(), dt))
            } else if *kh == 1 {
                if *sh != 1 || pad.iter().any(|p| *p != 0) {
                    return Err(RuntimeError::Other(format!(
                        "1×1 conv stride/pad 미지원 (s={sh}, pad={pad:?})"
                    )));
                }
                Box::new(GemmPwSpec {
                    m: ih * iw,
                    kg: cin.div_ceil(4),
                    ng: cout.div_ceil(4),
                    act: *act,
                    residual: res.is_some(),
                    dt,
                })
            } else {
                Box::new(ConvIgemmSpec::from_op(&conv, ih, iw, res.is_some(), dt))
            };
            let mut bindings = vec![
                RtBinding::Tensor(map(*input)),
                RtBinding::Weights(*w),
                RtBinding::Weights(*b),
            ];
            let mut reads = vec![map(*input)];
            if let Some(r) = res {
                bindings.push(RtBinding::Tensor(map(*r)));
                reads.push(map(*r));
            }
            bindings.push(RtBinding::Tensor(map(*out)));
            LoweredOp {
                label: format!("conv {}x{} {}->{} k{}", ih, iw, cin, cout, kh),
                spec,
                bindings,
                params: [0; 16],
                reads,
                writes: vec![map(*out)],
            }
        }
        SwOp::Binary { a, b, out, op: bop, act } => {
            let (h, w, c) = tdesc(sw, *a);
            let len = (h * w * c.div_ceil(4)) as u32;
            let (operand, scalar, extra) = match b {
                SwOperand::Tensor { tid } => {
                    (EwOperand::Tensor, 0.0, Some(RtBinding::Tensor(map(*tid))))
                }
                SwOperand::Scalar { v, first } => {
                    (EwOperand::Scalar { scalar_first: *first }, *v, None)
                }
                SwOperand::Cvec { w, .. } => {
                    (EwOperand::ChannelVector, 0.0, Some(RtBinding::Weights(*w)))
                }
                SwOperand::CvecTensor { tid } => {
                    (EwOperand::ChannelVector, 0.0, Some(RtBinding::Tensor(map(*tid))))
                }
            };
            let spec = ElementwiseSpec { op: *bop, operand, act: *act, len_vec4: len, dt };
            let mut bindings = vec![RtBinding::Tensor(map(*a))];
            let mut reads = vec![map(*a)];
            if let Some(e) = extra {
                if let RtBinding::Tensor(t) = &e {
                    reads.push(*t);
                }
                bindings.push(e);
            }
            bindings.push(RtBinding::Tensor(map(*out)));
            LoweredOp {
                label: format!("ew {} {:?}", bop.tag(), operand),
                spec: Box::new(spec),
                bindings,
                params: ew_params(scalar, c.div_ceil(4), len),
                reads,
                writes: vec![map(*out)],
            }
        }
        SwOp::Act { input, out, act } => {
            let (h, w, c) = tdesc(sw, *input);
            let len = (h * w * c.div_ceil(4)) as u32;
            let spec = ElementwiseSpec {
                op: BinaryOp::Add,
                operand: EwOperand::Unary,
                act: *act,
                len_vec4: len,
                dt,
            };
            LoweredOp {
                label: format!("act {}", act.tag()),
                spec: Box::new(spec),
                bindings: vec![RtBinding::Tensor(map(*input)), RtBinding::Tensor(map(*out))],
                params: ew_params(0.0, c.div_ceil(4), len),
                reads: vec![map(*input)],
                writes: vec![map(*out)],
            }
        }
        SwOp::Gpool { input, out } => {
            let (h, w, c) = tdesc(sw, *input);
            LoweredOp {
                label: format!("gpool c{c}"),
                spec: Box::new(GpoolSpec { h, w, c, dt }),
                bindings: vec![RtBinding::Tensor(map(*input)), RtBinding::Tensor(map(*out))],
                params: [0; 16],
                reads: vec![map(*input)],
                writes: vec![map(*out)],
            }
        }
        SwOp::Avgpool { input, out, kh, kw, sh, sw: swid, pad } => {
            let (ih, iw, c) = tdesc(sw, *input);
            let spec = AvgPoolSpec {
                ih,
                iw,
                c,
                k: *kh,
                s: *sh,
                pad: *pad,
                dt,
            };
            debug_assert_eq!(kh, kw);
            debug_assert_eq!(sh, swid);
            LoweredOp {
                label: format!("avgpool k{kh}"),
                spec: Box::new(spec),
                bindings: vec![RtBinding::Tensor(map(*input)), RtBinding::Tensor(map(*out))],
                params: [0; 16],
                reads: vec![map(*input)],
                writes: vec![map(*out)],
            }
        }
        SwOp::Resize { input, out, oh, ow, mode } => {
            let (ih, iw, c) = tdesc(sw, *input);
            let spec = ResizeBilinearSpec { ih, iw, c, oh: *oh, ow: *ow, mode: *mode, dt };
            LoweredOp {
                label: format!("resize {ih}x{iw}->{oh}x{ow}"),
                spec: Box::new(spec),
                bindings: vec![RtBinding::Tensor(map(*input)), RtBinding::Tensor(map(*out))],
                params: [0; 16],
                reads: vec![map(*input)],
                writes: vec![map(*out)],
            }
        }
        SwOp::Concat { out, parts } => {
            let (h, w, c_out) = tdesc(sw, *out);
            let mut gparts = Vec::new();
            let mut bindings = Vec::new();
            let mut reads = Vec::new();
            for p in parts {
                let (_, _, in_c) = tdesc(sw, p.input);
                gparts.push(GatherPart { c: p.c, src_c: 0, in_c });
                bindings.push(RtBinding::Tensor(map(p.input)));
                reads.push(map(p.input));
            }
            bindings.push(RtBinding::Tensor(map(*out)));
            LoweredOp {
                label: format!("concat c{c_out}"),
                spec: Box::new(ChannelGatherSpec { px: h * w, c_out, parts: gparts, dt }),
                bindings,
                params: [0; 16],
                reads,
                writes: vec![map(*out)],
            }
        }
        SwOp::Chcopy { input, out, src_c, n } => {
            let (h, w, in_c) = tdesc(sw, *input);
            let spec = ChannelGatherSpec {
                px: h * w,
                c_out: *n,
                parts: vec![GatherPart { c: *n, src_c: *src_c, in_c }],
                dt,
            };
            LoweredOp {
                label: format!("chcopy {src_c}+{n}"),
                spec: Box::new(spec),
                bindings: vec![RtBinding::Tensor(map(*input)), RtBinding::Tensor(map(*out))],
                params: [0; 16],
                reads: vec![map(*input)],
                writes: vec![map(*out)],
            }
        }
        SwOp::Mix { .. } => {
            return Err(RuntimeError::Other("mix 커널 미구현 (변환기 기본 미방출)".into()))
        }
    })
}

/// 비정렬 뷰 실체화 op 합성: backing에서 view의 채널 구간을 복사
pub fn materialize_view(sw: &SwModel, view_tid: u32, backing_tid: u32, cg_off: u32) -> LoweredOp {
    let (h, w, c) = tdesc(sw, view_tid);
    let (_, _, in_c) = tdesc(sw, backing_tid);
    let spec = ChannelGatherSpec {
        px: h * w,
        c_out: c,
        parts: vec![GatherPart { c, src_c: cg_off * 4, in_c }],
        dt: sw.dt_default,
    };
    LoweredOp {
        label: format!("view {}", sw.tensors[view_tid as usize].name),
        spec: Box::new(spec),
        bindings: vec![RtBinding::Tensor(backing_tid), RtBinding::Tensor(view_tid)],
        params: [0; 16],
        reads: vec![backing_tid],
        writes: vec![view_tid],
    }
}
