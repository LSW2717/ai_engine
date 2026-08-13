//! 일반/pointwise conv — tap-루프 브로드캐스트 GEMM.
//!
//! 전략: 출력 픽셀 4개 × 출력채널 8개(NR) 블록을 레지스터에 상주시키고,
//! (tap, cin)을 순회하며 A(입력)는 스칼라 브로드캐스트, B(가중치)는 연속
//! 벡터 로드로 fma 한다. A가 브로드캐스트라 stride·concat 파트·비정렬
//! 채널 수에 무관 — im2col이 필요 없다.
//!
//! 가중치 레이아웃(이 파일이 계약 소유): `[tap][cin][cout_pad]`, cout_pad = 8배수.
//! bias: `[cout_pad]`. 에필로그 순서는 레퍼런스와 동일: bias → act → +residual.
//!
//! 경계 처리: 행(oy)별 유효 tap-행 목록 + 모든 kx가 유효한 interior ox 구간을
//! 미리 계산 → interior는 MR4(무검사), 가장자리는 MR1(tap별 검사)로 처리.

use ai_core::ops::Conv2d;

use crate::kernels::apply4;
use crate::simd::F32x4;
use crate::view::View;

/// 출력채널 블록 폭 (F32x4 두 개)
pub const NR: usize = 8;

/// 큰 마이크로커널의 픽셀 폭. 네이티브 8 (acc 16개 + A 8 + W 2 = 26 < 32 NEON 레지스터).
/// wasm 6 — V8 arm64 가용 vreg ~28에서 MR8은 스필 경계라 실험으로 정한다.
#[cfg(target_arch = "wasm32")]
const MR_BIG: usize = 6;
#[cfg(not(target_arch = "wasm32"))]
const MR_BIG: usize = 8;

/// concat 융합 파트: 뷰 + 가중치 K축에서의 시작 채널
#[derive(Clone, Copy)]
pub struct ConvPart<'a> {
    pub view: View<'a>,
    /// 이 파트의 첫 채널이 대응하는 가중치 입력채널 인덱스 (누적, 패딩 포함)
    pub ic0: usize,
    /// 벡터 루프 경계. 보통 view.c (c%4는 스칼라 꼬리). K축이 제로패딩된
    /// 파트(스템 cin=3 등)는 next4(c) — 4레인 로드가 이웃 채널을 읽어도
    /// 가중치가 0이라 무해하다 (슬롯 +4 패딩이 마지막 픽셀을 보호).
    pub c4: usize,
}

/// OIHW → `[tap][cin_pad][cout_pad]` 재패킹. 반환 (data, cout_pad).
/// cin_pad > cin이면 K축 제로패딩 (cin=3 스템을 벡터 경로에 태우는 용도).
pub fn repack_weights(
    w_oihw: &[f32],
    cout: u32,
    cin: u32,
    cin_pad: usize,
    kh: u32,
    kw: u32,
) -> (Vec<f32>, usize) {
    let (cout, cin, kh, kw) = (cout as usize, cin as usize, kh as usize, kw as usize);
    assert_eq!(w_oihw.len(), cout * cin * kh * kw);
    debug_assert!(cin_pad >= cin);
    let cout_pad = cout.next_multiple_of(NR);
    let taps = kh * kw;
    let mut out = vec![0f32; taps * cin_pad * cout_pad];
    for oc in 0..cout {
        for ic in 0..cin {
            for t in 0..taps {
                out[(t * cin_pad + ic) * cout_pad + oc] = w_oihw[((oc * cin) + ic) * taps + t];
            }
        }
    }
    (out, cout_pad)
}

/// bias `[cout]` → `[cout_pad]` (0패딩)
pub fn pad_bias(bias: &[f32], cout_pad: usize) -> Vec<f32> {
    let mut out = vec![0f32; cout_pad];
    out[..bias.len()].copy_from_slice(bias);
    out
}

/// 출력 행 구간 [y0, y1)을 계산한다 (스레드 분할 단위).
///
/// `wts`/`bias`는 repack_weights/pad_bias 레이아웃. `out`은 **이 밴드만큼의**
/// 버퍼(행 y0부터, 길이 ≥ (y1-y0)*ow*cout) — 밴드별 서로소 슬라이스라
/// 스레드가 안전하게 나눠 쓴다. residual은 전체 버퍼(절대 픽셀 인덱스, 읽기 전용).
#[allow(clippy::too_many_arguments)]
pub fn conv_std(
    op: &Conv2d,
    ih: u32,
    iw: u32,
    parts: &[ConvPart],
    wts: &[f32],
    bias: &[f32],
    residual: Option<View>,
    out: &mut [f32],
    y0: u32,
    y1: u32,
) {
    debug_assert_eq!(op.groups, 1, "dw는 kernels::dw 사용");
    let (oh, ow) = op.out_hw(ih, iw);
    debug_assert!(y1 <= oh);
    let _ = oh;
    let cout = op.cout as usize;
    let cout_pad = bias.len();
    // 가중치 K축 스트라이드 = c4 합 (패딩 파트는 c4 > c)
    let cin_total: usize = parts.iter().map(|p| p.c4).sum();
    debug_assert!(cin_total >= op.cin as usize);
    debug_assert_eq!(wts.len(), (op.kh * op.kw) as usize * cin_total * cout_pad);

    let (sh, sw, dil) = (op.sh as i64, op.sw as i64, op.dil as i64);
    let (pt, pl) = (op.pad[0] as i64, op.pad[1] as i64);
    let (kw_i, iw_i, ih_i) = (op.kw as i64, iw as i64, ih as i64);

    // 모든 kx에 대해 ix가 유효한 interior ox 구간 [lo, hi] (inclusive)
    let mut lo: i64 = 0;
    let mut hi: i64 = ow as i64 - 1;
    for kx in 0..kw_i {
        let off = kx * dil - pl; // ix = ox*sw + off
        // ox*sw + off >= 0  →  ox >= ceil(-off / sw)
        let mn = (-off + sw - 1).div_euclid(sw).max(0);
        // ox*sw + off <= iw-1  →  ox <= floor((iw-1-off) / sw)
        let mx = (iw_i - 1 - off).div_euclid(sw);
        lo = lo.max(mn);
        hi = hi.min(mx);
    }

    // 밴드-로컬 쓰기 기준 픽셀 (out은 y0행부터 시작하는 슬라이스)
    let px_base = y0 as usize * ow as usize;
    debug_assert!(out.len() >= (y1 - y0) as usize * ow as usize * cout);

    for oy in y0 as usize..y1 as usize {
        // 유효 tap 행: (tap 행 시작 인덱스, iy)
        let mut ky_list: [(usize, usize); 16] = [(0, 0); 16];
        let mut n_ky = 0usize;
        for ky in 0..op.kh as i64 {
            let iy = oy as i64 * sh + ky * dil - pt;
            if iy >= 0 && iy < ih_i {
                ky_list[n_ky] = ((ky * kw_i) as usize, iy as usize);
                n_ky += 1;
            }
        }
        let ky_list = &ky_list[..n_ky];

        let mut oc0 = 0usize;
        while oc0 < cout {
            // wasm: cout 16배수 구간은 NR16 — 셔플 1개가 madd 4개를 서빙
            // (NR8은 2개) → 브로드캐스트의 wasm 셔플세를 반으로.
            let nr16 = cfg!(target_arch = "wasm32") && cout - oc0 >= 16;
            let mut ox = 0usize;
            while ox < ow as usize {
                let inb = |n: usize| {
                    ox as i64 >= lo && (ox + n - 1) as i64 <= hi && ox + n <= ow as usize
                };
                if nr16 && inb(4) {
                    mr4_nr16(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox, residual, out, px_base, cout,
                    );
                    ox += 4;
                } else if !nr16 && inb(MR_BIG) {
                    // MR_BIG 우선: 누산 체인 2×MR개가 fma 지연을 채운다 (4픽셀
                    // 8체인으로는 절반 — 디코더 conv가 46 GF/s에서 멈추던 원인).
                    mr_n::<MR_BIG>(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox, residual, out, px_base, cout,
                    );
                    ox += MR_BIG;
                } else if !nr16 && inb(4) {
                    mr_n::<4>(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox, residual, out, px_base, cout,
                    );
                    ox += 4;
                } else {
                    mr1(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox, residual, out, px_base, cout,
                    );
                    if nr16 {
                        mr1(
                            op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad,
                            oc0 + NR, ky_list, oy, ox, residual, out, px_base, cout,
                        );
                    }
                    ox += 1;
                }
            }
            oc0 += if nr16 { 16 } else { NR };
        }
    }
}

/// 에필로그: act → +residual → 저장. 풀 블록(nc≥NR)은 벡터, 부분 블록은 스칼라.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn epilogue(
    op: &Conv2d,
    acc: [F32x4; 2],
    px: usize,
    px_base: usize,
    cout: usize,
    oc0: usize,
    residual: Option<View>,
    out: &mut [f32],
) {
    let o = (px - px_base) * cout + oc0;
    let nc = cout - oc0;
    if nc >= NR {
        let mut v0 = apply4(op.act, acc[0]);
        let mut v1 = apply4(op.act, acc[1]);
        if let Some(r) = residual {
            let rb = r.base(px) + oc0;
            v0 = v0.add(F32x4::load(r.data, rb));
            v1 = v1.add(F32x4::load(r.data, rb + 4));
        }
        v0.store(out, o);
        v1.store(out, o + 4);
    } else {
        // 부분 블록: 앞 4레인은 벡터, 나머지 스칼라
        let mut l0 = 0usize;
        if nc >= 4 {
            let mut v0 = apply4(op.act, acc[0]);
            if let Some(r) = residual {
                v0 = v0.add(F32x4::load(r.data, r.base(px) + oc0));
            }
            v0.store(out, o);
            l0 = 4;
        }
        let lanes = [acc[0].to_array(), acc[1].to_array()];
        for l in l0..nc {
            let mut v = op.act.apply(lanes[l / 4][l % 4]);
            if let Some(r) = residual {
                v += r.data[r.base(px) + oc0 + l];
            }
            out[o + l] = v;
        }
    }
}

/// interior MR픽셀 × NR 마이크로커널 — 경계 검사 없음. MR은 4/8만 쓴다
/// (8 = 누산 체인 16개로 fma 지연 은닉, 4 = 잔여 구간).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mr_n<const MR: usize>(
    op: &Conv2d,
    iw: u32,
    ow: usize,
    parts: &[ConvPart],
    wts: &[f32],
    bias: &[f32],
    cin_total: usize,
    cout_pad: usize,
    oc0: usize,
    ky_list: &[(usize, usize)],
    oy: usize,
    ox: usize,
    residual: Option<View>,
    out: &mut [f32],
    px_base: usize,
    cout: usize,
) {
    let b0 = F32x4::load(bias, oc0);
    let b1 = F32x4::load(bias, oc0 + 4);
    let mut acc = [[b0, b1]; MR];

    let (sw, dil, pl) = (op.sw as usize, op.dil as usize, op.pad[1] as usize);
    let kw = op.kw as usize;
    for &(t_row, iy) in ky_list {
        for kx in 0..kw {
            let t = t_row + kx;
            // interior 보장: ix ≥ 0
            let ix0 = ox * sw + kx * dil - pl;
            let row = iy * iw as usize;
            for part in parts {
                let base: [usize; MR] =
                    std::array::from_fn(|p| part.view.base(row + ix0 + p * sw));
                let x = part.view.data;
                let pc = part.view.c;
                let mut w_idx = (t * cin_total + part.ic0) * cout_pad + oc0;
                let mut ic = 0usize;
                // ic 4블록: 픽셀당 A 벡터 로드 1 + lane-fma 4 — splat 로드 4개를
                // 로드 1개로 (XNNPACK gemm-splat 구조. NHWC라 채널 4개가 연속).
                // 경계는 c4가 보장: 보통 4배수 하한, 패딩 파트는 가중치 0.
                while ic + 4 <= part.c4 {
                    let a: [F32x4; MR] = std::array::from_fn(|p| F32x4::load(x, base[p] + ic));
                    macro_rules! lane {
                        ($l:literal) => {{
                            let w0 = F32x4::load(wts, w_idx);
                            let w1 = F32x4::load(wts, w_idx + 4);
                            for p in 0..MR {
                                acc[p][0] = acc[p][0].fma_lane::<$l>(a[p], w0);
                                acc[p][1] = acc[p][1].fma_lane::<$l>(a[p], w1);
                            }
                            w_idx += cout_pad;
                        }};
                    }
                    lane!(0);
                    lane!(1);
                    lane!(2);
                    lane!(3);
                    ic += 4;
                }
                // 꼬리 채널: splat 로드
                while ic < pc {
                    let w0 = F32x4::load(wts, w_idx);
                    let w1 = F32x4::load(wts, w_idx + 4);
                    for p in 0..MR {
                        let a = F32x4::load_splat(x, base[p] + ic);
                        acc[p][0] = acc[p][0].fma(a, w0);
                        acc[p][1] = acc[p][1].fma(a, w1);
                    }
                    w_idx += cout_pad;
                    ic += 1;
                }
            }
        }
    }

    for p in 0..MR {
        let px = oy * ow + ox + p;
        epilogue(op, acc[p], px, px_base, cout, oc0, residual, out);
    }
}

/// interior 4픽셀 × 16출력채널 (wasm 전용 경로) — 셔플 브로드캐스트 1개가
/// madd 4개를 서빙해 fma당 셔플이 NR8의 절반이다. 같은 (a, lane) 셔플은
/// LLVM CSE가 합친다 (fma_lane 반복 호출로 표현).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mr4_nr16(
    op: &Conv2d,
    iw: u32,
    ow: usize,
    parts: &[ConvPart],
    wts: &[f32],
    bias: &[f32],
    cin_total: usize,
    cout_pad: usize,
    oc0: usize,
    ky_list: &[(usize, usize)],
    oy: usize,
    ox: usize,
    residual: Option<View>,
    out: &mut [f32],
    px_base: usize,
    cout: usize,
) {
    let b: [F32x4; 4] = std::array::from_fn(|i| F32x4::load(bias, oc0 + 4 * i));
    let mut acc = [b; 4];

    let (sw, dil, pl) = (op.sw as usize, op.dil as usize, op.pad[1] as usize);
    let kw = op.kw as usize;
    for &(t_row, iy) in ky_list {
        for kx in 0..kw {
            let t = t_row + kx;
            let ix0 = ox * sw + kx * dil - pl;
            let row = iy * iw as usize;
            for part in parts {
                let base: [usize; 4] =
                    std::array::from_fn(|p| part.view.base(row + ix0 + p * sw));
                let x = part.view.data;
                let pc = part.view.c;
                let mut w_idx = (t * cin_total + part.ic0) * cout_pad + oc0;
                let mut ic = 0usize;
                while ic + 4 <= part.c4 {
                    let a: [F32x4; 4] = std::array::from_fn(|p| F32x4::load(x, base[p] + ic));
                    macro_rules! lane {
                        ($l:literal) => {{
                            let w: [F32x4; 4] =
                                std::array::from_fn(|i| F32x4::load(wts, w_idx + 4 * i));
                            for p in 0..4 {
                                for i in 0..4 {
                                    acc[p][i] = acc[p][i].fma_lane::<$l>(a[p], w[i]);
                                }
                            }
                            w_idx += cout_pad;
                        }};
                    }
                    lane!(0);
                    lane!(1);
                    lane!(2);
                    lane!(3);
                    ic += 4;
                }
                while ic < pc {
                    let w: [F32x4; 4] =
                        std::array::from_fn(|i| F32x4::load(wts, w_idx + 4 * i));
                    for p in 0..4 {
                        let a = F32x4::load_splat(x, base[p] + ic);
                        for i in 0..4 {
                            acc[p][i] = acc[p][i].fma(a, w[i]);
                        }
                    }
                    w_idx += cout_pad;
                    ic += 1;
                }
            }
        }
    }

    for p in 0..4 {
        let px = oy * ow + ox + p;
        epilogue(op, [acc[p][0], acc[p][1]], px, px_base, cout, oc0, residual, out);
        epilogue(op, [acc[p][2], acc[p][3]], px, px_base, cout, oc0 + NR, residual, out);
    }
}

/// 가장자리 1픽셀 × NR — kx별 경계 검사
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mr1(
    op: &Conv2d,
    iw: u32,
    ow: usize,
    parts: &[ConvPart],
    wts: &[f32],
    bias: &[f32],
    cin_total: usize,
    cout_pad: usize,
    oc0: usize,
    ky_list: &[(usize, usize)],
    oy: usize,
    ox: usize,
    residual: Option<View>,
    out: &mut [f32],
    px_base: usize,
    cout: usize,
) {
    let mut acc = [F32x4::load(bias, oc0), F32x4::load(bias, oc0 + 4)];

    let (sw, dil, pl) = (op.sw as i64, op.dil as i64, op.pad[1] as i64);
    let kw = op.kw as usize;
    for &(t_row, iy) in ky_list {
        for kx in 0..kw {
            let ix = ox as i64 * sw + kx as i64 * dil - pl;
            if ix < 0 || ix >= iw as i64 {
                continue;
            }
            let t = t_row + kx;
            let lin = iy * iw as usize + ix as usize;
            for part in parts {
                let base = part.view.base(lin);
                let x = part.view.data;
                let pc = part.view.c;
                let mut w_idx = (t * cin_total + part.ic0) * cout_pad + oc0;
                let mut ic = 0usize;
                while ic + 4 <= part.c4 {
                    let a = F32x4::load(x, base + ic);
                    macro_rules! lane {
                        ($l:literal) => {{
                            acc[0] = acc[0].fma_lane::<$l>(a, F32x4::load(wts, w_idx));
                            acc[1] = acc[1].fma_lane::<$l>(a, F32x4::load(wts, w_idx + 4));
                            w_idx += cout_pad;
                        }};
                    }
                    lane!(0);
                    lane!(1);
                    lane!(2);
                    lane!(3);
                    ic += 4;
                }
                while ic < pc {
                    let a = F32x4::load_splat(x, base + ic);
                    acc[0] = acc[0].fma(a, F32x4::load(wts, w_idx));
                    acc[1] = acc[1].fma(a, F32x4::load(wts, w_idx + 4));
                    w_idx += cout_pad;
                    ic += 1;
                }
            }
        }
    }

    epilogue(op, acc, oy * ow + ox, px_base, cout, oc0, residual, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;
    use ai_core::{reference, Activation};

    fn run_case(op: &Conv2d, ih: u32, iw: u32, with_res: bool, seed: u32) {
        let mut rng = XorShift32::new(seed);
        let mut x = rng.vec_f32((ih * iw * op.cin) as usize);
        let w = rng.vec_f32((op.cout * op.cin * op.kh * op.kw) as usize);
        let b = rng.vec_f32(op.cout as usize);
        let (oh, ow) = op.out_hw(ih, iw);
        let res = if with_res {
            Some(rng.vec_f32((oh * ow * op.cout) as usize))
        } else {
            None
        };

        let want = reference::conv::conv2d(op, ih, iw, &x, &w, Some(&b), res.as_deref());

        // 단일 파트는 plan과 동일하게 K 제로패딩 (cin 3/5 케이스가 패딩 경로 검증).
        // 입력 +4 패딩은 exec 슬롯 패딩과 같은 역할.
        x.extend_from_slice(&[0.0; 4]);
        let cin_pad = (op.cin as usize).next_multiple_of(4);
        let (wts, cout_pad) = repack_weights(&w, op.cout, op.cin, cin_pad, op.kh, op.kw);
        let bias = pad_bias(&b, cout_pad);
        let parts =
            [ConvPart { view: View::dense(&x, op.cin as usize), ic0: 0, c4: cin_pad }];
        let mut got = vec![0f32; want.len()];
        let res_view = res.as_deref().map(|r| View::dense(r, op.cout as usize));
        conv_std(op, ih, iw, &parts, &wts, &bias, res_view, &mut got, 0, oh);

        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                "불일치 @{i}: want {a}, got {b} (op {op:?})"
            );
        }
    }

    #[test]
    fn pointwise_matches_reference() {
        // cout 5 (< NR, 패딩 경로), cin 3 (비4배수)
        run_case(&Conv2d::pointwise(3, 5, Activation::Relu), 6, 7, false, 1);
        // cout 16 (NR 2블록), residual
        run_case(&Conv2d::pointwise(8, 16, Activation::Hardswish), 5, 9, true, 2);
    }

    #[test]
    fn k3_stride_pad_matches_reference() {
        let op = Conv2d {
            cin: 6,
            cout: 10,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::None,
        };
        run_case(&op, 8, 8, false, 3);

        let s2 = Conv2d { sh: 2, sw: 2, act: Activation::Relu, ..op };
        run_case(&s2, 9, 11, true, 4);
    }

    #[test]
    fn asymmetric_pad_and_dilation() {
        let op = Conv2d {
            cin: 4,
            cout: 8,
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
            pad: [1, 0, 0, 1],
            dil: 1,
            groups: 1,
            act: Activation::None,
        };
        run_case(&op, 6, 6, false, 5);

        let d2 = Conv2d {
            cin: 3,
            cout: 5,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            pad: [2; 4],
            dil: 2,
            groups: 1,
            act: Activation::Sigmoid,
        };
        run_case(&d2, 7, 7, false, 6);
    }

    /// concat 융합: 두 파트(스트라이드 뷰)를 단일 conv 입력으로 — 레퍼런스는
    /// 실체화된 concat과 비교
    #[test]
    fn concat_parts_match_materialized() {
        let mut rng = XorShift32::new(7);
        let (ih, iw) = (5u32, 6u32);
        let (c1, c2) = (3usize, 5usize);
        let cin = (c1 + c2) as u32;
        let cout = 12u32;
        // 백킹 텐서 두 개 — 두 번째는 큰 텐서의 채널 슬라이스(alias 시뮬레이션)
        let x1 = rng.vec_f32((ih * iw) as usize * c1);
        let big_c = 8usize;
        let big = rng.vec_f32((ih * iw) as usize * big_c);
        let c_off = 2usize; // big의 채널 2..7을 뷰로

        // 실체화된 concat 입력
        let px = (ih * iw) as usize;
        let mut xcat = vec![0f32; px * (c1 + c2)];
        for p in 0..px {
            xcat[p * cin as usize..p * cin as usize + c1]
                .copy_from_slice(&x1[p * c1..(p + 1) * c1]);
            xcat[p * cin as usize + c1..(p + 1) * cin as usize]
                .copy_from_slice(&big[p * big_c + c_off..p * big_c + c_off + c2]);
        }

        let op = Conv2d {
            cin,
            cout,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
        };
        let w = rng.vec_f32((cout * cin * 9) as usize);
        let b = rng.vec_f32(cout as usize);
        let want = reference::conv::conv2d(&op, ih, iw, &xcat, &w, Some(&b), None);

        let (wts, cout_pad) = repack_weights(&w, cout, cin, cin as usize, 3, 3);
        let bias = pad_bias(&b, cout_pad);
        let parts = [
            ConvPart { view: View::dense(&x1, c1), ic0: 0, c4: c1 },
            ConvPart {
                view: View { data: &big, c_off, stride: big_c, c: c2 },
                ic0: c1,
                c4: c2,
            },
        ];
        let mut got = vec![0f32; want.len()];
        let (oh, _) = op.out_hw(ih, iw);
        conv_std(&op, ih, iw, &parts, &wts, &bias, None, &mut got, 0, oh);

        for (i, (a, g)) in want.iter().zip(&got).enumerate() {
            assert!((a - g).abs() <= 1e-4 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
        }
    }

    /// 행 분할 실행(y0..y1 두 구간)이 전체 실행과 동일해야 한다 (스레딩 전제)
    #[test]
    fn row_split_equals_full() {
        let op = Conv2d {
            cin: 5,
            cout: 9,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
        };
        let (ih, iw) = (11u32, 13u32);
        let mut rng = XorShift32::new(8);
        let mut x = rng.vec_f32((ih * iw * op.cin) as usize);
        x.extend_from_slice(&[0.0; 4]); // 4레인 오버리드 패딩
        let w = rng.vec_f32((op.cout * op.cin * 9) as usize);
        let b = rng.vec_f32(op.cout as usize);
        let cin_pad = (op.cin as usize).next_multiple_of(4);
        let (wts, cout_pad) = repack_weights(&w, op.cout, op.cin, cin_pad, 3, 3);
        let bias = pad_bias(&b, cout_pad);
        let parts =
            [ConvPart { view: View::dense(&x, op.cin as usize), ic0: 0, c4: cin_pad }];
        let (oh, ow) = op.out_hw(ih, iw);

        let mut full = vec![0f32; (oh * ow * op.cout) as usize];
        conv_std(&op, ih, iw, &parts, &wts, &bias, None, &mut full, 0, oh);
        // 밴드 슬라이스 규약: 각 밴드는 자기 행부터 시작하는 서로소 구간에 쓴다
        let mut split = vec![0f32; full.len()];
        let mid = oh / 2;
        let cut = (mid * ow * op.cout) as usize;
        let (band0, band1) = split.split_at_mut(cut);
        conv_std(&op, ih, iw, &parts, &wts, &bias, None, band0, 0, mid);
        conv_std(&op, ih, iw, &parts, &wts, &bias, None, band1, mid, oh);
        assert_eq!(full, split);
    }
}
