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

use crate::simd::F32x4;
use crate::view::View;

/// 출력채널 블록 폭 (F32x4 두 개)
pub const NR: usize = 8;

/// concat 융합 파트: 뷰 + 가중치 K축에서의 시작 채널
#[derive(Clone, Copy)]
pub struct ConvPart<'a> {
    pub view: View<'a>,
    /// 이 파트의 첫 채널이 대응하는 가중치 입력채널 인덱스 (누적)
    pub ic0: usize,
}

/// OIHW → `[tap][cin][cout_pad]` 재패킹. 반환 (data, cout_pad).
pub fn repack_weights(w_oihw: &[f32], cout: u32, cin: u32, kh: u32, kw: u32) -> (Vec<f32>, usize) {
    let (cout, cin, kh, kw) = (cout as usize, cin as usize, kh as usize, kw as usize);
    assert_eq!(w_oihw.len(), cout * cin * kh * kw);
    let cout_pad = cout.next_multiple_of(NR);
    let taps = kh * kw;
    let mut out = vec![0f32; taps * cin * cout_pad];
    for oc in 0..cout {
        for ic in 0..cin {
            for t in 0..taps {
                out[(t * cin + ic) * cout_pad + oc] = w_oihw[((oc * cin) + ic) * taps + t];
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
    let cin_total: usize = parts.iter().map(|p| p.view.c).sum();
    debug_assert_eq!(cin_total, op.cin as usize);
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

        for oc0 in (0..cout).step_by(NR) {
            let mut ox = 0usize;
            while ox < ow as usize {
                let interior =
                    ox as i64 >= lo && (ox + 3) as i64 <= hi && ox + 4 <= ow as usize;
                // 에필로그: bias는 acc 초기값 → act → +residual (conv-tail 규약)
                let sink = |px: usize, l: usize, acc: f32, out: &mut [f32]| {
                    let mut v = op.act.apply(acc);
                    if let Some(r) = residual {
                        v += r.data[r.base(px) + oc0 + l];
                    }
                    out[(px - px_base) * cout + oc0 + l] = v;
                };
                if interior {
                    mr4(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox,
                        |px, l, acc| sink(px, l, acc, out),
                        cout - oc0,
                    );
                    ox += 4;
                } else {
                    mr1(
                        op, iw, ow as usize, parts, wts, bias, cin_total, cout_pad, oc0,
                        ky_list, oy, ox,
                        |px, l, acc| sink(px, l, acc, out),
                        cout - oc0,
                    );
                    ox += 1;
                }
            }
        }
    }
}

/// interior 4픽셀 × NR 마이크로커널 — 경계 검사 없음
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mr4(
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
    mut sink: impl FnMut(usize, usize, f32),
    nc: usize,
) {
    let b0 = F32x4::load(bias, oc0);
    let b1 = F32x4::load(bias, oc0 + 4);
    let mut acc = [[b0, b1]; 4];

    let (sw, dil, pl) = (op.sw as usize, op.dil as usize, op.pad[1] as usize);
    let kw = op.kw as usize;
    for &(t_row, iy) in ky_list {
        for kx in 0..kw {
            let t = t_row + kx;
            // interior 보장: ix ≥ 0
            let ix0 = ox * sw + kx * dil - pl;
            let row = iy * iw as usize;
            for part in parts {
                let base = [
                    part.view.base(row + ix0),
                    part.view.base(row + ix0 + sw),
                    part.view.base(row + ix0 + 2 * sw),
                    part.view.base(row + ix0 + 3 * sw),
                ];
                let x = part.view.data;
                let mut w_idx = (t * cin_total + part.ic0) * cout_pad + oc0;
                for ic in 0..part.view.c {
                    let w0 = F32x4::load(wts, w_idx);
                    let w1 = F32x4::load(wts, w_idx + 4);
                    for p in 0..4 {
                        let a = F32x4::splat(x[base[p] + ic]);
                        acc[p][0] = acc[p][0].fma(a, w0);
                        acc[p][1] = acc[p][1].fma(a, w1);
                    }
                    w_idx += cout_pad;
                }
            }
        }
    }

    for p in 0..4 {
        let px = oy * ow + ox + p;
        let lanes = [acc[p][0].to_array(), acc[p][1].to_array()];
        for l in 0..nc.min(NR) {
            sink(px, l, lanes[l / 4][l % 4]);
        }
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
    mut sink: impl FnMut(usize, usize, f32),
    nc: usize,
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
                let mut w_idx = (t * cin_total + part.ic0) * cout_pad + oc0;
                for ic in 0..part.view.c {
                    let a = F32x4::splat(x[base + ic]);
                    acc[0] = acc[0].fma(a, F32x4::load(wts, w_idx));
                    acc[1] = acc[1].fma(a, F32x4::load(wts, w_idx + 4));
                    w_idx += cout_pad;
                }
            }
        }
    }

    let px = oy * ow + ox;
    let lanes = [acc[0].to_array(), acc[1].to_array()];
    for l in 0..nc.min(NR) {
        sink(px, l, lanes[l / 4][l % 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;
    use ai_core::{reference, Activation};

    fn run_case(op: &Conv2d, ih: u32, iw: u32, with_res: bool, seed: u32) {
        let mut rng = XorShift32::new(seed);
        let x = rng.vec_f32((ih * iw * op.cin) as usize);
        let w = rng.vec_f32((op.cout * op.cin * op.kh * op.kw) as usize);
        let b = rng.vec_f32(op.cout as usize);
        let (oh, ow) = op.out_hw(ih, iw);
        let res = if with_res {
            Some(rng.vec_f32((oh * ow * op.cout) as usize))
        } else {
            None
        };

        let want = reference::conv::conv2d(op, ih, iw, &x, &w, Some(&b), res.as_deref());

        let (wts, cout_pad) = repack_weights(&w, op.cout, op.cin, op.kh, op.kw);
        let bias = pad_bias(&b, cout_pad);
        let parts = [ConvPart { view: View::dense(&x, op.cin as usize), ic0: 0 }];
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

        let (wts, cout_pad) = repack_weights(&w, cout, cin, 3, 3);
        let bias = pad_bias(&b, cout_pad);
        let parts = [
            ConvPart { view: View::dense(&x1, c1), ic0: 0 },
            ConvPart {
                view: View { data: &big, c_off, stride: big_c, c: c2 },
                ic0: c1,
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
        let x = rng.vec_f32((ih * iw * op.cin) as usize);
        let w = rng.vec_f32((op.cout * op.cin * 9) as usize);
        let b = rng.vec_f32(op.cout as usize);
        let (wts, cout_pad) = repack_weights(&w, op.cout, op.cin, 3, 3);
        let bias = pad_bias(&b, cout_pad);
        let parts = [ConvPart { view: View::dense(&x, op.cin as usize), ic0: 0 }];
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
