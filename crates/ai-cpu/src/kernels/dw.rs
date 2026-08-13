//! depthwise conv — NHWC에서 채널이 픽셀 내 연속이므로 채널 축 벡터화.
//!
//! 가중치 레이아웃(이 파일이 계약 소유): `[tap][c_pad4]` (0패딩), bias `[c_pad4]`.
//! 픽셀당: acc[c] = bias[c]; 유효 tap마다 acc += w[tap][c] * x[in_px][c].
//! 에필로그: act → +residual (conv-tail 규약 동일).
//!
//! 경계 처리(conv_std와 같은 3분할): 행별 유효 ky 목록 + 모든 kx가 유효한
//! interior ox 구간을 미리 계산 → interior는 tap 경계검사 없는 고속 픽셀,
//! 좌우 가장자리만 kx별 검사. 예산표에서 tap당 i64 경계검사가 5 GF/s
//! 바닥의 주범이었다 (하한의 9.6배).
//!
//! 채널 꼬리(c%4)는 스칼라 — 뷰(c_off≠0)의 마지막 4레인 로드가 백킹 텐서의
//! 다음 픽셀 채널을 읽으면 안 되기 때문에 벡터 로드는 c/4*4까지만 한다.

use ai_core::ops::Conv2d;

use crate::kernels::apply4;
use crate::simd::F32x4;
use crate::view::View;

/// `[c][kh][kw]` → `[tap][c_pad4]` 재패킹. 반환 (data, c_pad4).
pub fn repack_weights(w: &[f32], c: u32, kh: u32, kw: u32) -> (Vec<f32>, usize) {
    let (c, kh, kw) = (c as usize, kh as usize, kw as usize);
    assert_eq!(w.len(), c * kh * kw);
    let c_pad = c.next_multiple_of(4);
    let taps = kh * kw;
    let mut out = vec![0f32; taps * c_pad];
    for ch in 0..c {
        for t in 0..taps {
            out[t * c_pad + ch] = w[ch * taps + t];
        }
    }
    (out, c_pad)
}

/// 출력 행 구간 [y0, y1) 계산. `input`은 단일 뷰 (dw에는 concat 융합이 안 온다 —
/// 오면 plan이 먼저 실체화한다).
/// `out`은 밴드 슬라이스(행 y0부터), residual은 전체 버퍼 — conv_std와 같은 규약.
#[allow(clippy::too_many_arguments)]
pub fn conv_dw(
    op: &Conv2d,
    ih: u32,
    iw: u32,
    input: View,
    wts: &[f32],
    bias: &[f32],
    residual: Option<View>,
    out: &mut [f32],
    y0: u32,
    y1: u32,
) {
    debug_assert_eq!(op.groups, op.cin, "일반 conv는 kernels::conv 사용");
    debug_assert_eq!(op.cin, op.cout);
    let c = op.cout as usize;
    let c_pad = bias.len();
    debug_assert_eq!(input.c, c);
    debug_assert_eq!(wts.len(), (op.kh * op.kw) as usize * c_pad);

    let (_, ow) = op.out_hw(ih, iw);
    let ow = ow as usize;
    let (sh, sw, dil) = (op.sh as i64, op.sw as i64, op.dil as i64);
    let (pt, pl) = (op.pad[0] as i64, op.pad[1] as i64);
    let cv = c / 4 * 4; // 벡터화 가능한 채널 수
    let kw = op.kw as usize;

    // 모든 kx에 대해 ix가 유효한 interior ox 구간 [lo, hi) (conv_std와 같은 식)
    let mut lo: i64 = 0;
    let mut hi: i64 = ow as i64 - 1;
    for kx in 0..kw as i64 {
        let off = kx * dil - pl;
        lo = lo.max((-off + sw - 1).div_euclid(sw).max(0));
        hi = hi.min((iw as i64 - 1 - off).div_euclid(sw));
    }
    let x_lo = lo.clamp(0, ow as i64) as usize;
    let x_hi = (hi + 1).clamp(x_lo as i64, ow as i64) as usize; // exclusive

    let px_base = y0 as usize * ow;
    debug_assert!(out.len() >= (y1 - y0) as usize * ow * c);

    for oy in y0 as usize..y1 as usize {
        // 유효 tap 행: (tap 행 시작 인덱스, iy)
        let mut ky_list: [(usize, usize); 16] = [(0, 0); 16];
        let mut n_ky = 0usize;
        for ky in 0..op.kh as i64 {
            let iy = oy as i64 * sh + ky * dil - pt;
            if iy >= 0 && iy < ih as i64 {
                ky_list[n_ky] = ((ky * kw as i64) as usize, iy as usize);
                n_ky += 1;
            }
        }
        let ky_list = &ky_list[..n_ky];

        // 좌 가장자리 (kx별 검사)
        for ox in 0..x_lo {
            edge_px(op, iw, input, wts, bias, residual, out, ky_list, oy, ox, ow, px_base, c, c_pad, cv);
        }
        // interior — tap 경계검사 없음. 채널블록 바깥 루프: (행, 블록)마다
        // tap 오프셋·가중치 테이블을 한 번 만들고, 픽셀 루프는 순수
        // load-fma-덧셈만 남긴다 (주소 곱셈이 4.3배 배율의 잔여 주범이었다).
        const MAX_TAPS: usize = 32;
        if x_hi > x_lo && (op.kh * op.kw) as usize <= MAX_TAPS {
            let step = sw as usize * input.stride;
            for cc in (0..cv).step_by(4) {
                let mut off0 = [0usize; MAX_TAPS];
                let mut wv = [F32x4::splat(0.0); MAX_TAPS];
                let mut nt = 0usize;
                for &(t_row, iy) in ky_list {
                    for kx in 0..kw {
                        // interior 보장: x_lo*sw + kx*dil - pl ≥ 0
                        let ix = (x_lo as i64 * sw + kx as i64 * dil - pl) as usize;
                        off0[nt] = input.base(iy * iw as usize + ix) + cc;
                        wv[nt] = F32x4::load(wts, (t_row + kx) * c_pad + cc);
                        nt += 1;
                    }
                }
                let b = F32x4::load(bias, cc);
                let mut out_idx = (oy * ow + x_lo - px_base) * c + cc;
                let mut r_idx = residual.map(|r| r.base(oy * ow + x_lo) + cc);
                let mut px_off = 0usize;
                // 4픽셀 동시: 누산 체인 4개(fma 지연 은닉) + tap당 가중치 로드 1회
                let mut ox = x_lo;
                while ox + 4 <= x_hi {
                    let mut acc = [b; 4];
                    for t in 0..nt {
                        let w = wv[t];
                        let o = off0[t] + px_off;
                        acc[0] = acc[0].fma(F32x4::load(input.data, o), w);
                        acc[1] = acc[1].fma(F32x4::load(input.data, o + step), w);
                        acc[2] = acc[2].fma(F32x4::load(input.data, o + 2 * step), w);
                        acc[3] = acc[3].fma(F32x4::load(input.data, o + 3 * step), w);
                    }
                    for a in acc {
                        let mut v = apply4(op.act, a);
                        if let Some(r) = residual {
                            let ri = r_idx.as_mut().unwrap();
                            v = v.add(F32x4::load(r.data, *ri));
                            *ri += r.stride;
                        }
                        v.store(out, out_idx);
                        out_idx += c;
                    }
                    px_off += 4 * step;
                    ox += 4;
                }
                for _ox in ox..x_hi {
                    let mut acc = b;
                    for t in 0..nt {
                        acc = acc.fma(F32x4::load(input.data, off0[t] + px_off), wv[t]);
                    }
                    let mut v = apply4(op.act, acc);
                    if let Some(r) = residual {
                        let ri = r_idx.as_mut().unwrap();
                        v = v.add(F32x4::load(r.data, *ri));
                        *ri += r.stride;
                    }
                    v.store(out, out_idx);
                    out_idx += c;
                    px_off += step;
                }
            }
            // 꼬리 채널: 스칼라 (같은 테이블 방식)
            for ch in cv..c {
                let mut off0 = [0usize; MAX_TAPS];
                let mut ws = [0f32; MAX_TAPS];
                let mut nt = 0usize;
                for &(t_row, iy) in ky_list {
                    for kx in 0..kw {
                        let ix = (x_lo as i64 * sw + kx as i64 * dil - pl) as usize;
                        off0[nt] = input.base(iy * iw as usize + ix) + ch;
                        ws[nt] = wts[(t_row + kx) * c_pad + ch];
                        nt += 1;
                    }
                }
                let mut out_idx = (oy * ow + x_lo - px_base) * c + ch;
                let mut r_idx = residual.map(|r| r.base(oy * ow + x_lo) + ch);
                let mut px_off = 0usize;
                for _ox in x_lo..x_hi {
                    let mut acc = bias[ch];
                    for t in 0..nt {
                        acc += input.data[off0[t] + px_off] * ws[t];
                    }
                    let mut v = op.act.apply(acc);
                    if let Some(r) = residual {
                        let ri = r_idx.as_mut().unwrap();
                        v += r.data[*ri];
                        *ri += r.stride;
                    }
                    out[out_idx] = v;
                    out_idx += c;
                    px_off += step;
                }
            }
        } else {
            // 초대형 커널(k>5 상당) — 검사 경로로 폴백
            for ox in x_lo..x_hi {
                edge_px(op, iw, input, wts, bias, residual, out, ky_list, oy, ox, ow, px_base, c, c_pad, cv);
            }
        }
        // 우 가장자리
        for ox in x_hi..ow {
            edge_px(op, iw, input, wts, bias, residual, out, ky_list, oy, ox, ow, px_base, c, c_pad, cv);
        }
    }
}

/// 가장자리 1픽셀 — kx별 경계 검사 (interior 밖에서만 호출)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn edge_px(
    op: &Conv2d,
    iw: u32,
    input: View,
    wts: &[f32],
    bias: &[f32],
    residual: Option<View>,
    out: &mut [f32],
    ky_list: &[(usize, usize)],
    oy: usize,
    ox: usize,
    ow: usize,
    px_base: usize,
    c: usize,
    c_pad: usize,
    cv: usize,
) {
    let (sw, dil, pl) = (op.sw as i64, op.dil as i64, op.pad[1] as i64);
    let kw = op.kw as usize;
    let px_out = oy * ow + ox;
    let out_base = (px_out - px_base) * c;

    let mut cc = 0usize;
    while cc < cv {
        let mut acc = F32x4::load(bias, cc);
        for &(t_row, iy) in ky_list {
            for kx in 0..kw {
                let ix = ox as i64 * sw + kx as i64 * dil - pl;
                if ix < 0 || ix >= iw as i64 {
                    continue;
                }
                let base = input.base(iy * iw as usize + ix as usize);
                acc = acc.fma(
                    F32x4::load(input.data, base + cc),
                    F32x4::load(wts, (t_row + kx) * c_pad + cc),
                );
            }
        }
        let mut v = apply4(op.act, acc);
        if let Some(r) = residual {
            v = v.add(F32x4::load(r.data, r.base(px_out) + cc));
        }
        v.store(out, out_base + cc);
        cc += 4;
    }
    for ch in cv..c {
        let mut acc = bias[ch];
        for &(t_row, iy) in ky_list {
            for kx in 0..kw {
                let ix = ox as i64 * sw + kx as i64 * dil - pl;
                if ix < 0 || ix >= iw as i64 {
                    continue;
                }
                let base = input.base(iy * iw as usize + ix as usize);
                acc += input.data[base + ch] * wts[(t_row + kx) * c_pad + ch];
            }
        }
        let mut v = op.act.apply(acc);
        if let Some(r) = residual {
            v += r.data[r.base(px_out) + ch];
        }
        out[out_base + ch] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;
    use ai_core::{reference, Activation};

    fn run_case(c: u32, k: u32, s: u32, pad: u32, act: Activation, ih: u32, iw: u32, seed: u32) {
        let op = Conv2d::depthwise(c, k, s, act);
        let op = Conv2d { pad: [pad; 4], ..op };
        let mut rng = XorShift32::new(seed);
        let x = rng.vec_f32((ih * iw * c) as usize);
        let w = rng.vec_f32((c * k * k) as usize);
        let b = rng.vec_f32(c as usize);
        let (oh, ow) = op.out_hw(ih, iw);
        let res = rng.vec_f32((oh * ow * c) as usize);

        let want = reference::conv::conv2d(&op, ih, iw, &x, &w, Some(&b), Some(&res));

        let (wts, c_pad) = repack_weights(&w, c, k, k);
        let bias = crate::kernels::conv::pad_bias(&b, c_pad);
        let mut got = vec![0f32; want.len()];
        conv_dw(
            &op, ih, iw,
            View::dense(&x, c as usize),
            &wts, &bias, Some(View::dense(&res, c as usize)), &mut got, 0, oh,
        );
        for (i, (a, g)) in want.iter().zip(&got).enumerate() {
            assert!((a - g).abs() <= 1e-4 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
        }
    }

    #[test]
    fn dw_matches_reference() {
        run_case(8, 3, 1, 1, Activation::Relu, 7, 9, 1); // 4배수 채널
        run_case(7, 3, 2, 1, Activation::Hardswish, 8, 8, 2); // 꼬리 채널 + s2
        run_case(3, 5, 1, 2, Activation::None, 9, 6, 3); // c<4 전부 꼬리, k5
    }

    /// 뷰 입력(백킹 채널 슬라이스)에서도 레퍼런스와 일치 — 꼬리가 백킹의
    /// 이웃 채널을 침범하지 않는지가 관건
    #[test]
    fn strided_view_input() {
        let (ih, iw) = (6u32, 5u32);
        let big_c = 10usize;
        let c = 6u32; // 뷰 채널 6 (c_off 3) → cv=4, 꼬리 2
        let c_off = 3usize;
        let mut rng = XorShift32::new(4);
        let big = rng.vec_f32((ih * iw) as usize * big_c);
        let w = rng.vec_f32((c * 9) as usize);
        let b = rng.vec_f32(c as usize);

        // 실체화한 입력으로 레퍼런스 계산
        let px = (ih * iw) as usize;
        let mut xm = vec![0f32; px * c as usize];
        for p in 0..px {
            xm[p * c as usize..(p + 1) * c as usize]
                .copy_from_slice(&big[p * big_c + c_off..p * big_c + c_off + c as usize]);
        }
        let op = Conv2d::depthwise(c, 3, 1, Activation::Relu);
        let op = Conv2d { pad: [1; 4], ..op };
        let want = reference::conv::conv2d(&op, ih, iw, &xm, &w, Some(&b), None);

        let (wts, c_pad) = repack_weights(&w, c, 3, 3);
        let bias = crate::kernels::conv::pad_bias(&b, c_pad);
        let mut got = vec![0f32; want.len()];
        let (oh, _) = op.out_hw(ih, iw);
        conv_dw(
            &op, ih, iw,
            View { data: &big, c_off, stride: big_c, c: c as usize },
            &wts, &bias, None, &mut got, 0, oh,
        );
        for (i, (a, g)) in want.iter().zip(&got).enumerate() {
            assert!((a - g).abs() <= 1e-4 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
        }
    }
}
