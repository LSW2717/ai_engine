//! 풀링 — Gpool(전역 평균)·Avgpool·Maxpool. 레퍼런스와 같은 규약
//! (avgpool 분모 = 커널 크기 고정, maxpool 패딩 = -inf 항등).

use ai_core::ops::{AvgPool2d, MaxPool2d};

use crate::simd::F32x4;
use crate::view::View;

/// GlobalAveragePool: 뷰 → `[c]`
pub fn global_avg(input: View, px: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), input.c);
    out.fill(0.0);
    for p in 0..px {
        let b = input.base(p);
        for ch in 0..input.c {
            out[ch] += input.data[b + ch];
        }
    }
    let inv = 1.0 / px as f32;
    for v in out {
        *v *= inv;
    }
}

pub fn avg_pool(op: &AvgPool2d, ih: u32, iw: u32, input: View, out: &mut [f32]) {
    let (oh, ow) = op.out_hw(ih, iw);
    let c = input.c;
    let inv = 1.0 / (op.kh * op.kw) as f32;
    debug_assert_eq!(out.len(), (oh * ow) as usize * c);
    for oy in 0..oh as i64 {
        for ox in 0..ow as i64 {
            let ob = (oy * ow as i64 + ox) as usize * c;
            out[ob..ob + c].fill(0.0);
            for ky in 0..op.kh as i64 {
                for kx in 0..op.kw as i64 {
                    let iy = oy * op.sh as i64 + ky - op.pad[0] as i64;
                    let ix = ox * op.sw as i64 + kx - op.pad[1] as i64;
                    if iy < 0 || iy >= ih as i64 || ix < 0 || ix >= iw as i64 {
                        continue;
                    }
                    let b = input.base((iy * iw as i64 + ix) as usize);
                    for ch in 0..c {
                        out[ob + ch] += input.data[b + ch];
                    }
                }
            }
            for ch in 0..c {
                out[ob + ch] *= inv;
            }
        }
    }
}

/// MaxPool — 채널 4배수 구간 벡터, 꼬리 스칼라. pad_c 채널은 0
/// (BlazeFace "MaxPool→Pad(C)→Add"의 Pad를 접은 것 — ops/pool.rs 참조).
pub fn max_pool(op: &MaxPool2d, ih: u32, iw: u32, input: View, out: &mut [f32]) {
    let (oh, ow) = op.out_hw(ih, iw);
    let c = input.c;
    let oc = c + op.pad_c as usize;
    let cv = c / 4 * 4;
    debug_assert_eq!(out.len(), (oh * ow) as usize * oc);
    for oy in 0..oh as i64 {
        for ox in 0..ow as i64 {
            let ob = (oy * ow as i64 + ox) as usize * oc;
            let mut cc = 0usize;
            while cc < cv {
                let mut m = F32x4::splat(f32::NEG_INFINITY);
                for ky in 0..op.kh as i64 {
                    let iy = oy * op.sh as i64 + ky - op.pad[0] as i64;
                    if iy < 0 || iy >= ih as i64 {
                        continue;
                    }
                    for kx in 0..op.kw as i64 {
                        let ix = ox * op.sw as i64 + kx - op.pad[1] as i64;
                        if ix < 0 || ix >= iw as i64 {
                            continue;
                        }
                        let b = input.base((iy * iw as i64 + ix) as usize);
                        m = m.max(F32x4::load(input.data, b + cc));
                    }
                }
                m.store(out, ob + cc);
                cc += 4;
            }
            for ch in cv..c {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..op.kh as i64 {
                    let iy = oy * op.sh as i64 + ky - op.pad[0] as i64;
                    if iy < 0 || iy >= ih as i64 {
                        continue;
                    }
                    for kx in 0..op.kw as i64 {
                        let ix = ox * op.sw as i64 + kx - op.pad[1] as i64;
                        if ix < 0 || ix >= iw as i64 {
                            continue;
                        }
                        let b = input.base((iy * iw as i64 + ix) as usize);
                        m = m.max(input.data[b + ch]);
                    }
                }
                out[ob + ch] = m;
            }
            out[ob + c..ob + oc].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::reference;
    use ai_core::rng::XorShift32;

    #[test]
    fn matches_reference() {
        let mut rng = XorShift32::new(1);
        let (h, w, c) = (5u32, 6u32, 7u32);
        let x = rng.vec_f32((h * w * c) as usize);

        let want = reference::pool::global_avg_pool(&x, h, w, c);
        let mut got = vec![0f32; c as usize];
        global_avg(View::dense(&x, c as usize), (h * w) as usize, &mut got);
        for (a, g) in want.iter().zip(&got) {
            assert!((a - g).abs() < 1e-5);
        }

        let op = AvgPool2d { kh: 2, kw: 2, sh: 2, sw: 2, pad: [0; 4] };
        let want = reference::pool::avg_pool(&op, h, w, c, &x);
        let mut got = vec![0f32; want.len()];
        avg_pool(&op, h, w, View::dense(&x, c as usize), &mut got);
        assert_eq!(want, got);
    }

    #[test]
    fn maxpool_matches_reference() {
        let mut rng = XorShift32::new(2);
        // c=7 → 벡터 4 + 꼬리 3, pad_c=2, 홀수 입력(우하단 경계)
        let (h, w, c) = (5u32, 7u32, 7u32);
        let x = rng.vec_f32((h * w * c) as usize);
        for (op_pad, pad_c) in [([0u32; 4], 0u32), ([0; 4], 2), ([1, 1, 0, 0], 3)] {
            let op = MaxPool2d { kh: 2, kw: 2, sh: 2, sw: 2, pad: op_pad, pad_c };
            let want = reference::pool::max_pool(&op, h, w, c, &x);
            let mut got = vec![0f32; want.len()];
            max_pool(&op, h, w, View::dense(&x, c as usize), &mut got);
            assert_eq!(want, got);
        }
    }
}
