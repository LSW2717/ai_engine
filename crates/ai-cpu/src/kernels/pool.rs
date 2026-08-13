//! 풀링 — Gpool(전역 평균)·Avgpool. 레퍼런스와 같은 규약
//! (avgpool 분모 = 커널 크기 고정).

use ai_core::ops::AvgPool2d;

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
}
