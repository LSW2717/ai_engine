//! 풀링 CPU 레퍼런스.

use crate::ops::{AvgPool2d, MaxPool2d};

/// GlobalAveragePool: `[h][w][c]` → `[c]`
pub fn global_avg_pool(input: &[f32], h: u32, w: u32, c: u32) -> Vec<f32> {
    assert_eq!(input.len(), (h * w * c) as usize);
    let mut out = vec![0f32; c as usize];
    for i in 0..(h * w) as usize {
        for ch in 0..c as usize {
            out[ch] += input[i * c as usize + ch];
        }
    }
    let inv = 1.0 / (h * w) as f32;
    for v in &mut out {
        *v *= inv;
    }
    out
}

/// AveragePool (분모 = 커널 크기 고정; RVM은 k==s, pad=0 경로만 사용)
pub fn avg_pool(op: &AvgPool2d, ih: u32, iw: u32, c: u32, input: &[f32]) -> Vec<f32> {
    assert_eq!(input.len(), (ih * iw * c) as usize);
    let (oh, ow) = op.out_hw(ih, iw);
    let inv = 1.0 / (op.kh * op.kw) as f32;
    let mut out = vec![0f32; (oh * ow * c) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for ch in 0..c {
                let mut acc = 0f32;
                for ky in 0..op.kh {
                    for kx in 0..op.kw {
                        let iy = (oy * op.sh + ky) as i64 - op.pad[0] as i64;
                        let ix = (ox * op.sw + kx) as i64 - op.pad[1] as i64;
                        if iy >= 0 && iy < ih as i64 && ix >= 0 && ix < iw as i64 {
                            acc += input[((iy as u32 * iw + ix as u32) * c + ch) as usize];
                        }
                    }
                }
                out[((oy * ow + ox) * c + ch) as usize] = acc * inv;
            }
        }
    }
    out
}

/// MaxPool (+pad_c 채널 제로패딩) — 패딩 영역은 max에서 제외
pub fn max_pool(op: &MaxPool2d, ih: u32, iw: u32, c: u32, input: &[f32]) -> Vec<f32> {
    assert_eq!(input.len(), (ih * iw * c) as usize);
    let (oh, ow) = op.out_hw(ih, iw);
    let oc = c + op.pad_c;
    let mut out = vec![0f32; (oh * ow * oc) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for ch in 0..c {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..op.kh {
                    for kx in 0..op.kw {
                        let iy = (oy * op.sh + ky) as i64 - op.pad[0] as i64;
                        let ix = (ox * op.sw + kx) as i64 - op.pad[1] as i64;
                        if iy >= 0 && iy < ih as i64 && ix >= 0 && ix < iw as i64 {
                            m = m.max(input[((iy as u32 * iw + ix as u32) * c + ch) as usize]);
                        }
                    }
                }
                out[((oy * ow + ox) * oc + ch) as usize] = m;
            }
            // pad_c 채널은 0으로 남는다 (vec! 초기값)
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool_k2s2_hand_computed() {
        // 4×4 ramp 1..16, k2 s2, c=1 → 각 2×2 블록 최대 + pad_c 1
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let op = MaxPool2d { kh: 2, kw: 2, sh: 2, sw: 2, pad: [0; 4], pad_c: 1 };
        let got = max_pool(&op, 4, 4, 1, &input);
        assert_eq!(got, vec![6.0, 0.0, 8.0, 0.0, 14.0, 0.0, 16.0, 0.0]);
    }

    #[test]
    fn gpool_hand_computed() {
        // 2×2, c=2: ch0 = [1,2,3,4] → 2.5, ch1 = [4,8,12,16] → 10
        let input = vec![1.0, 4.0, 2.0, 8.0, 3.0, 12.0, 4.0, 16.0];
        assert_eq!(global_avg_pool(&input, 2, 2, 2), vec![2.5, 10.0]);
    }

    #[test]
    fn avgpool_k2s2_hand_computed() {
        // 4×4 ramp 1..16, k2 s2 → 각 2×2 블록 평균
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let op = AvgPool2d { kh: 2, kw: 2, sh: 2, sw: 2, pad: [0; 4] };
        let out = avg_pool(&op, 4, 4, 1, &input);
        assert_eq!(out, vec![3.5, 5.5, 11.5, 13.5]);
    }
}
