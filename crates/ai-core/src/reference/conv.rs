//! conv2d CPU 레퍼런스 — groups로 일반(1)/depthwise(cin) 모두 커버.
//!
//! 입력/출력: 논리 NHWC. 가중치: OIHW `[cout][cin/groups][kh][kw]`
//! (depthwise는 cin/groups=1이라 `[c][kh][kw]`와 동일).
//! 에필로그: bias → act → residual(활성화 후 더함 — WebGL2 엔진의 conv-tail 융합 규약).

use crate::ops::Conv2d;

pub fn conv2d(
    op: &Conv2d,
    ih: u32,
    iw: u32,
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(input.len(), (ih * iw * op.cin) as usize);
    let cin_g = op.cin / op.groups;
    let cout_g = op.cout / op.groups;
    assert_eq!(weights.len(), (op.cout * cin_g * op.kh * op.kw) as usize);
    if let Some(b) = bias {
        assert_eq!(b.len(), op.cout as usize);
    }
    let (oh, ow) = op.out_hw(ih, iw);
    if let Some(r) = residual {
        assert_eq!(r.len(), (oh * ow * op.cout) as usize);
    }

    let mut out = vec![0f32; (oh * ow * op.cout) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for oc in 0..op.cout {
                let g = oc / cout_g;
                let mut acc = bias.map_or(0.0, |b| b[oc as usize]);
                for ky in 0..op.kh {
                    for kx in 0..op.kw {
                        let iy = (oy * op.sh + ky * op.dil) as i64 - op.pad[0] as i64;
                        let ix = (ox * op.sw + kx * op.dil) as i64 - op.pad[1] as i64;
                        if iy < 0 || iy >= ih as i64 || ix < 0 || ix >= iw as i64 {
                            continue; // zero padding
                        }
                        for icg in 0..cin_g {
                            let ic = g * cin_g + icg;
                            let iv = input[((iy as u32 * iw + ix as u32) * op.cin + ic) as usize];
                            let wv = weights
                                [((((oc * cin_g) + icg) * op.kh + ky) * op.kw + kx) as usize];
                            acc += iv * wv;
                        }
                    }
                }
                let idx = ((oy * ow + ox) * op.cout + oc) as usize;
                let mut v = op.act.apply(acc);
                if let Some(r) = residual {
                    v += r[idx];
                }
                out[idx] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;

    /// 손계산 오라클: 3×3 입력(1채널) ramp, k3 s1 p1, 가중치 전부 1 → 각 출력 = 이웃 합
    #[test]
    fn hand_computed_box_sum() {
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect(); // 1..9, 3x3
        let op = Conv2d {
            cin: 1,
            cout: 1,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::None,
        };
        let w = vec![1f32; 9];
        let out = conv2d(&op, 3, 3, &input, &w, None, None);
        // 손계산: 중앙 = 1+2+...+9 = 45, 좌상 = 1+2+4+5 = 12, 우하 = 5+6+8+9 = 28
        assert_eq!(out[4], 45.0);
        assert_eq!(out[0], 12.0);
        assert_eq!(out[8], 28.0);
    }

    /// 항등 1×1 conv → 입력 그대로
    #[test]
    fn pointwise_identity() {
        let input: Vec<f32> = (0..2 * 2 * 3).map(|v| v as f32 * 0.5).collect();
        let op = Conv2d::pointwise(3, 3, Activation::None);
        let mut w = vec![0f32; 9];
        for i in 0..3 {
            w[i * 3 + i] = 1.0;
        }
        assert_eq!(conv2d(&op, 2, 2, &input, &w, None, None), input);
    }

    /// 손계산 pointwise: cin=2→cout=1, w=[2,3], bias=1, relu
    #[test]
    fn hand_computed_pointwise_bias_act() {
        let input = vec![1.0, -1.0, 0.5, 2.0]; // 1x2 px, c=2
        let op = Conv2d::pointwise(2, 1, Activation::Relu);
        let w = vec![2.0, 3.0];
        let out = conv2d(&op, 1, 2, &input, &w, Some(&[1.0]), None);
        // px0: 2*1 + 3*(-1) + 1 = 0 → relu 0; px1: 2*0.5 + 3*2 + 1 = 8
        assert_eq!(out, vec![0.0, 8.0]);
    }

    /// depthwise 손계산: c=2, k3 s1 p1, 채널별 가중치 상수(1, 2)
    #[test]
    fn hand_computed_depthwise() {
        // 2×2 입력, 채널0 = [1,2,3,4], 채널1 = [10,20,30,40]
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let op = Conv2d::depthwise(2, 3, 1, Activation::None);
        let mut w = vec![0f32; 2 * 9];
        w[..9].fill(1.0); // 채널0: box sum
        w[9..].fill(2.0); // 채널1: 2 * box sum
        let out = conv2d(&op, 2, 2, &input, &w, None, None);
        // 채널0 (0,0): 1+2+3+4 = 10 (p1이라 전 픽셀 포함), 채널1 (0,0): (10+20+30+40)*2 = 200
        assert_eq!(out[0], 10.0);
        assert_eq!(out[1], 200.0);
    }

    /// residual은 활성화 후에 더해진다 (에필로그 규약)
    #[test]
    fn residual_applied_after_activation() {
        let input = vec![-1.0];
        let op = Conv2d::pointwise(1, 1, Activation::Relu);
        let out = conv2d(&op, 1, 1, &input, &[1.0], None, Some(&[5.0]));
        // relu(-1) = 0, 0 + 5 = 5 (활성화 전 더하면 relu(4) = 4가 되어 다름)
        assert_eq!(out, vec![5.0]);
    }

    /// stride 2 + 비대칭 pad 조합의 출력 크기와 값
    #[test]
    fn stride2_asymmetric_pad() {
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect(); // 4x4
        let op = Conv2d {
            cin: 1,
            cout: 1,
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
            pad: [1, 0, 0, 1], // top 1, right 1
            dil: 1,
            groups: 1,
            act: Activation::None,
        };
        let w = vec![1f32; 4];
        let (oh, ow) = op.out_hw(4, 4);
        assert_eq!((oh, ow), (2, 2));
        let out = conv2d(&op, 4, 4, &input, &w, None, None);
        // (0,0): top pad 행 제외 → 입력 (0,0),(0,1) = 1+2 = 3
        assert_eq!(out[0], 3.0);
        // (1,1): iy = 1*2+ky-1 → rows 1..2, cols 2..3 → 7+8+11+12 = 38
        assert_eq!(out[3], 38.0);
    }
}
