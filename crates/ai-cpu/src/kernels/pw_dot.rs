//! cout이 아주 작은 pointwise conv — 채널축 내적(dot) 커널.
//!
//! NR8 브로드캐스트 GEMM은 cout=2(세그 헤드)에서 레인 6개를 버린다
//! (wasm 스텝벤치에서 하한의 5배). 여기는 픽셀당 출력채널별로
//! acc += x[4ch] * w[oc][4ch] 를 벡터로 돌리고 마지막에 수평합 —
//! 유효 연산만 한다.
//!
//! 가중치 레이아웃(이 파일 소유): `[cout][k_pad4]` 행-major (0패딩).
//! bias `[cout]`. cout ≤ MAX_COUT(4)만 — 그 이상은 conv_std가 낫다.

use ai_core::ops::Conv2d;

use crate::simd::F32x4;
use crate::view::View;

pub const MAX_COUT: usize = 4;

/// `[cout][cin]` (OIHW의 1x1) → `[cout][k_pad4]` 0패딩
pub fn repack_weights(w: &[f32], cout: u32, cin: u32) -> (Vec<f32>, usize) {
    let (cout, cin) = (cout as usize, cin as usize);
    assert_eq!(w.len(), cout * cin);
    let k_pad = cin.next_multiple_of(4);
    let mut out = vec![0f32; cout * k_pad];
    for oc in 0..cout {
        out[oc * k_pad..oc * k_pad + cin].copy_from_slice(&w[oc * cin..(oc + 1) * cin]);
    }
    (out, k_pad)
}

/// 1x1, 단일 파트, cout ≤ MAX_COUT. 뷰의 4레인 로드가 픽셀 채널을 넘을 수
/// 있으므로 +4 패딩 백킹 전제 (exec 슬롯 패딩) — 초과 레인은 가중치 0.
pub fn conv_pw_dot(
    op: &Conv2d,
    input: View,
    wts: &[f32],
    k_pad: usize,
    bias: &[f32],
    px: usize,
    out: &mut [f32],
) {
    let cout = op.cout as usize;
    debug_assert!(cout <= MAX_COUT);
    debug_assert_eq!(wts.len(), cout * k_pad);
    debug_assert!(out.len() >= px * cout);

    for p in 0..px {
        let b = input.base(p);
        let mut acc = [F32x4::splat(0.0); MAX_COUT];
        let mut kc = 0usize;
        while kc < k_pad {
            let xv = F32x4::load(input.data, b + kc);
            for oc in 0..cout {
                acc[oc] = acc[oc].fma(xv, F32x4::load(wts, oc * k_pad + kc));
            }
            kc += 4;
        }
        for oc in 0..cout {
            out[p * cout + oc] = op.act.apply(acc[oc].sum() + bias[oc]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;
    use ai_core::{reference, Activation};

    #[test]
    fn matches_reference() {
        for (cin, cout, act) in
            [(24u32, 2u32, Activation::None), (7, 3, Activation::Relu), (16, 4, Activation::Sigmoid)]
        {
            let op = Conv2d::pointwise(cin, cout, act);
            let (ih, iw) = (5u32, 6u32);
            let mut rng = XorShift32::new(cin + cout);
            let mut x = rng.vec_f32((ih * iw * cin) as usize);
            let w = rng.vec_f32((cout * cin) as usize);
            let b = rng.vec_f32(cout as usize);
            let want = reference::conv::conv2d(&op, ih, iw, &x, &w, Some(&b), None);

            x.extend_from_slice(&[0.0; 4]); // exec 슬롯 패딩과 동일
            let (wts, k_pad) = repack_weights(&w, cout, cin);
            let mut got = vec![0f32; want.len()];
            conv_pw_dot(
                &op,
                View::dense(&x, cin as usize),
                &wts,
                k_pad,
                &b,
                (ih * iw) as usize,
                &mut got,
            );
            for (i, (a, g)) in want.iter().zip(&got).enumerate() {
                assert!((a - g).abs() <= 1e-4 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
            }
        }
    }
}
