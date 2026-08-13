//! 소채널(cin≤4) k>1 conv의 im2row — 패치 행렬 `[out_px][k_pad]`를 실체화해
//! conv를 순수 1x1 GEMM으로 바꾼다 (스템 3x3 s2 cin3이 표적).
//!
//! 왜: 브로드캐스트 GEMM은 tap마다 주소 계산·part 루프를 다시 도는데 cin=3은
//! tap당 일이 4레인 하나뿐이라 오버헤드가 지배한다 (wasm 스텝벤치 1.03ms,
//! 네이티브 대비 1.74배). 패치로 펴면 K=27 연속 — mr_n 벡터 경로가 그대로 탄다.
//!
//! K 순서는 (ky, kx, ic) — NHWC 입력에서 커널행이 **연속 kw*c 플로트**라
//! interior 복사가 벡터 3개다. 전제: dense 뷰(c_off=0, stride=c), dil=1.
//!
//! 오버런 규약: 복사는 12플로트 벡터로 하고 초과분(≤3)은 같은 픽셀의 다음
//! ky 구간 또는 다음 픽셀 시작을 침범한다 — 픽셀을 오름차순으로, 픽셀 안에서
//! ky를 오름차순으로 쓰므로 나중 쓰기가 덮는다. 마지막 픽셀의 초과분만
//! 스크래치 +4 패딩에 떨어진다. k_pad 여분 레인은 가중치 0이 소거.

use ai_core::ops::Conv2d;

use crate::simd::F32x4;
use crate::view::View;

/// input(dense, c≤4) → out `[oh*ow][k_pad]` 패치. out.len() ≥ oh*ow*k_pad+4.
pub fn im2row(op: &Conv2d, ih: u32, iw: u32, input: View, k_pad: usize, out: &mut [f32]) {
    debug_assert_eq!(input.c_off, 0);
    debug_assert_eq!(input.stride, input.c);
    debug_assert_eq!(op.dil, 1);
    let (oh, ow) = op.out_hw(ih, iw);
    let (oh, ow) = (oh as usize, ow as usize);
    let c = input.c;
    let (kh, kw) = (op.kh as usize, op.kw as usize);
    let (sh, sw) = (op.sh as usize, op.sw as usize);
    let (pt, pl) = (op.pad[0] as i64, op.pad[1] as i64);
    let row_k = kw * c;
    debug_assert!(k_pad >= kh * row_k);
    debug_assert!(out.len() >= oh * ow * k_pad + 4);
    let d = input.data;

    for oy in 0..oh {
        // 이 출력행의 ky별 입력행 (없으면 usize::MAX)
        let mut iy_of = [usize::MAX; 16];
        for (ky, slot) in iy_of.iter_mut().enumerate().take(kh) {
            let iy = (oy * sh + ky) as i64 - pt;
            if iy >= 0 && iy < ih as i64 {
                *slot = iy as usize;
            }
        }
        for ox in 0..ow {
            let ix0 = (ox * sw) as i64 - pl;
            let dst_px = (oy * ow + ox) * k_pad;
            let interior_x = ix0 >= 0 && ix0 + kw as i64 <= iw as i64;
            for ky in 0..kh {
                let dst = dst_px + ky * row_k;
                let iy = iy_of[ky];
                if iy == usize::MAX {
                    // 패딩 행: 0 (12플로트 단위, 초과는 뒤 쓰기가 덮는다)
                    let z = F32x4::splat(0.0);
                    let mut o = 0usize;
                    while o < row_k {
                        z.store(out, dst + o);
                        o += 4;
                    }
                    continue;
                }
                if interior_x {
                    let src = (iy * iw as usize + ix0 as usize) * c;
                    let mut o = 0usize;
                    while o < row_k {
                        F32x4::load(d, src + o).store(out, dst + o);
                        o += 4;
                    }
                } else {
                    // x 에지: kx별 검사 (행당 몇 픽셀뿐)
                    for kx in 0..kw {
                        let ix = ix0 + kx as i64;
                        for ch in 0..c {
                            out[dst + kx * c + ch] = if ix >= 0 && ix < iw as i64 {
                                d[(iy * iw as usize + ix as usize) * c + ch]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
    }
}

/// OIHW → (ky,kx,ic) K-major 1행 가중치 `[cout][kh*kw*c]` (im2row 패치 순서와 일치)
pub fn permute_weights(w_oihw: &[f32], cout: u32, cin: u32, kh: u32, kw: u32) -> Vec<f32> {
    let (cout, cin, kh, kw) = (cout as usize, cin as usize, kh as usize, kw as usize);
    assert_eq!(w_oihw.len(), cout * cin * kh * kw);
    let k = kh * kw * cin;
    let mut out = vec![0f32; cout * k];
    for oc in 0..cout {
        for ic in 0..cin {
            for t in 0..kh * kw {
                out[oc * k + t * cin + ic] = w_oihw[(oc * cin + ic) * kh * kw + t];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv;
    use ai_core::rng::XorShift32;
    use ai_core::{reference, Activation};

    /// im2row + 1x1 GEMM == 레퍼런스 conv (스템 기하 그대로)
    #[test]
    fn stem_via_im2row_matches_reference() {
        let op = Conv2d {
            cin: 3,
            cout: 8,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
        };
        let (ih, iw) = (10u32, 12u32);
        let mut rng = XorShift32::new(9);
        let mut x = rng.vec_f32((ih * iw * op.cin) as usize);
        let w = rng.vec_f32((op.cout * op.cin * 9) as usize);
        let b = rng.vec_f32(op.cout as usize);
        let want = reference::conv::conv2d(&op, ih, iw, &x, &w, Some(&b), None);
        x.extend_from_slice(&[0.0; 4]);

        let (oh, ow) = op.out_hw(ih, iw);
        let k = (op.kh * op.kw * op.cin) as usize;
        let k_pad = k.next_multiple_of(4);
        let mut patches = vec![0f32; (oh * ow) as usize * k_pad + 4];
        im2row(&op, ih, iw, View::dense(&x, op.cin as usize), k_pad, &mut patches);

        let w_perm = permute_weights(&w, op.cout, op.cin, op.kh, op.kw);
        let (wts, cout_pad) = conv::repack_weights(&w_perm, op.cout, k as u32, k_pad, 1, 1);
        let bias = conv::pad_bias(&b, cout_pad);
        let pw = Conv2d::pointwise(k as u32, op.cout, op.act);
        let parts = [conv::ConvPart {
            view: View { data: &patches, c_off: 0, stride: k_pad, c: k_pad },
            ic0: 0,
            c4: k_pad,
        }];
        let mut got = vec![0f32; want.len()];
        conv::conv_std(&pw, oh, ow, &parts, &wts, &bias, None, &mut got, 0, oh);
        for (i, (a, g)) in want.iter().zip(&got).enumerate() {
            assert!((a - g).abs() <= 1e-4 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
        }
    }
}
