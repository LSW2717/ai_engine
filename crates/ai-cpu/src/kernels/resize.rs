//! bilinear 리사이즈 — concat 융합(parts) 지원.
//!
//! 리사이즈는 채널 독립이라 파트별로 출력 채널 구간에 직접 쓴다 —
//! concat을 실체화할 필요가 없다. 좌표 규약은 레퍼런스와 동일.
//!
//! ox 좌표(x0/x1/tx)는 행 불변이라 호출당 한 번만 계산한다 — 픽셀마다
//! floor/clamp를 다시 하던 게 c2 리사이즈를 하한의 26배로 만들던 주범.
//! 테이블 Vec은 프레임당 리사이즈 수(수 개) × 수 KB라 dispatch의 parts
//! Vec과 같은 급으로 허용.

use ai_core::ops::{CoordMode, ResizeBilinear};

use crate::simd::F32x4;
use crate::view::View;

pub fn resize_bilinear(
    op: &ResizeBilinear,
    ih: u32,
    iw: u32,
    parts: &[View],
    out: &mut [f32],
) {
    let (oh, ow) = (op.oh as usize, op.ow as usize);
    let c_total: usize = parts.iter().map(|p| p.c).sum();
    debug_assert_eq!(out.len(), oh * ow * c_total);
    let sy = ih as f32 / op.oh as f32;
    let sx = iw as f32 / op.ow as f32;
    let src = |dst: usize, scale: f32| -> f32 {
        match op.mode {
            CoordMode::HalfPixel => (dst as f32 + 0.5) * scale - 0.5,
            CoordMode::Asymmetric => dst as f32 * scale,
        }
    };

    // ox 테이블 (행 불변): (x0, x1, tx)
    let xt: Vec<(usize, usize, f32)> = (0..ow)
        .map(|ox| {
            let fx = src(ox, sx);
            let x0 = (fx.floor() as i64).clamp(0, iw as i64 - 1) as usize;
            let x1 = (fx.floor() as i64 + 1).clamp(0, iw as i64 - 1) as usize;
            (x0, x1, (fx - fx.floor()).clamp(0.0, 1.0))
        })
        .collect();

    // c=2 단일 파트 (세그 마스크 최종 업샘플): 출력 픽셀 2개를 한 벡터로 패킹.
    // 4레인 로드가 픽셀 채널(2개)을 넘어 읽으므로 +4 패딩된 백킹 전제
    // (exec의 슬롯 패딩. 단독 사용 시 호출자가 보장할 것).
    if parts.len() == 1 && parts[0].c == 2 {
        let part = &parts[0];
        let d = part.data;
        #[inline(always)]
        fn lerp2(
            d: &[f32],
            b00: usize,
            b01: usize,
            b10: usize,
            b11: usize,
            tx: f32,
            vty: F32x4,
            vty1: F32x4,
        ) -> F32x4 {
            let (vtx, vtx1) = (F32x4::splat(tx), F32x4::splat(1.0 - tx));
            let top = F32x4::load(d, b00).mul(vtx1).fma(F32x4::load(d, b01), vtx);
            let bot = F32x4::load(d, b10).mul(vtx1).fma(F32x4::load(d, b11), vtx);
            top.mul(vty1).fma(bot, vty)
        }
        for oy in 0..oh {
            let fy = src(oy, sy);
            let y0 = (fy.floor() as i64).clamp(0, ih as i64 - 1) as usize;
            let y1 = (fy.floor() as i64 + 1).clamp(0, ih as i64 - 1) as usize;
            let ty = (fy - fy.floor()).clamp(0.0, 1.0);
            let (vty, vty1) = (F32x4::splat(ty), F32x4::splat(1.0 - ty));
            let (row0, row1) = (y0 * iw as usize, y1 * iw as usize);
            let mut ob = oy * ow * 2;
            let mut ox = 0usize;
            while ox + 2 <= ow {
                let (x0a, x1a, txa) = xt[ox];
                let (x0b, x1b, txb) = xt[ox + 1];
                let va = lerp2(
                    d,
                    part.base(row0 + x0a), part.base(row0 + x1a),
                    part.base(row1 + x0a), part.base(row1 + x1a),
                    txa, vty, vty1,
                );
                let vb = lerp2(
                    d,
                    part.base(row0 + x0b), part.base(row0 + x1b),
                    part.base(row1 + x0b), part.base(row1 + x1b),
                    txb, vty, vty1,
                );
                va.low2_concat(vb).store(out, ob);
                ob += 4;
                ox += 2;
            }
            if ox < ow {
                let (x0, x1, tx) = xt[ox];
                let v = lerp2(
                    d,
                    part.base(row0 + x0), part.base(row0 + x1),
                    part.base(row1 + x0), part.base(row1 + x1),
                    tx, vty, vty1,
                )
                .to_array();
                out[ob] = v[0];
                out[ob + 1] = v[1];
            }
        }
        return;
    }

    for oy in 0..oh {
        let fy = src(oy, sy);
        let y0 = (fy.floor() as i64).clamp(0, ih as i64 - 1) as usize;
        let y1 = (fy.floor() as i64 + 1).clamp(0, ih as i64 - 1) as usize;
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);
        let (vty, vty1) = (F32x4::splat(ty), F32x4::splat(1.0 - ty));
        let (row0, row1) = (y0 * iw as usize, y1 * iw as usize);

        let mut ob = oy * ow * c_total;
        for &(x0, x1, tx) in &xt {
            let (vtx, vtx1) = (F32x4::splat(tx), F32x4::splat(1.0 - tx));
            let mut c_off_out = 0usize;
            for part in parts {
                let d = part.data;
                let (b00, b01) = (part.base(row0 + x0), part.base(row0 + x1));
                let (b10, b11) = (part.base(row1 + x0), part.base(row1 + x1));
                let cv = part.c / 4 * 4;
                let mut cc = 0usize;
                while cc < cv {
                    let top = F32x4::load(d, b00 + cc).mul(vtx1).fma(F32x4::load(d, b01 + cc), vtx);
                    let bot = F32x4::load(d, b10 + cc).mul(vtx1).fma(F32x4::load(d, b11 + cc), vtx);
                    top.mul(vty1).fma(bot, vty).store(out, ob + c_off_out + cc);
                    cc += 4;
                }
                for ch in cv..part.c {
                    let top = d[b00 + ch] * (1.0 - tx) + d[b01 + ch] * tx;
                    let bot = d[b10 + ch] * (1.0 - tx) + d[b11 + ch] * tx;
                    out[ob + c_off_out + ch] = top * (1.0 - ty) + bot * ty;
                }
                c_off_out += part.c;
            }
            ob += c_total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::reference;
    use ai_core::rng::XorShift32;

    #[test]
    fn single_part_matches_reference() {
        let mut rng = XorShift32::new(1);
        let (ih, iw, c) = (5u32, 7u32, 3u32);
        let x = rng.vec_f32((ih * iw * c) as usize);
        for mode in [CoordMode::HalfPixel, CoordMode::Asymmetric] {
            let op = ResizeBilinear { oh: 10, ow: 14, mode };
            let want = reference::resize::resize_bilinear(&op, ih, iw, c, &x);
            let mut got = vec![0f32; want.len()];
            resize_bilinear(&op, ih, iw, &[View::dense(&x, c as usize)], &mut got);
            assert_eq!(want, got);
        }
    }

    /// c=2 페어 패킹 고속경로 — 레퍼런스와 일치 (+4 패딩 백킹 전제)
    #[test]
    fn c2_fast_path_matches_reference() {
        let mut rng = XorShift32::new(3);
        let (ih, iw, c) = (5u32, 7u32, 2u32);
        let mut x = rng.vec_f32((ih * iw * c) as usize);
        let want_src = x.clone();
        x.extend_from_slice(&[0.0; 4]); // exec 슬롯 패딩과 동일
        for (oh, ow) in [(10u32, 14u32), (9, 13)] {
            // 짝/홀 너비 둘 다
            let op = ResizeBilinear { oh, ow, mode: CoordMode::HalfPixel };
            let want = reference::resize::resize_bilinear(&op, ih, iw, c, &want_src);
            let mut got = vec![0f32; want.len()];
            resize_bilinear(&op, ih, iw, &[View::dense(&x, c as usize)], &mut got);
            // fma는 중간 반올림이 없어 마지막 비트가 다를 수 있다 — 근사 비교
            for (i, (a, g)) in want.iter().zip(&got).enumerate() {
                assert!((a - g).abs() <= 1e-6 * a.abs().max(1.0), "불일치 @{i}: {a} vs {g}");
            }
        }
    }

    /// 두 파트 = 실체화된 concat을 리사이즈한 것과 동일
    #[test]
    fn concat_parts_match_materialized() {
        let mut rng = XorShift32::new(2);
        let (ih, iw) = (4u32, 6u32);
        let (c1, c2) = (2usize, 3usize);
        let x1 = rng.vec_f32((ih * iw) as usize * c1);
        let x2 = rng.vec_f32((ih * iw) as usize * c2);
        let px = (ih * iw) as usize;
        let mut cat = vec![0f32; px * (c1 + c2)];
        for p in 0..px {
            cat[p * 5..p * 5 + c1].copy_from_slice(&x1[p * c1..(p + 1) * c1]);
            cat[p * 5 + c1..(p + 1) * 5].copy_from_slice(&x2[p * c2..(p + 1) * c2]);
        }
        let op = ResizeBilinear { oh: 8, ow: 12, mode: CoordMode::HalfPixel };
        let want = reference::resize::resize_bilinear(&op, ih, iw, 5, &cat);
        let mut got = vec![0f32; want.len()];
        resize_bilinear(
            &op, ih, iw,
            &[View::dense(&x1, c1), View::dense(&x2, c2)],
            &mut got,
        );
        assert_eq!(want, got);
    }
}
