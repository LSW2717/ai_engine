//! bilinear 리사이즈 — concat 융합(parts) 지원.
//!
//! 리사이즈는 채널 독립이라 파트별로 출력 채널 구간에 직접 쓴다 —
//! concat을 실체화할 필요가 없다. 좌표 규약은 레퍼런스와 동일.

use ai_core::ops::{CoordMode, ResizeBilinear};

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

    for oy in 0..oh {
        let fy = src(oy, sy);
        let y0 = (fy.floor() as i64).clamp(0, ih as i64 - 1) as usize;
        let y1 = (fy.floor() as i64 + 1).clamp(0, ih as i64 - 1) as usize;
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);
        for ox in 0..ow {
            let fx = src(ox, sx);
            let x0 = (fx.floor() as i64).clamp(0, iw as i64 - 1) as usize;
            let x1 = (fx.floor() as i64 + 1).clamp(0, iw as i64 - 1) as usize;
            let tx = (fx - fx.floor()).clamp(0.0, 1.0);
            let ob = (oy * ow + ox) * c_total;
            let mut c_off_out = 0usize;
            for part in parts {
                let (b00, b01) = (part.base(y0 * iw as usize + x0), part.base(y0 * iw as usize + x1));
                let (b10, b11) = (part.base(y1 * iw as usize + x0), part.base(y1 * iw as usize + x1));
                for ch in 0..part.c {
                    let top = part.data[b00 + ch] * (1.0 - tx) + part.data[b01 + ch] * tx;
                    let bot = part.data[b10 + ch] * (1.0 - tx) + part.data[b11 + ch] * tx;
                    out[ob + c_off_out + ch] = top * (1.0 - ty) + bot * ty;
                }
                c_off_out += part.c;
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
