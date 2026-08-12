//! bilinear 리사이즈 CPU 레퍼런스.

use crate::ops::{CoordMode, ResizeBilinear};

pub fn resize_bilinear(
    op: &ResizeBilinear,
    ih: u32,
    iw: u32,
    c: u32,
    input: &[f32],
) -> Vec<f32> {
    assert_eq!(input.len(), (ih * iw * c) as usize);
    let (oh, ow) = (op.oh, op.ow);
    let sy = ih as f32 / oh as f32;
    let sx = iw as f32 / ow as f32;
    let src_coord = |dst: u32, scale: f32| -> f32 {
        match op.mode {
            CoordMode::HalfPixel => (dst as f32 + 0.5) * scale - 0.5,
            CoordMode::Asymmetric => dst as f32 * scale,
        }
    };
    let mut out = vec![0f32; (oh * ow * c) as usize];
    for oy in 0..oh {
        let fy = src_coord(oy, sy);
        let y0 = (fy.floor() as i64).clamp(0, ih as i64 - 1) as u32;
        let y1 = (fy.floor() as i64 + 1).clamp(0, ih as i64 - 1) as u32;
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);
        for ox in 0..ow {
            let fx = src_coord(ox, sx);
            let x0 = (fx.floor() as i64).clamp(0, iw as i64 - 1) as u32;
            let x1 = (fx.floor() as i64 + 1).clamp(0, iw as i64 - 1) as u32;
            let tx = (fx - fx.floor()).clamp(0.0, 1.0);
            for ch in 0..c {
                let g = |y: u32, x: u32| input[((y * iw + x) * c + ch) as usize];
                let top = g(y0, x0) * (1.0 - tx) + g(y0, x1) * tx;
                let bot = g(y1, x0) * (1.0 - tx) + g(y1, x1) * tx;
                out[((oy * ow + ox) * c + ch) as usize] = top * (1.0 - ty) + bot * ty;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_2x_half_pixel_1d_hand_computed() {
        // 1×2 [0,1] → 1×4, half_pixel: [0, 0.25, 0.75, 1]
        let op = ResizeBilinear { oh: 1, ow: 4, mode: CoordMode::HalfPixel };
        let out = resize_bilinear(&op, 1, 2, 1, &[0.0, 1.0]);
        assert_eq!(out, vec![0.0, 0.25, 0.75, 1.0]);
    }

    #[test]
    fn upsample_2x_asymmetric_1d_hand_computed() {
        // 1×2 [0,1] → 1×4, asymmetric: [0, 0.5, 1, 1]
        let op = ResizeBilinear { oh: 1, ow: 4, mode: CoordMode::Asymmetric };
        let out = resize_bilinear(&op, 1, 2, 1, &[0.0, 1.0]);
        assert_eq!(out, vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn identity_and_constant_invariants() {
        let input = vec![3.0; 5 * 7 * 2];
        // 같은 크기 → 항등
        let op = ResizeBilinear { oh: 5, ow: 7, mode: CoordMode::HalfPixel };
        assert_eq!(resize_bilinear(&op, 5, 7, 2, &input), input);
        // 상수 입력 2× → 상수 유지
        let op2 = ResizeBilinear { oh: 10, ow: 14, mode: CoordMode::HalfPixel };
        assert!(resize_bilinear(&op2, 5, 7, 2, &input).iter().all(|v| (v - 3.0).abs() < 1e-6));
    }
}
