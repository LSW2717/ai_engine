//! I420(YUV420p) ↔ RGBA — 모바일 프레임 임포트/익스포트 (플랫폼 몫).
//! 변환 상수는 BT.601 full-range — vcxrust_ai(광원 프로브 포함)와 동일 규약.
//! ⚠ CPU 왕복은 뼈대의 임시 단순화 — 저사양 원칙상 GPU 상주 YUV 커널이
//! 모바일 실연결의 필수 카드다 (NEXT.md).

/// I420 → RGBA8 (알파 255). 호출자가 out 크기(w*h*4)를 보장.
#[allow(clippy::too_many_arguments)]
pub fn i420_to_rgba(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: usize,
    h: usize,
    stride_y: usize,
    stride_u: usize,
    stride_v: usize,
    out: &mut [u8],
) {
    debug_assert!(out.len() >= w * h * 4);
    for yy in 0..h {
        for xx in 0..w {
            let yv = y[yy * stride_y + xx] as f32;
            let uv = u[(yy / 2) * stride_u + xx / 2] as f32 - 128.0;
            let vv = v[(yy / 2) * stride_v + xx / 2] as f32 - 128.0;
            let o = (yy * w + xx) * 4;
            out[o] = (yv + 1.402 * vv).clamp(0.0, 255.0) as u8;
            out[o + 1] = (yv - 0.344136 * uv - 0.714136 * vv).clamp(0.0, 255.0) as u8;
            out[o + 2] = (yv + 1.772 * uv).clamp(0.0, 255.0) as u8;
            out[o + 3] = 255;
        }
    }
}

/// RGBA8 → I420 (in-place로 호출자의 평면에 씀). 크로마는 2×2 블록 좌상단 샘플
/// (평균 아님 — 왕복 시 원본 크로마 보존이 우선).
#[allow(clippy::too_many_arguments)]
pub fn rgba_to_i420(
    rgba: &[u8],
    w: usize,
    h: usize,
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    stride_y: usize,
    stride_u: usize,
    stride_v: usize,
) {
    for yy in 0..h {
        for xx in 0..w {
            let o = (yy * w + xx) * 4;
            let (r, g, b) = (rgba[o] as f32, rgba[o + 1] as f32, rgba[o + 2] as f32);
            y[yy * stride_y + xx] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
            if yy % 2 == 0 && xx % 2 == 0 {
                let co = (yy / 2) * stride_u + xx / 2;
                u[co] = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0).clamp(0.0, 255.0) as u8;
                let co = (yy / 2) * stride_v + xx / 2;
                v[co] = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// RGBA 프레임 뒤집기 — mirror(수평) / 180도 회전(수평+수직). 프레임 변환은
/// 호스트 몫 계약이라 모바일 호스트(ffi)가 추론 전에 적용한다 (웹 워커 전처리 등가).
pub fn flip_rgba(rgba: &mut [u8], w: usize, h: usize, flip_h: bool, flip_v: bool) {
    let stride = w * 4;
    if flip_h {
        for row in rgba[..stride * h].chunks_exact_mut(stride) {
            for x in 0..w / 2 {
                let (a, b) = (x * 4, (w - 1 - x) * 4);
                for k in 0..4 {
                    row.swap(a + k, b + k);
                }
            }
        }
    }
    if flip_v {
        for yy in 0..h / 2 {
            let (top, bottom) = rgba.split_at_mut((h - 1 - yy) * stride);
            top[yy * stride..(yy + 1) * stride].swap_with_slice(&mut bottom[..stride]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_h_v() {
        // 2×2 픽셀 인덱스 [0,1 / 2,3] — R 채널에 인덱스 기록
        let base: Vec<u8> = (0..4u8).flat_map(|i| [i, 0, 0, 255]).collect();
        let r = |buf: &[u8]| [buf[0], buf[4], buf[8], buf[12]];
        let mut m = base.clone();
        flip_rgba(&mut m, 2, 2, true, false);
        assert_eq!(r(&m), [1, 0, 3, 2], "수평");
        let mut m = base.clone();
        flip_rgba(&mut m, 2, 2, false, true);
        assert_eq!(r(&m), [2, 3, 0, 1], "수직");
        let mut m = base.clone();
        flip_rgba(&mut m, 2, 2, true, true);
        assert_eq!(r(&m), [3, 2, 1, 0], "180도");
    }

    #[test]
    fn roundtrip_flat_colors() {
        // 단색 4×2 — Y/UV 왕복 오차 ≤ 3 (f32 반올림 + 420 서브샘플)
        let (w, h) = (4usize, 2usize);
        for color in [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]] {
            let rgba: Vec<u8> =
                (0..w * h).flat_map(|_| [color[0], color[1], color[2], 255]).collect();
            let mut y = vec![0u8; w * h];
            let mut u = vec![0u8; w * h / 4];
            let mut v = vec![0u8; w * h / 4];
            rgba_to_i420(&rgba, w, h, &mut y, &mut u, &mut v, w, w / 2, w / 2);
            let mut back = vec![0u8; w * h * 4];
            i420_to_rgba(&y, &u, &v, w, h, w, w / 2, w / 2, &mut back);
            for i in 0..w * h {
                for c in 0..3 {
                    let d = (back[i * 4 + c] as i32 - color[c] as i32).abs();
                    assert!(d <= 3, "color {color:?} ch{c} diff {d}");
                }
            }
        }
    }
}
