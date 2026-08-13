//! 레터박스 — MediaPipe `ImageToTensorCalculator`(keep_aspect_ratio)의 패딩 계산과
//! 그 역투영. 디텍터는 정사각 입력이라 원본 프레임을 비율 유지로 축소하고 남는
//! 변을 대칭 패딩하는데, 검출 좌표는 그 패딩 공간 기준이므로 원본 좌표로
//! 되돌리려면 같은 패딩 값으로 역투영해야 한다.
//!
//! 픽셀을 실제로 채우는 쪽(JS 캔버스·네이티브 테스트)도 반드시 이 계산과 같은
//! 기하(중앙 정렬, min-scale)를 써야 한다 — 패딩이 1px만 달라도 좌표가 어긋난다.

/// 정규화 패딩 (dst 기준 한쪽 값 — 대칭이라 좌=우, 상=하)
#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    pub fn fit(src_w: f32, src_h: f32, dst_w: f32, dst_h: f32) -> Self {
        let scale = (dst_w / src_w).min(dst_h / src_h);
        Letterbox {
            pad_x: (1.0 - scale * src_w / dst_w) * 0.5,
            pad_y: (1.0 - scale * src_h / dst_h) * 0.5,
        }
    }

    /// dst(레터박스) 정규화 좌표 → src(원본) 정규화 좌표
    pub fn unproject(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.pad_x) / (1.0 - 2.0 * self.pad_x),
            (y - self.pad_y) / (1.0 - 2.0 * self.pad_y),
        )
    }
}

/// u8 RGB 프레임 → dst 크기 레터박스 + [-1,1] 정규화 (bilinear, 패딩은 검정=-1).
///
/// MediaPipe ImageToTensor(BORDER_ZERO)와 같은 기하: 검정 패딩 픽셀이 범위 변환을
/// 거쳐 -1이 된다. 웹은 검정으로 채운 캔버스에 drawImage가 같은 일을 하므로
/// 이 함수는 캔버스 없는 호스트(ffi)와 네이티브 테스트용이다.
pub fn letterbox_u8_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), src_w * src_h * 3);
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let (cw, ch) = (src_w as f32 * scale, src_h as f32 * scale);
    let (ox, oy) = ((dst_w as f32 - cw) * 0.5, (dst_h as f32 - ch) * 0.5);
    let mut out = vec![-1.0f32; dst_w * dst_h * 3];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let fx = dx as f32 + 0.5 - ox;
            let fy = dy as f32 + 0.5 - oy;
            if fx < 0.0 || fy < 0.0 || fx > cw || fy > ch {
                continue;
            }
            // 픽셀 중심 정렬 bilinear (OpenCV INTER_LINEAR과 같은 좌표계)
            let sx = (fx / scale - 0.5).clamp(0.0, src_w as f32 - 1.0);
            let sy = (fy / scale - 0.5).clamp(0.0, src_h as f32 - 1.0);
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(src_w - 1), (y0 + 1).min(src_h - 1));
            let (tx, ty) = (sx - x0 as f32, sy - y0 as f32);
            for c in 0..3 {
                let p = |x: usize, y: usize| src[(y * src_w + x) * 3 + c] as f32;
                let v = p(x0, y0) * (1.0 - tx) * (1.0 - ty)
                    + p(x1, y0) * tx * (1.0 - ty)
                    + p(x0, y1) * (1.0 - tx) * ty
                    + p(x1, y1) * tx * ty;
                out[(dy * dst_w + dx) * 3 + c] = v / 127.5 - 1.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_pixels_pad_black() {
        // 2×1 흰 프레임 → 4×4: 콘텐츠는 가운데 두 행 부근, 위아래는 -1
        let src = [255u8; 6];
        let out = letterbox_u8_rgb(&src, 2, 1, 4, 4);
        assert_eq!(out.len(), 48);
        assert!(out[0] == -1.0, "좌상단은 패딩");
        // 중앙 행(y=1,2 경계)의 중심 픽셀은 흰색
        let mid = (1 * 4 + 1) * 3;
        assert!((out[mid] - 1.0).abs() < 1e-4, "중앙 {}", out[mid]);
    }

    #[test]
    fn wide_frame_pads_vertically() {
        // 256×144 → 128×128: scale 0.5, 세로 72px → 패딩 (128-72)/2/128
        let lb = Letterbox::fit(256.0, 144.0, 128.0, 128.0);
        assert!(lb.pad_x.abs() < 1e-6);
        assert!((lb.pad_y - 0.21875).abs() < 1e-6);
        // dst 세로 중앙은 src 세로 중앙
        let (x, y) = lb.unproject(0.5, 0.5);
        assert!((x - 0.5).abs() < 1e-6 && (y - 0.5).abs() < 1e-6);
        // dst 패딩 경계 = src 0.0 / 1.0
        let (_, top) = lb.unproject(0.5, 0.21875);
        let (_, bot) = lb.unproject(0.5, 1.0 - 0.21875);
        assert!(top.abs() < 1e-6 && (bot - 1.0).abs() < 1e-6);
    }

    #[test]
    fn square_is_identity() {
        let lb = Letterbox::fit(640.0, 640.0, 128.0, 128.0);
        let (x, y) = lb.unproject(0.3, 0.7);
        assert!((x - 0.3).abs() < 1e-6 && (y - 0.7).abs() < 1e-6);
    }
}
