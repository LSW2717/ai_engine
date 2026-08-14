//! 게이즈 전처리 — 비회전 bbox 크롭 → 종횡비 무시 448² 리사이즈(cv2.resize
//! bilinear 규약) → ImageNet 정규화. 웹 focus-tracker 파리티 (크롭 규약이
//! FaceTask의 MediaPipe 회전 크롭과 **다르다** — L2CS가 이 크롭으로 학습됨).
//!
//! cv2.resize bilinear: src 좌표 = (dst + 0.5)·scale − 0.5 (반픽셀 정합),
//! 경계는 클램프(replicate).

/// 정규화 bbox [x0, y0, x1, y1] (프레임 좌표 0..1)
pub type CropBox = [f32; 4];

/// u8 RGB 프레임에서 bbox를 크롭해 out_size² RGB f32 [0,1]로 (인터리브).
/// 크롭 좌표는 픽셀로 반올림하지 않고 **연속 좌표**로 샘플링한다 —
/// cv2.resize(crop) 등가가 되도록 크롭 원점을 소스 좌표 오프셋으로 흡수.
pub fn crop_resize_rgb(
    rgb: &[u8],
    w: usize,
    h: usize,
    b: CropBox,
    out_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(rgb.len(), w * h * 3);
    debug_assert_eq!(out.len(), out_size * out_size * 3);
    let (x0, y0) = (b[0] * w as f32, b[1] * h as f32);
    let (cw, ch) = ((b[2] - b[0]) * w as f32, (b[3] - b[1]) * h as f32);
    let sx = cw / out_size as f32;
    let sy = ch / out_size as f32;
    for oy in 0..out_size {
        let fy = y0 + (oy as f32 + 0.5) * sy - 0.5;
        let fy = fy.clamp(0.0, (h - 1) as f32);
        let iy = (fy.floor() as usize).min(h - 2);
        let ty = fy - iy as f32;
        for ox in 0..out_size {
            let fx = x0 + (ox as f32 + 0.5) * sx - 0.5;
            let fx = fx.clamp(0.0, (w - 1) as f32);
            let ix = (fx.floor() as usize).min(w - 2);
            let tx = fx - ix as f32;
            let idx = |x: usize, y: usize, c: usize| rgb[(y * w + x) * 3 + c] as f32;
            let o = (oy * out_size + ox) * 3;
            for c in 0..3 {
                let top = idx(ix, iy, c) * (1.0 - tx) + idx(ix + 1, iy, c) * tx;
                let bot = idx(ix, iy + 1, c) * (1.0 - tx) + idx(ix + 1, iy + 1, c) * tx;
                out[o + c] = (top * (1.0 - ty) + bot * ty) / 255.0;
            }
        }
    }
}

/// ImageNet 정규화 (RGB [0,1] 인터리브 → in-place)
pub fn imagenet_normalize(buf: &mut [f32]) {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    for px in buf.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] = (px[c] - MEAN[c]) / STD[c];
        }
    }
}

/// 90-bin 로짓 → softmax 기댓값 → 각도 (도): E[i]·4 − 180 (웹 규약)
pub fn decode_bins(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let e: f32 = exps.iter().enumerate().map(|(i, p)| i as f32 * p).sum::<f32>() / sum;
    e * 4.0 - 180.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_identity_when_same_size() {
        // 크롭 = 전체, out=입력 크기 → 항등 (반픽셀 규약 검증)
        let (w, h) = (4usize, 4usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 5 % 251) as u8).collect();
        let mut out = vec![0f32; w * h * 3];
        crop_resize_rgb(&rgb, w, h, [0.0, 0.0, 1.0, 1.0], 4, &mut out);
        for i in 0..rgb.len() {
            assert!((out[i] - rgb[i] as f32 / 255.0).abs() < 1e-6, "@{i}");
        }
    }

    #[test]
    fn decode_bins_peak() {
        // bin 45에 피크 → 45·4−180 = 0도 부근
        let mut l = vec![0f32; 90];
        l[45] = 20.0;
        assert!(decode_bins(&l).abs() < 0.5);
    }
}

/// 478 랜드마크 → 크롭 박스 (웹 faceCrop.ts 1:1): bbox 전체 min/max →
/// margin ×bbox 크기(X 0.18, Y 0.22) → **비대칭 클램프**(재센터 없음) →
/// 픽셀 8px 미만이면 None
pub fn face_crop_box(pts: &[[f32; 2]], vw: f32, vh: f32) -> Option<CropBox> {
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    for p in pts {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    let (bw, bh) = (max_x - min_x, max_y - min_y);
    if bw <= 0.0 || bh <= 0.0 {
        return None;
    }
    let x0 = (min_x - bw * 0.18).max(0.0);
    let y0 = (min_y - bh * 0.22).max(0.0);
    let x1 = (max_x + bw * 0.18).min(1.0);
    let y1 = (max_y + bh * 0.22).min(1.0);
    if (x1 - x0) * vw < 8.0 || (y1 - y0) * vh < 8.0 {
        return None;
    }
    Some([x0, y0, x1, y1])
}

/// EAR 기반 눈감김 (웹 blink.ts — blendshape 없이도 동작하는 절반).
/// 호스트가 blendshape(eyeBlinkL/R ≥ 0.55 AND)를 주면 OR로 결합할 것.
pub fn ears_closed(pts: &[[f32; 2]]) -> bool {
    if pts.len() < 478 {
        return false;
    }
    let d = |a: usize, b: usize| {
        ((pts[a][0] - pts[b][0]).powi(2) + (pts[a][1] - pts[b][1]).powi(2)).sqrt()
    };
    let ear = |ta: usize, ba: usize, tb: usize, bb: usize, o: usize, i: usize| {
        (d(ta, ba) + d(tb, bb)) / (2.0 * d(o, i).max(1e-6))
    };
    let r = ear(159, 145, 158, 153, 33, 133);
    let l = ear(386, 374, 385, 380, 263, 362);
    r < 0.17 && l < 0.17
}
