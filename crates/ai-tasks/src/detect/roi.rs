//! ROI — 검출/랜드마크 → 회전 정규화 크롭 → 역투영.
//!
//! MediaPipe calculator 3개의 등가: `DetectionsToRectsCalculator`(키포인트 2점으로
//! 회전 계산), `RectTransformationCalculator`(square_long + scale 확장),
//! `ImageToTensorCalculator`의 ROI warp(+`LandmarkProjectionCalculator` 역투영).
//!
//! 좌표 규약: `Roi`는 **절대 픽셀** (cx,cy,w,h,rotation rad) — MediaPipe는
//! NormalizedRect를 들고 다니며 calculator마다 이미지 크기로 환산하는데, 그
//! 왕복에서 실수하기 쉬워 내부는 절대 픽셀 하나로 고정한다. square_long 확장
//! 덕에 w==h라서 회전을 정규화 공간에 적용하는 MediaPipe 투영식과도 일치한다.

use super::Detection;

#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    /// 라디안, 이미지 좌표(y 아래) 기준
    pub rotation: f32,
}

/// MediaPipe NormalizeRadians — (-π, π]로 접기
fn normalize_radians(a: f32) -> f32 {
    a - 2.0 * std::f32::consts::PI
        * ((a + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)).floor()
}

/// 두 키포인트(절대 px)를 잇는 선이 X축과 이루는 각을 target 0으로 보정하는 회전
/// (`DetectionsToRectsCalculator::ComputeRotation` 등가)
fn rotation_from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    normalize_radians(-(-(y1 - y0)).atan2(x1 - x0))
}

/// 중심·크기(절대 px)에 square_long + scale을 적용해 Roi로
/// (`RectTransformationCalculator{scale, square_long, shift 0}` 등가)
fn expand_square(cx: f32, cy: f32, w: f32, h: f32, rotation: f32, scale: f32) -> Roi {
    let side = w.max(h) * scale;
    Roi { cx, cy, w: side, h: side, rotation }
}

/// 검출 → ROI: 박스 중심/크기 + 키포인트 2점 회전 + square_long·scale 확장.
/// face: kp (0,1)=양눈, scale 1.5. (hand는 kp·scale·shift가 달라 별도 프리셋으로.)
pub fn roi_from_detection(
    d: &Detection,
    kp_start: usize,
    kp_end: usize,
    scale: f32,
    img_w: f32,
    img_h: f32,
) -> Roi {
    let rotation = rotation_from_points(
        d.keypoints[kp_start][0] * img_w,
        d.keypoints[kp_start][1] * img_h,
        d.keypoints[kp_end][0] * img_w,
        d.keypoints[kp_end][1] * img_h,
    );
    let cx = (d.xmin + d.xmax) * 0.5 * img_w;
    let cy = (d.ymin + d.ymax) * 0.5 * img_h;
    let w = (d.xmax - d.xmin) * img_w;
    let h = (d.ymax - d.ymin) * img_h;
    expand_square(cx, cy, w, h, rotation, scale)
}

/// 랜드마크(원본 정규화 [x,y,z]) → 다음 프레임 ROI:
/// 전체 min/max 박스 + 지정 랜드마크 2점 회전 + 동일 확장
/// (`LandmarksToDetectionCalculator` + face_landmarks_to_roi 등가.
/// face: kp (33,263)=양눈 바깥꼬리, scale 1.5)
pub fn roi_from_landmarks(
    pts: &[[f32; 3]],
    kp_start: usize,
    kp_end: usize,
    scale: f32,
    img_w: f32,
    img_h: f32,
) -> Roi {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let rotation = rotation_from_points(
        pts[kp_start][0] * img_w,
        pts[kp_start][1] * img_h,
        pts[kp_end][0] * img_w,
        pts[kp_end][1] * img_h,
    );
    expand_square(
        (x0 + x1) * 0.5 * img_w,
        (y0 + y1) * 0.5 * img_h,
        (x1 - x0) * img_w,
        (y1 - y0) * img_h,
        rotation,
        scale,
    )
}

/// u8 RGB 프레임에서 회전 ROI를 dst×dst로 warp — bilinear, **replicate 경계**,
/// [0,1] 정규화 (랜드마크 모델 입력 규약).
///
/// 샘플링 규약은 OpenCV `warpPerspective`(MediaPipe CPU 경로)와 동일: dst 정수
/// 픽셀 (x,y)가 코너 정합 아핀 그대로 매핑된다 (+0.5 텍셀 센터 보정 없음 —
/// GL 경로와 반픽셀 차이가 있지만 기준 게이트가 CPU delegate라 이쪽을 따른다).
pub fn crop_u8_rgb(
    frame: &[u8],
    img_w: usize,
    img_h: usize,
    roi: &Roi,
    dst: usize,
) -> Vec<f32> {
    assert_eq!(frame.len(), img_w * img_h * 3);
    let (sinr, cosr) = roi.rotation.sin_cos();
    let mut out = vec![0.0f32; dst * dst * 3];
    for y in 0..dst {
        let dy = y as f32 / dst as f32 - 0.5;
        for x in 0..dst {
            let dx = x as f32 / dst as f32 - 0.5;
            let sx = roi.cx + roi.w * dx * cosr - roi.h * dy * sinr;
            let sy = roi.cy + roi.w * dx * sinr + roi.h * dy * cosr;
            let sx = sx.clamp(0.0, img_w as f32 - 1.0);
            let sy = sy.clamp(0.0, img_h as f32 - 1.0);
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(img_w - 1), (y0 + 1).min(img_h - 1));
            let (tx, ty) = (sx - x0 as f32, sy - y0 as f32);
            let o = (y * dst + x) * 3;
            for c in 0..3 {
                let p = |px: usize, py: usize| frame[(py * img_w + px) * 3 + c] as f32;
                let v = p(x0, y0) * (1.0 - tx) * (1.0 - ty)
                    + p(x1, y0) * tx * (1.0 - ty)
                    + p(x0, y1) * (1.0 - tx) * ty
                    + p(x1, y1) * tx * ty;
                out[o + c] = v / 255.0;
            }
        }
    }
    out
}

/// 크롭 정규화 랜드마크 → 원본 프레임 정규화 (`LandmarkProjectionCalculator` 등가).
/// z는 rect 폭으로 스케일. ⚠ 회전을 정규화 공간에서 적용하는 MediaPipe 식 그대로 —
/// square_long이 보장하는 w==h(절대 px) 전제에서만 크롭 변환의 정확한 역이다.
pub fn project_landmarks(pts: &mut [[f32; 3]], roi: &Roi, img_w: f32, img_h: f32) {
    let (sinr, cosr) = roi.rotation.sin_cos();
    let (rw, rh) = (roi.w / img_w, roi.h / img_h);
    let (rcx, rcy) = (roi.cx / img_w, roi.cy / img_h);
    for p in pts {
        let (x, y) = (p[0] - 0.5, p[1] - 0.5);
        p[0] = (cosr * x - sinr * y) * rw + rcx;
        p[1] = (sinr * x + cosr * y) * rh + rcy;
        p[2] *= rw;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_level_eyes_is_zero() {
        assert!(rotation_from_points(10.0, 50.0, 60.0, 50.0).abs() < 1e-6);
        // 끝점(kp1)이 아래로 → 선이 시계방향(이미지 좌표) → 보정 회전은 양수
        let r = rotation_from_points(10.0, 50.0, 60.0, 60.0);
        assert!(r > 0.0 && r < std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn expand_makes_square() {
        let d = Detection {
            score: 1.0,
            xmin: 0.4,
            ymin: 0.3,
            xmax: 0.6,
            ymax: 0.7,
            keypoints: vec![[0.45, 0.4], [0.55, 0.4]],
        };
        // 100×100 이미지: 박스 20×40 → long 40 × 1.5 = 60² 정사각
        let roi = roi_from_detection(&d, 0, 1, 1.5, 100.0, 100.0);
        assert!((roi.w - 60.0).abs() < 1e-4 && (roi.h - 60.0).abs() < 1e-4);
        assert!((roi.cx - 50.0).abs() < 1e-4 && (roi.cy - 50.0).abs() < 1e-4);
        assert!(roi.rotation.abs() < 1e-6);
    }

    #[test]
    fn crop_project_roundtrip() {
        // 회전 없는 ROI: 크롭 중심(0.5,0.5)의 역투영은 ROI 중심이어야 한다
        let roi = Roi { cx: 60.0, cy: 40.0, w: 30.0, h: 30.0, rotation: 0.3 };
        let mut pts = [[0.5f32, 0.5, 0.0]];
        project_landmarks(&mut pts, &roi, 100.0, 100.0);
        assert!((pts[0][0] - 0.6).abs() < 1e-5 && (pts[0][1] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn crop_reads_expected_pixel() {
        // 4×4 프레임, (2,1)만 빨강 — ROI를 그 픽셀 중심에 두면 크롭 중심이 빨강
        let mut frame = vec![0u8; 4 * 4 * 3];
        frame[(1 * 4 + 2) * 3] = 255;
        let roi = Roi { cx: 2.0, cy: 1.0, w: 2.0, h: 2.0, rotation: 0.0 };
        let crop = crop_u8_rgb(&frame, 4, 4, &roi, 4);
        // dst(2,2): dx=dy=0 → src (2.0, 1.0) 정확히
        let c = (2 * 4 + 2) * 3;
        assert!((crop[c] - 1.0).abs() < 1e-6, "r={}", crop[c]);
        assert!(crop[c + 1].abs() < 1e-6);
    }
}
