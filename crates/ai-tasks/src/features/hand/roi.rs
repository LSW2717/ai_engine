//! 손 ROI 수학 — MediaPipe hand_landmarker 그래프의 두 변환을 1:1 이식.
//!
//! ① 팜 검출 → ROI (`hand_detector_graph.cc`):
//!    DetectionsToRects kp0(손목 중심)→kp2(중지 MCP) + RectTransformation
//!    scale 2.6 / shift_y −0.5 / square_long.
//!    ⚠ **target angle은 90 "라디안"** — MediaPipe tasks가 도(°) 필드
//!    (`rotation_vector_target_angle_degrees`) 대신 라디안 필드에 90을 넣는다
//!    (`set_rotation_vector_target_angle(90)`, 검증: detections_to_rects
//!    calculator는 이 필드를 라디안으로 그대로 쓴다). NormalizeRadians(90−θ)
//!    ≈ 116.6°−θ — 의도(90°)와 다르지만 **출하된 동작이므로 파리티 우선 복제**.
//!    lm 모델이 회전에 강건해 실사용은 문제없고, 역투영이 같은 회전을 쓰므로
//!    자기일관적이다.
//!
//! ② 랜드마크 → 다음 프레임 ROI (`hand_landmarks_to_rect_calculator.cc` +
//!    RectTransformation scale 2.0 / shift_y −0.1 / square_long):
//!    회전 = 손목(0) → (thumb-tip(4)+index-tip(8))/2와 index-PIP(6)의 평균,
//!    target π/2. ⚠ 인덱스 4/6/8은 **12점 서브셋 시절 인덱스가 전체 21점에
//!    그대로 적용된 quirk** (원래 의도: index/middle/ring MCP — calculator
//!    주석의 TODO 참조). 이것도 출하된 동작 그대로. 사각형은 회전 프레임에서의
//!    타이트 bbox를 원 프레임으로 되돌린 것.

use crate::detect::roi::{expand_shift_square, normalize_radians, rotation_from_points_target, Roi};
use crate::detect::Detection;

/// 팜 det → ROI 상수 (hand_detector_graph.cc)
const PALM_TARGET_ANGLE_RAD: f32 = 90.0; // ⚠ 라디안 필드에 90 — 헤더 주석 참조
const PALM_KP: (usize, usize) = (0, 2); // 손목 중심 → 중지 MCP
const PALM_SCALE: f32 = 2.6;
const PALM_SHIFT_Y: f32 = -0.5;

/// lm → 다음 ROI 상수 (hand_landmarks_detector_graph.cc)
const LM_SCALE: f32 = 2.0;
const LM_SHIFT_Y: f32 = -0.1;

/// 팜 검출 → 랜드마크 크롭 ROI
pub fn roi_from_palm_detection(d: &Detection, img_w: f32, img_h: f32) -> Roi {
    let rotation = rotation_from_points_target(
        d.keypoints[PALM_KP.0][0] * img_w,
        d.keypoints[PALM_KP.0][1] * img_h,
        d.keypoints[PALM_KP.1][0] * img_w,
        d.keypoints[PALM_KP.1][1] * img_h,
        PALM_TARGET_ANGLE_RAD,
    );
    let cx = (d.xmin + d.xmax) * 0.5 * img_w;
    let cy = (d.ymin + d.ymax) * 0.5 * img_h;
    let w = (d.xmax - d.xmin) * img_w;
    let h = (d.ymax - d.ymin) * img_h;
    expand_shift_square(cx, cy, w, h, rotation, PALM_SCALE, 0.0, PALM_SHIFT_Y)
}

/// 21 랜드마크(원본 정규화 [x,y,z]) → 다음 프레임 ROI
pub fn roi_from_hand_landmarks(pts: &[[f32; 3]], img_w: f32, img_h: f32) -> Roi {
    // ComputeRotation — quirk 인덱스 0/4/6/8 (헤더 주석)
    let x0 = pts[0][0] * img_w;
    let y0 = pts[0][1] * img_h;
    let mut x1 = (pts[4][0] + pts[8][0]) / 2.0;
    let mut y1 = (pts[4][1] + pts[8][1]) / 2.0;
    x1 = (x1 + pts[6][0]) / 2.0 * img_w;
    y1 = (y1 + pts[6][1]) / 2.0 * img_h;
    let rotation =
        normalize_radians(std::f32::consts::FRAC_PI_2 - (-(y1 - y0)).atan2(x1 - x0));
    let reverse = normalize_radians(-rotation);
    let (sin_rev, cos_rev) = reverse.sin_cos();

    // 축정렬 bbox 중심 (정규화)
    let (mut min_x, mut min_y, mut max_x, mut max_y) =
        (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let acx = (min_x + max_x) / 2.0;
    let acy = (min_y + max_y) / 2.0;

    // 회전 프레임에서의 타이트 bbox (절대 px)
    let (mut min_x, mut min_y, mut max_x, mut max_y) =
        (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        let ox = (p[0] - acx) * img_w;
        let oy = (p[1] - acy) * img_h;
        let px = ox * cos_rev - oy * sin_rev;
        let py = ox * sin_rev + oy * cos_rev;
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }
    let pcx = (min_x + max_x) / 2.0;
    let pcy = (min_y + max_y) / 2.0;
    let (sin_rot, cos_rot) = rotation.sin_cos();
    let cx = pcx * cos_rot - pcy * sin_rot + img_w * acx;
    let cy = pcx * sin_rot + pcy * cos_rot + img_h * acy;
    expand_shift_square(cx, cy, max_x - min_x, max_y - min_y, rotation, LM_SCALE, 0.0, LM_SHIFT_Y)
}

/// 축정렬 IoU — `HandAssociationCalculator`의 중복 판정
/// (`rectangle_util.cc` — **회전 무시**, 정규화 좌표. 원본의 TODO 그대로)
pub fn iou_axis_aligned(a: &Roi, b: &Roi, img_w: f32, img_h: f32) -> f32 {
    // Roi는 절대 px — 원본은 정규화 좌표로 계산하지만 IoU는 스케일 불변이라 동일
    let _ = (img_w, img_h);
    let ax0 = a.cx - a.w / 2.0;
    let ay0 = a.cy - a.h / 2.0;
    let bx0 = b.cx - b.w / 2.0;
    let by0 = b.cy - b.h / 2.0;
    let ix = (ax0 + a.w).min(bx0 + b.w) - ax0.max(bx0);
    let iy = (ay0 + a.h).min(by0 + b.h) - ay0.max(by0);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let inter = ix * iy;
    let union = a.w * a.h + b.w * b.h - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palm_roi_upright_hand() {
        // 손목(0.5, 0.8) → 중지 MCP(0.5, 0.6): 위를 향한 손.
        // 의도된 90°였다면 회전 0에 가깝지만, 90rad quirk로는
        // NormalizeRadians(90 − π/2) ≈ 0.4646 rad이어야 한다.
        let d = Detection {
            score: 0.9,
            xmin: 0.4,
            ymin: 0.55,
            xmax: 0.6,
            ymax: 0.85,
            keypoints: vec![
                [0.5, 0.8],
                [0.45, 0.7],
                [0.5, 0.6],
                [0.55, 0.7],
                [0.42, 0.75],
                [0.58, 0.75],
                [0.5, 0.9],
            ],
        };
        let roi = roi_from_palm_detection(&d, 100.0, 100.0);
        let expect = normalize_radians(90.0 - std::f32::consts::FRAC_PI_2);
        assert!((roi.rotation - expect).abs() < 1e-4, "rot={} expect={expect}", roi.rotation);
        // square_long: 박스 20×30 → long 30 × 2.6 = 78
        assert!((roi.w - 78.0).abs() < 1e-3 && (roi.h - 78.0).abs() < 1e-3);
        // shift −0.5는 회전 프레임 기준 — 회전이 0이 아니라 cx/cy 둘 다 움직인다
        let (sinr, cosr) = roi.rotation.sin_cos();
        let ecx = 50.0 - 30.0 * -0.5 * sinr;
        let ecy = 70.0 + 30.0 * -0.5 * cosr;
        assert!((roi.cx - ecx).abs() < 1e-3 && (roi.cy - ecy).abs() < 1e-3);
    }

    #[test]
    fn lm_roi_rotation_uses_quirk_joints() {
        // 전 랜드마크를 한 점에 두고 quirk 조인트(0/4/6/8)만 배치:
        // 손목(0.5,0.8), 4/6/8 전부 (0.5,0.6) → 수직 위 → rot = π/2 − π/2 = 0
        let mut pts = [[0.5f32, 0.7, 0.0]; 21];
        pts[0] = [0.5, 0.8, 0.0];
        pts[4] = [0.5, 0.6, 0.0];
        pts[6] = [0.5, 0.6, 0.0];
        pts[8] = [0.5, 0.6, 0.0];
        let roi = roi_from_hand_landmarks(&pts, 100.0, 100.0);
        assert!(roi.rotation.abs() < 1e-5, "rot={}", roi.rotation);
        // bbox 0×20px → square_long 20 × 2.0 = 40, shift −0.1(회전0) → cy −2
        assert!((roi.w - 40.0).abs() < 1e-3);
        let ecy = (0.6 + 0.8) / 2.0 * 100.0 - 20.0 * 0.1;
        assert!((roi.cy - ecy).abs() < 1e-3, "cy={} expect={ecy}", roi.cy);
    }

    #[test]
    fn iou_identical_and_disjoint() {
        let a = Roi { cx: 50.0, cy: 50.0, w: 20.0, h: 20.0, rotation: 1.0 };
        assert!((iou_axis_aligned(&a, &a, 100.0, 100.0) - 1.0).abs() < 1e-6);
        let b = Roi { cx: 90.0, cy: 90.0, w: 20.0, h: 20.0, rotation: 0.0 };
        assert_eq!(iou_axis_aligned(&a, &b, 100.0, 100.0), 0.0);
    }
}
