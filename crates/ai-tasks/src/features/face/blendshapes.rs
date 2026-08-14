//! face_blendshapes (MLP-Mixer GhumMarkerPoser) 입력·출력 규약 — MediaPipe
//! face_blendshapes_graph.cc에서 확정 (2026-08-14, 원본 소스 대조):
//!
//! - 입력 [1,146,2]: 478점 메시의 **146점 서브셋**(kLandmarksSubsetIdxs)을
//!   `LandmarksToTensorCalculator`(attributes X,Y / flatten=false / IMAGE_SIZE
//!   공급)로 변환 — 즉 **프레임 픽셀 좌표 (x×W, y×H), 센터링·추가 스케일 없음**
//!   (모델이 내부에서 L2 정규화 — 변환기 canon의 ReduceSum이 그것).
//! - 출력 [52]: TensorsToClassification 순서 고정 — 0 `_neutral`,
//!   **9 `eyeBlinkLeft`, 10 `eyeBlinkRight`**, 44/45 mouthSmileL/R.
//!
//! 웹 blink 규약(focus-tracker/vision/blink.ts): bsL≥0.55 AND bsR≥0.55가
//! EAR 절반과 OR로 결합된다.

/// kLandmarksSubsetIdxs — 478점 → 146점 (MediaPipe 순서 그대로)
pub const SUBSET_146: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65,
    66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136,
    144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168,
    172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276,
    282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
    381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
];

pub const EYE_BLINK_LEFT: usize = 9;
pub const EYE_BLINK_RIGHT: usize = 10;
/// 웹 BLINK.closedScore
pub const BLINK_CLOSED_SCORE: f32 = 0.55;

/// 정규화 랜드마크(x,y ∈ 프레임 정규화) → 모델 입력 292 f32 (프레임 px, 인터리브).
/// 점이 478개 미만이면 None (refined 메시 전제 — 서브셋에 홍채 468~477 포함).
pub fn input_from_landmarks(pts: &[[f32; 2]], w: f32, h: f32) -> Option<Vec<f32>> {
    if pts.len() < 478 {
        return None;
    }
    let mut out = Vec::with_capacity(146 * 2);
    for &i in SUBSET_146.iter() {
        out.push(pts[i][0] * w);
        out.push(pts[i][1] * h);
    }
    Some(out)
}

/// 52계수에서 blink 절반 판정 (양눈 AND — 웹 blink.ts bsClosed)
pub fn blink_closed(coeffs: &[f32]) -> bool {
    coeffs.len() > EYE_BLINK_RIGHT
        && coeffs[EYE_BLINK_LEFT] >= BLINK_CLOSED_SCORE
        && coeffs[EYE_BLINK_RIGHT] >= BLINK_CLOSED_SCORE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_shape_and_order() {
        assert_eq!(SUBSET_146.len(), 146);
        // 단조증가 (MediaPipe 리스트 특성 — 오타 방지 가드)
        assert!(SUBSET_146.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*SUBSET_146.last().unwrap(), 477, "홍채 포함 refined 메시");
        let pts = vec![[0.5f32, 0.25]; 478];
        let inp = input_from_landmarks(&pts, 640.0, 360.0).unwrap();
        assert_eq!(inp.len(), 292);
        assert_eq!(inp[0], 320.0);
        assert_eq!(inp[1], 90.0);
        assert!(input_from_landmarks(&pts[..468], 640.0, 360.0).is_none());
    }

    #[test]
    fn blink_and_gate() {
        let mut c = vec![0.0f32; 52];
        c[EYE_BLINK_LEFT] = 0.9;
        assert!(!blink_closed(&c), "한눈만 감김은 미발화 (AND)");
        c[EYE_BLINK_RIGHT] = 0.6;
        assert!(blink_closed(&c));
    }
}
