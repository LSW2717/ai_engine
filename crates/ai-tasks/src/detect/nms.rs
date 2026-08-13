//! 가중 NMS — MediaPipe `NonMaxSuppressionCalculator`의 WEIGHTED 알고리즘 등가.
//!
//! 일반 NMS처럼 겹치는 검출을 버리는 게 아니라, IoU가 문턱을 넘는 후보들의
//! 박스·키포인트를 점수 가중 평균해 하나로 합친다 (점수는 최고점 것을 유지).
//! BlazeFace는 한 얼굴에 앵커 수십 개가 동시에 반응하므로 이 평균이
//! 사실상의 서브픽셀 정밀화다 — 일반 NMS로 바꾸면 박스가 눈에 띄게 떤다.

use super::Detection;

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix = (a.xmax.min(b.xmax) - a.xmin.max(b.xmin)).max(0.0);
    let iy = (a.ymax.min(b.ymax) - a.ymin.max(b.ymin)).max(0.0);
    let inter = ix * iy;
    let ua = (a.xmax - a.xmin) * (a.ymax - a.ymin)
        + (b.xmax - b.xmin) * (b.ymax - b.ymin)
        - inter;
    if ua <= 0.0 {
        0.0
    } else {
        inter / ua
    }
}

pub fn weighted_nms(mut dets: Vec<Detection>, min_suppression: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    let mut remained = dets;
    while let Some(pivot) = remained.first().cloned() {
        let mut candidates = Vec::new();
        let mut rest = Vec::new();
        for d in remained {
            // pivot 자신도 IoU 1.0으로 후보에 들어간다 (MediaPipe와 동일)
            if iou(&pivot, &d) > min_suppression {
                candidates.push(d);
            } else {
                rest.push(d);
            }
        }
        let mut merged = pivot;
        let total: f32 = candidates.iter().map(|d| d.score).sum();
        if total > 0.0 {
            merged.xmin = candidates.iter().map(|d| d.xmin * d.score).sum::<f32>() / total;
            merged.ymin = candidates.iter().map(|d| d.ymin * d.score).sum::<f32>() / total;
            merged.xmax = candidates.iter().map(|d| d.xmax * d.score).sum::<f32>() / total;
            merged.ymax = candidates.iter().map(|d| d.ymax * d.score).sum::<f32>() / total;
            for k in 0..merged.keypoints.len() {
                let kx = candidates.iter().map(|d| d.keypoints[k][0] * d.score).sum::<f32>();
                let ky = candidates.iter().map(|d| d.keypoints[k][1] * d.score).sum::<f32>();
                merged.keypoints[k] = [kx / total, ky / total];
            }
        }
        out.push(merged);
        remained = rest;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(score: f32, xmin: f32, ymin: f32, size: f32, kp: f32) -> Detection {
        Detection {
            score,
            xmin,
            ymin,
            xmax: xmin + size,
            ymax: ymin + size,
            keypoints: vec![[kp, kp]],
        }
    }

    #[test]
    fn overlapping_blend() {
        // 0.2 오프셋의 0.4×0.4 두 박스: IoU = 0.04/0.28 ≈ 0.143... 겹치게 더 크게.
        let a = det(0.8, 0.10, 0.10, 0.40, 0.30);
        let b = det(0.4, 0.14, 0.14, 0.40, 0.34);
        assert!(iou(&a, &b) > 0.3);
        let out = weighted_nms(vec![b.clone(), a.clone()], 0.3);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        // 가중 평균: (0.10*0.8 + 0.14*0.4) / 1.2 = 0.11333
        assert!((m.xmin - 0.11333).abs() < 1e-4, "xmin {}", m.xmin);
        assert!((m.keypoints[0][0] - (0.30 * 0.8 + 0.34 * 0.4) / 1.2).abs() < 1e-5);
        // 점수는 최고점 유지
        assert!((m.score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn disjoint_kept() {
        let a = det(0.9, 0.0, 0.0, 0.3, 0.1);
        let b = det(0.7, 0.6, 0.6, 0.3, 0.7);
        let out = weighted_nms(vec![a, b], 0.3);
        assert_eq!(out.len(), 2);
        // 점수 내림차순 출력
        assert!(out[0].score > out[1].score);
    }
}
