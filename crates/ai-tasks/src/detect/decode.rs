//! 날 텐서 → 검출 — MediaPipe `TensorsToDetectionsCalculator` 등가.
//!
//! 디텍터 출력은 [앵커수, num_coords] 회귀와 [앵커수, 1] 점수 로짓이다.
//! 두 프리셋(BlazeFace/palm) 모두 `reverse_output_order=true`라 회귀 순서는
//! [cx, cy, w, h, kp0x, kp0y, ...]로 고정한다.

use super::anchors::Anchor;
use super::Detection;

pub struct DecodeOpts {
    /// 앵커당 회귀 값 개수 (face 16 / palm 18)
    pub num_coords: usize,
    pub num_keypoints: usize,
    /// 회귀 행 안에서 키포인트가 시작하는 오프셋 (둘 다 4)
    pub keypoint_offset: usize,
    /// 좌표 역정규화 스케일 — MediaPipe는 x/y/w/h 별도지만 우리 프리셋은 전부 입력 한 변
    pub x_scale: f32,
    pub y_scale: f32,
    pub w_scale: f32,
    pub h_scale: f32,
    /// 점수 로짓 클리핑 (sigmoid 전, ±100)
    pub score_clip: f32,
    /// 이 미만은 NMS 전에 버린다
    pub min_score: f32,
}

pub fn decode(
    anchors: &[Anchor],
    opts: &DecodeOpts,
    raw_boxes: &[f32],
    raw_scores: &[f32],
) -> Vec<Detection> {
    debug_assert_eq!(raw_boxes.len(), anchors.len() * opts.num_coords);
    debug_assert_eq!(raw_scores.len(), anchors.len());

    let mut out = Vec::new();
    for (i, a) in anchors.iter().enumerate() {
        let logit = raw_scores[i].clamp(-opts.score_clip, opts.score_clip);
        let score = 1.0 / (1.0 + (-logit).exp());
        if score < opts.min_score {
            continue;
        }
        let r = &raw_boxes[i * opts.num_coords..(i + 1) * opts.num_coords];
        let cx = r[0] / opts.x_scale * a.w + a.x;
        let cy = r[1] / opts.y_scale * a.h + a.y;
        let w = r[2] / opts.w_scale * a.w;
        let h = r[3] / opts.h_scale * a.h;
        let keypoints = (0..opts.num_keypoints)
            .map(|k| {
                let kx = r[opts.keypoint_offset + 2 * k] / opts.x_scale * a.w + a.x;
                let ky = r[opts.keypoint_offset + 2 * k + 1] / opts.y_scale * a.h + a.y;
                [kx, ky]
            })
            .collect();
        out.push(Detection {
            score,
            xmin: cx - w * 0.5,
            ymin: cy - h * 0.5,
            xmax: cx + w * 0.5,
            ymax: cy + h * 0.5,
            keypoints,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> DecodeOpts {
        DecodeOpts {
            num_coords: 6,
            num_keypoints: 1,
            keypoint_offset: 4,
            x_scale: 128.0,
            y_scale: 128.0,
            w_scale: 128.0,
            h_scale: 128.0,
            score_clip: 100.0,
            min_score: 0.5,
        }
    }

    #[test]
    fn box_and_keypoint() {
        let anchors = [Anchor { x: 0.5, y: 0.5, w: 1.0, h: 1.0 }];
        // cx+16px, cy 0, 64×64px 박스, 키포인트 (+12.8, -12.8)px
        let raw = [16.0, 0.0, 64.0, 64.0, 12.8, -12.8];
        let d = decode(&anchors, &opts(), &raw, &[10.0]);
        assert_eq!(d.len(), 1);
        let d = &d[0];
        assert!((d.xmin - 0.375).abs() < 1e-6, "xmin {}", d.xmin);
        assert!((d.xmax - 0.875).abs() < 1e-6);
        assert!((d.ymin - 0.25).abs() < 1e-6);
        assert!((d.ymax - 0.75).abs() < 1e-6);
        assert!((d.keypoints[0][0] - 0.6).abs() < 1e-6);
        assert!((d.keypoints[0][1] - 0.4).abs() < 1e-6);
        assert!(d.score > 0.99);
    }

    #[test]
    fn low_score_dropped_and_clip() {
        let anchors = [Anchor { x: 0.5, y: 0.5, w: 1.0, h: 1.0 }];
        let raw = [0.0; 6];
        // 로짓 0 → score 0.5 = 문턱 통과, -1 → 0.27 탈락, 1e9 → 클립 후 1.0
        assert_eq!(decode(&anchors, &opts(), &raw, &[0.0]).len(), 1);
        assert_eq!(decode(&anchors, &opts(), &raw, &[-1.0]).len(), 0);
        let d = decode(&anchors, &opts(), &raw, &[1e9]);
        assert!(d[0].score <= 1.0 && d[0].score > 0.999);
    }
}
