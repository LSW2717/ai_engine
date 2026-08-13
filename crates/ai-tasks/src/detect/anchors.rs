//! SSD 앵커 생성 — MediaPipe `SsdAnchorsCalculator` 등가.
//!
//! 디텍터(BlazeFace/palm)는 출력 행 하나가 앵커 하나에 대응한다. 앵커 좌표가
//! 한 칸이라도 어긋나면 디코드된 박스 전체가 틀어지므로, 생성 순서까지
//! MediaPipe와 같아야 한다 (같은 stride 층을 묶어 셀당 앵커를 이어 붙이는 순서).

/// 정규화 좌표 앵커. `fixed_anchor_size`면 w=h=1 (BlazeFace/palm 둘 다).
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// `SsdAnchorsCalculatorOptions` 중 우리 프리셋이 쓰는 부분집합.
/// (`reduce_boxes_in_lowest_layer`는 두 프리셋 다 false라 구현하지 않는다.)
pub struct AnchorConfig {
    pub input_w: u32,
    pub input_h: u32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub strides: &'static [u32],
    pub aspect_ratios: &'static [f32],
    pub anchor_offset_x: f32,
    pub anchor_offset_y: f32,
    /// >0이면 층마다 다음 층 scale과의 기하평균 앵커를 하나 더 얹는다
    pub interpolated_scale_aspect_ratio: f32,
    pub fixed_anchor_size: bool,
}

fn calc_scale(min: f32, max: f32, idx: usize, num: usize) -> f32 {
    if num == 1 {
        (min + max) * 0.5
    } else {
        min + (max - min) * idx as f32 / (num as f32 - 1.0)
    }
}

pub fn generate(cfg: &AnchorConfig) -> Vec<Anchor> {
    let n = cfg.strides.len();
    let mut anchors = Vec::new();
    let mut layer = 0;
    while layer < n {
        // 같은 stride가 연달아 나오는 층들은 한 피처맵에 셀당 앵커로 합쳐진다
        let mut widths = Vec::new();
        let mut heights = Vec::new();
        let mut last = layer;
        while last < n && cfg.strides[last] == cfg.strides[layer] {
            let scale = calc_scale(cfg.min_scale, cfg.max_scale, last, n);
            for &ar in cfg.aspect_ratios {
                let r = ar.sqrt();
                widths.push(scale * r);
                heights.push(scale / r);
            }
            if cfg.interpolated_scale_aspect_ratio > 0.0 {
                let next = if last == n - 1 {
                    1.0
                } else {
                    calc_scale(cfg.min_scale, cfg.max_scale, last + 1, n)
                };
                let s = (scale * next).sqrt();
                let r = cfg.interpolated_scale_aspect_ratio.sqrt();
                widths.push(s * r);
                heights.push(s / r);
            }
            last += 1;
        }
        let stride = cfg.strides[layer];
        let fh = cfg.input_h.div_ceil(stride);
        let fw = cfg.input_w.div_ceil(stride);
        for y in 0..fh {
            for x in 0..fw {
                for i in 0..widths.len() {
                    anchors.push(Anchor {
                        x: (x as f32 + cfg.anchor_offset_x) / fw as f32,
                        y: (y as f32 + cfg.anchor_offset_y) / fh as f32,
                        w: if cfg.fixed_anchor_size { 1.0 } else { widths[i] },
                        h: if cfg.fixed_anchor_size { 1.0 } else { heights[i] },
                    });
                }
            }
        }
        layer = last;
    }
    anchors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{FACE_ANCHORS, PALM_ANCHORS};

    #[test]
    fn face_anchor_grid() {
        let a = generate(&FACE_ANCHORS);
        // 16×16×2(stride 8) + 8×8×6(stride 16×3층) = 512 + 384
        assert_eq!(a.len(), 896);
        // 첫 셀: 센터 (0.5/16, 0.5/16), 셀당 2개 (ar 1.0 + interpolated)
        assert!((a[0].x - 0.03125).abs() < 1e-6 && (a[0].y - 0.03125).abs() < 1e-6);
        assert!((a[1].x - 0.03125).abs() < 1e-6);
        // stride 8 그룹 마지막 셀 = (15.5/16, 15.5/16)
        assert!((a[511].x - 0.96875).abs() < 1e-6 && (a[511].y - 0.96875).abs() < 1e-6);
        // stride 16 그룹 첫 셀 = (0.5/8, 0.5/8), 셀당 6개
        for i in 512..518 {
            assert!((a[i].x - 0.0625).abs() < 1e-6 && (a[i].y - 0.0625).abs() < 1e-6);
        }
        assert!(a.iter().all(|an| an.w == 1.0 && an.h == 1.0));
    }

    #[test]
    fn palm_anchor_grid() {
        let a = generate(&PALM_ANCHORS);
        // 24×24×2 + 12×12×6 = 1152 + 864
        assert_eq!(a.len(), 2016);
        assert!((a[0].x - 0.5 / 24.0).abs() < 1e-6);
    }
}
