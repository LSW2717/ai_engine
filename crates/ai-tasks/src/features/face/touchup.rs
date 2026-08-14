//! 터치업(내 모습 보정) 피부 마스크 래스터라이즈 — vcxrust_ai face/touchup.rs 이식.
//!
//! 웹 drawTouchUp의 클립 기하: 얼굴 오벌에서 눈썹/눈/입 구멍을 뺀 영역(even-odd).
//! GPU로 폴리곤을 넘기는 대신 CPU에서 저해상(128×128) 알파 마스크로 굽고,
//! compose 셰이더가 이 마스크 가중으로 소프트 블러를 섞는다. 프레임당 ~수십 µs.
//!
//! 좌표 규약: 입력 랜드마크는 **프레임 절대 px** (FaceResult.points × 프레임 크기).

use serde::Deserialize;

/// 내 모습 보정(터치업) — 웹 VBOptions.touchUp · vcxrust_ai TouchUpOptions와
/// 동일 스키마 (EffectsPatch `touchUp` 필드)
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TouchUpOptions {
    pub enabled: bool,
    /// 0..1
    pub strength: f32,
}

/// MediaPipe FaceMesh 표준 인덱스 — 웹 face-effects-2d.ts와 동일.
const FACE_OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const LIPS_OUTER: [usize; 20] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];
const BROW_RIGHT: [usize; 5] = [70, 63, 105, 66, 107];
const BROW_LEFT: [usize; 5] = [336, 296, 334, 293, 300];
const LASH_RIGHT: [usize; 9] = [33, 246, 161, 160, 159, 158, 157, 173, 133];
const LASH_LEFT: [usize; 9] = [263, 466, 388, 387, 386, 385, 384, 398, 362];

pub const MASK_SIZE: usize = 128;

/// 프레임 픽셀 좌표계 bbox + 128×128 알파 마스크.
pub struct TouchUpMask {
    pub data: Vec<u8>,
    pub x0: f32,
    pub y0: f32,
    pub w: f32,
    pub h: f32,
    /// 광대 폭(px) — 셰이더 블러 반경 산출용 (웹 faceW와 동일: lm234↔454 거리).
    pub face_w: f32,
}

/// 468+ 랜드마크(프레임 px) → 터치업 마스크. 퇴화 얼굴이면 None.
pub fn rasterize(points: &[[f32; 3]]) -> Option<TouchUpMask> {
    if points.len() < 468 {
        return None;
    }
    let pt = |i: usize| [points[i][0], points[i][1]];
    let face_w = {
        let a = pt(234);
        let b = pt(454);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    };
    if face_w < 16.0 {
        return None;
    }

    // bbox — 오벌 기준 8% 여유
    let oval: Vec<[f32; 2]> = FACE_OVAL.iter().map(|&i| pt(i)).collect();
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for p in &oval {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let pad_x = (max_x - min_x) * 0.08;
    let pad_y = (max_y - min_y) * 0.08;
    let x0 = min_x - pad_x;
    let y0 = min_y - pad_y;
    let w = (max_x - min_x) + pad_x * 2.0;
    let h = (max_y - min_y) + pad_y * 2.0;
    if w < 8.0 || h < 8.0 {
        return None;
    }

    // even-odd 폴리곤 집합: 오벌(채움) + 구멍들(패리티 반전).
    // 구멍 타원은 웹 holeEllipse와 동일 파라미터: 중심=점집합 평균, 반폭=점집합
    // 폭/2×wScale, 반높이=faceW×hFrac. 20각형으로 폴리곤화.
    let mut polys: Vec<Vec<[f32; 2]>> = vec![oval];
    polys.push(LIPS_OUTER.iter().map(|&i| pt(i)).collect());
    for (idxs, h_frac, w_scale) in [
        (&LASH_RIGHT[..], 0.075f32, 1.25f32),
        (&LASH_LEFT[..], 0.075, 1.25),
        (&BROW_RIGHT[..], 0.05, 1.2),
        (&BROW_LEFT[..], 0.05, 1.2),
    ] {
        let pts: Vec<[f32; 2]> = idxs.iter().map(|&i| pt(i)).collect();
        let n = pts.len() as f32;
        let cx = pts.iter().map(|p| p[0]).sum::<f32>() / n;
        let cy = pts.iter().map(|p| p[1]).sum::<f32>() / n;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in &pts {
            lo = lo.min(p[0]);
            hi = hi.max(p[0]);
        }
        let rx = (hi - lo) / 2.0 * w_scale;
        let ry = face_w * h_frac;
        let mut ellipse = Vec::with_capacity(20);
        for k in 0..20 {
            let a = k as f32 / 20.0 * std::f32::consts::TAU;
            ellipse.push([cx + rx * a.cos(), cy + ry * a.sin()]);
        }
        polys.push(ellipse);
    }

    // 마스크 그리드 좌표로 변환
    let sx = MASK_SIZE as f32 / w;
    let sy = MASK_SIZE as f32 / h;
    for poly in polys.iter_mut() {
        for p in poly.iter_mut() {
            p[0] = (p[0] - x0) * sx;
            p[1] = (p[1] - y0) * sy;
        }
    }

    // 스캔라인 even-odd 채움
    let mut mask = vec![0u8; MASK_SIZE * MASK_SIZE];
    let mut xs: Vec<f32> = Vec::with_capacity(32);
    for row in 0..MASK_SIZE {
        let yc = row as f32 + 0.5;
        xs.clear();
        for poly in &polys {
            let n = poly.len();
            for i in 0..n {
                let a = poly[i];
                let b = poly[(i + 1) % n];
                if (a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc) {
                    xs.push(a[0] + (yc - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
                }
            }
        }
        xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i + 1 < xs.len() {
            let from = xs[i].max(0.0).round() as usize;
            let to = (xs[i + 1].max(0.0).round() as usize).min(MASK_SIZE);
            for x in from..to {
                mask[row * MASK_SIZE + x] = 255;
            }
            i += 2;
        }
    }

    // 소프트 엣지 — 3×3 박스 블러 2회
    for _ in 0..2 {
        let src = mask.clone();
        for y in 0..MASK_SIZE {
            for x in 0..MASK_SIZE {
                let mut acc = 0u32;
                let mut cnt = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let yy = y as i32 + dy;
                        let xx = x as i32 + dx;
                        if yy >= 0 && yy < MASK_SIZE as i32 && xx >= 0 && xx < MASK_SIZE as i32 {
                            acc += src[yy as usize * MASK_SIZE + xx as usize] as u32;
                            cnt += 1;
                        }
                    }
                }
                mask[y * MASK_SIZE + x] = (acc / cnt.max(1)) as u8;
            }
        }
    }

    Some(TouchUpMask { data: mask, x0, y0, w, h, face_w })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 합성 얼굴: 오벌 인덱스를 원 위에, 구멍 요소들을 안쪽에 배치.
    fn synthetic_points() -> Vec<[f32; 3]> {
        let mut pts = vec![[0.0f32; 3]; 468];
        let (cx, cy, r) = (200.0f32, 200.0f32, 100.0f32);
        for (k, &i) in FACE_OVAL.iter().enumerate() {
            let a = k as f32 / FACE_OVAL.len() as f32 * std::f32::consts::TAU;
            pts[i] = [cx + r * a.cos(), cy + r * a.sin(), 0.0];
        }
        // 광대 폭 기준점
        pts[234] = [cx - r, cy, 0.0];
        pts[454] = [cx + r, cy, 0.0];
        // 입/눈/눈썹 — 안쪽 작은 클러스터
        for &i in LIPS_OUTER.iter().take(10) {
            pts[i] = [cx - 20.0, cy + 50.0, 0.0];
        }
        for &i in LIPS_OUTER.iter().skip(10) {
            pts[i] = [cx + 20.0, cy + 55.0, 0.0];
        }
        for &i in LASH_RIGHT.iter() {
            pts[i] = [cx - 40.0, cy - 20.0, 0.0];
        }
        for &i in LASH_LEFT.iter() {
            pts[i] = [cx + 40.0, cy - 20.0, 0.0];
        }
        for &i in BROW_RIGHT.iter() {
            pts[i] = [cx - 40.0, cy - 45.0, 0.0];
        }
        for &i in BROW_LEFT.iter() {
            pts[i] = [cx + 40.0, cy - 45.0, 0.0];
        }
        pts
    }

    #[test]
    fn rasterize_fills_skin_and_carves_holes() {
        let m = rasterize(&synthetic_points()).expect("mask");
        let at = |fx: f32, fy: f32| {
            let gx = ((fx - m.x0) / m.w * MASK_SIZE as f32) as usize;
            let gy = ((fy - m.y0) / m.h * MASK_SIZE as f32) as usize;
            m.data[gy.min(MASK_SIZE - 1) * MASK_SIZE + gx.min(MASK_SIZE - 1)]
        };
        assert!(at(200.0, 160.0) > 200, "이마/코 영역은 채워져야");
        assert!(at(90.0, 90.0) < 30, "오벌 밖 코너는 비어야");
        assert!(m.face_w > 190.0);
    }

    #[test]
    fn rasterize_rejects_degenerate() {
        let pts = vec![[0.0f32; 3]; 468];
        assert!(rasterize(&pts).is_none());
    }
}
