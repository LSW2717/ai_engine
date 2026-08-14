//! 메이크업 컬러 오버레이 래스터라이즈 — vcxrust_ai face/makeup.rs 이식.
//!
//! 웹 drawMakeup의 핵심 3요소: 립 틴트, 블러셔, 아이섀도. 각 요소를 CPU에서
//! 128×128 RGBA 오버레이(색+알파)로 굽고, compose 셰이더가 2단으로 블렌드한다:
//! multiply 레이어 = base×mix(1,color,α), source-over 레이어 = mix(base,color,α).
//! gloss/래시/눈썹은 v1 생략(효과 미미) — 필요 시 후속.
//!
//! bbox/좌표 규약은 touchup과 동일(프레임 px). alpha는 소스오버로 누적.

use serde::Deserialize;

use super::touchup::MASK_SIZE;
use crate::features::vb::params::EffectsState;

/// 메이크업 룩 — 웹 VBOptions.makeup · vcxrust_ai MakeupOptions와 동일 스키마
/// (EffectsPatch `makeup` 필드)
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MakeupOptions {
    pub enabled: bool,
    /// 0..1 — 전체 강도(룩 알파에 곱함)
    pub intensity: f32,
    pub lip: MakeupTint,
    pub blush: MakeupBlush,
    pub shadow: MakeupTint,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MakeupTint {
    /// "#rrggbb"
    pub color: String,
    pub alpha: f32,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MakeupBlush {
    pub color: String,
    pub alpha: f32,
    /// 얼굴 폭(face_w) 배수 반경
    pub size: f32,
}

// 웹 face-effects-2d.ts와 동일 인덱스
const LIPS_OUTER: [usize; 20] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];
const LIPS_INNER: [usize; 20] = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
];
const LASH_RIGHT: [usize; 9] = [33, 246, 161, 160, 159, 158, 157, 173, 133];
const LASH_LEFT: [usize; 9] = [263, 466, 388, 387, 386, 385, 384, 398, 362];
/// 볼 중심(블러셔) — 웹과 동일.
const CHEEKS: [usize; 2] = [205, 425];

pub struct MakeupOverlay {
    /// multiply 레이어 RGBA8 128² (아이섀도·립 본체)
    pub mul: Vec<u8>,
    /// source-over 레이어 RGBA8 128² (블러셔·립 22% 보충)
    pub over: Vec<u8>,
    pub x0: f32,
    pub y0: f32,
    pub w: f32,
    pub h: f32,
}

fn hex_rgb(hex: &str) -> [f32; 3] {
    EffectsState::hex_rgb(hex).map(|c| c * 255.0)
}

/// 소스오버 컴포짓 한 픽셀 (스트레이트 알파).
fn blend(dst: &mut [u8], idx: usize, rgb: [f32; 3], a: f32) {
    let a = a.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let da = dst[idx * 4 + 3] as f32 / 255.0;
    let out_a = a + da * (1.0 - a);
    if out_a <= 1e-4 {
        return;
    }
    for k in 0..3 {
        let dc = dst[idx * 4 + k] as f32;
        let sc = rgb[k];
        let oc = (sc * a + dc * da * (1.0 - a)) / out_a;
        dst[idx * 4 + k] = oc.clamp(0.0, 255.0) as u8;
    }
    dst[idx * 4 + 3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

/// 468+ 랜드마크(프레임 px) + 룩 → 오버레이. 퇴화/비활성이면 None.
pub fn rasterize(points: &[[f32; 3]], mk: &MakeupOptions) -> Option<MakeupOverlay> {
    if points.len() < 468 || !mk.enabled || mk.intensity <= 0.0 {
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

    // bbox — 립·볼·눈 전부 감싸도록 전체 랜드마크 bbox + 10% 여유
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for p in points.iter() {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let pad_x = (max_x - min_x) * 0.1;
    let pad_y = (max_y - min_y) * 0.1;
    let x0 = min_x - pad_x;
    let y0 = min_y - pad_y;
    let w = (max_x - min_x) + pad_x * 2.0;
    let h = (max_y - min_y) + pad_y * 2.0;
    if w < 8.0 || h < 8.0 {
        return None;
    }

    let sx = MASK_SIZE as f32 / w;
    let sy = MASK_SIZE as f32 / h;
    let to_grid = |p: [f32; 2]| [(p[0] - x0) * sx, (p[1] - y0) * sy];
    let k = mk.intensity.clamp(0.0, 2.0);

    let mut mul = vec![0u8; MASK_SIZE * MASK_SIZE * 4];
    let mut over = vec![0u8; MASK_SIZE * MASK_SIZE * 4];

    // ── 아이섀도 — multiply, alpha = shadow.alpha × k (웹과 동일 수치) ──
    if mk.shadow.alpha > 0.0 {
        let color = hex_rgb(&mk.shadow.color);
        let a = mk.shadow.alpha * k;
        for lash in [&LASH_RIGHT[..], &LASH_LEFT[..]] {
            let pts: Vec<[f32; 2]> = lash.iter().map(|&i| to_grid(pt(i))).collect();
            // 라인 + 위로 lift만큼 볼록한 캡을 닫아 영역 형성
            let lift = face_w * 0.05 * sy;
            let first = pts[0];
            let last = *pts.last().unwrap();
            let mut region = pts.clone();
            // 위쪽 제어점 근사: 중점을 lift만큼 올린 점을 추가
            region.push([(first[0] + last[0]) / 2.0, first[1].min(last[1]) - lift * 1.8]);
            fill_polygon(&mut mul, &region, color, a);
        }
    }

    // ── 립 — multiply(본체) + source-over 22% 보충 (웹 drawMakeup 동일 구성) ──
    if mk.lip.alpha > 0.0 {
        let color = hex_rgb(&mk.lip.color);
        let outer: Vec<[f32; 2]> = LIPS_OUTER.iter().map(|&i| to_grid(pt(i))).collect();
        let inner: Vec<[f32; 2]> = LIPS_INNER.iter().map(|&i| to_grid(pt(i))).collect();
        fill_polygon_evenodd(&mut mul, &[outer.clone(), inner.clone()], color, mk.lip.alpha * k);
        fill_polygon_evenodd(&mut over, &[outer, inner], color, mk.lip.alpha * 0.22 * k);
    }

    // ── 블러셔 — source-over radial (웹 falloff: 0→α, 0.6→0.5α, 1→0) ──
    if mk.blush.alpha > 0.0 {
        let color = hex_rgb(&mk.blush.color);
        for &ci in CHEEKS.iter() {
            let c = to_grid(pt(ci));
            let r = face_w * mk.blush.size;
            fill_radial(&mut over, c, r * sx, r * sy, color, mk.blush.alpha * k);
        }
    }

    // 소프트 엣지 — premultiplied로 RGBA 전체 블러 (다크 헤일로 방지)
    feather_premultiplied(&mut mul);
    feather_premultiplied(&mut over);

    Some(MakeupOverlay { mul, over, x0, y0, w, h })
}

/// 단순 다각형 채움(non-zero 근사: 홀수 교차) + 소스오버.
fn fill_polygon(ov: &mut [u8], poly: &[[f32; 2]], rgb: [f32; 3], a: f32) {
    fill_polygon_evenodd(ov, std::slice::from_ref(&poly.to_vec()), rgb, a);
}

/// even-odd 다각형 집합 채움 + 소스오버.
fn fill_polygon_evenodd(ov: &mut [u8], polys: &[Vec<[f32; 2]>], rgb: [f32; 3], a: f32) {
    if a <= 0.0 {
        return;
    }
    let mut xs: Vec<f32> = Vec::with_capacity(16);
    for row in 0..MASK_SIZE {
        let yc = row as f32 + 0.5;
        xs.clear();
        for poly in polys {
            let n = poly.len();
            for i in 0..n {
                let p = poly[i];
                let q = poly[(i + 1) % n];
                if (p[1] <= yc && q[1] > yc) || (q[1] <= yc && p[1] > yc) {
                    xs.push(p[0] + (yc - p[1]) / (q[1] - p[1]) * (q[0] - p[0]));
                }
            }
        }
        xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i + 1 < xs.len() {
            let from = xs[i].max(0.0).round() as usize;
            let to = (xs[i + 1].max(0.0).round() as usize).min(MASK_SIZE);
            for x in from..to {
                blend(ov, row * MASK_SIZE + x, rgb, a);
            }
            i += 2;
        }
    }
}

/// radial 그라디언트 타원 채움 (중심 a → 가장자리 0), 소스오버.
fn fill_radial(ov: &mut [u8], center: [f32; 2], rx: f32, ry: f32, rgb: [f32; 3], a: f32) {
    if rx <= 0.5 || ry <= 0.5 || a <= 0.0 {
        return;
    }
    let ri_x = rx.ceil() as i32;
    let ri_y = ry.ceil() as i32;
    let cx = center[0];
    let cy = center[1];
    for dy in -ri_y..=ri_y {
        let y = cy as i32 + dy;
        if y < 0 || y >= MASK_SIZE as i32 {
            continue;
        }
        for dx in -ri_x..=ri_x {
            let x = cx as i32 + dx;
            if x < 0 || x >= MASK_SIZE as i32 {
                continue;
            }
            // 정규화 타원 거리
            let nx = ((x as f32 + 0.5) - cx) / rx;
            let ny = ((y as f32 + 0.5) - cy) / ry;
            let r01 = (nx * nx + ny * ny).sqrt().clamp(0.0, 1.0);
            // 웹 radial gradient 동일: 0→1.0, 0.6→0.5, 1→0
            let fall = if r01 <= 0.6 {
                1.0 - (r01 / 0.6) * 0.5
            } else {
                0.5 * (1.0 - (r01 - 0.6) / 0.4)
            };
            blend(ov, y as usize * MASK_SIZE + x as usize, rgb, a * fall);
        }
    }
}

/// premultiplied 공간에서 RGBA 전체 3×3 박스블러 2회 → unpremultiply.
/// 색과 알파가 같이 번져야 가장자리에 이물 색(검정)이 안 낀다.
fn feather_premultiplied(ov: &mut [u8]) {
    let n = MASK_SIZE * MASK_SIZE;
    // premultiply (f32 작업 버퍼)
    let mut pm = vec![0.0f32; n * 4];
    for i in 0..n {
        let a = ov[i * 4 + 3] as f32 / 255.0;
        pm[i * 4] = ov[i * 4] as f32 * a;
        pm[i * 4 + 1] = ov[i * 4 + 1] as f32 * a;
        pm[i * 4 + 2] = ov[i * 4 + 2] as f32 * a;
        pm[i * 4 + 3] = a * 255.0;
    }
    for _ in 0..2 {
        let src = pm.clone();
        for y in 0..MASK_SIZE {
            for x in 0..MASK_SIZE {
                let mut acc = [0.0f32; 4];
                let mut cnt = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let yy = y as i32 + dy;
                        let xx = x as i32 + dx;
                        if yy >= 0 && yy < MASK_SIZE as i32 && xx >= 0 && xx < MASK_SIZE as i32 {
                            let s = (yy as usize * MASK_SIZE + xx as usize) * 4;
                            for kch in 0..4 {
                                acc[kch] += src[s + kch];
                            }
                            cnt += 1.0;
                        }
                    }
                }
                let d = (y * MASK_SIZE + x) * 4;
                for kch in 0..4 {
                    pm[d + kch] = acc[kch] / cnt.max(1.0);
                }
            }
        }
    }
    // unpremultiply
    for i in 0..n {
        let a = pm[i * 4 + 3] / 255.0;
        if a > 1e-4 {
            ov[i * 4] = (pm[i * 4] / a).clamp(0.0, 255.0) as u8;
            ov[i * 4 + 1] = (pm[i * 4 + 1] / a).clamp(0.0, 255.0) as u8;
            ov[i * 4 + 2] = (pm[i * 4 + 2] / a).clamp(0.0, 255.0) as u8;
        } else {
            ov[i * 4] = 0;
            ov[i * 4 + 1] = 0;
            ov[i * 4 + 2] = 0;
        }
        ov[i * 4 + 3] = pm[i * 4 + 3].clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look() -> MakeupOptions {
        MakeupOptions {
            enabled: true,
            intensity: 1.0,
            lip: MakeupTint { color: "#d98f95".into(), alpha: 0.45 },
            blush: MakeupBlush { color: "#edaab2".into(), alpha: 0.18, size: 0.23 },
            shadow: MakeupTint { color: "#b98d84".into(), alpha: 0.16 },
        }
    }

    fn synthetic_points() -> Vec<[f32; 3]> {
        let mut pts = vec![[0.0f32; 3]; 468];
        let (cx, cy) = (200.0f32, 200.0f32);
        pts[234] = [cx - 100.0, cy, 0.0];
        pts[454] = [cx + 100.0, cy, 0.0];
        for (k, &i) in LIPS_OUTER.iter().enumerate() {
            let a = k as f32 / 20.0 * std::f32::consts::TAU;
            pts[i] = [cx + 25.0 * a.cos(), cy + 55.0 + 12.0 * a.sin(), 0.0];
        }
        for (k, &i) in LIPS_INNER.iter().enumerate() {
            let a = k as f32 / 20.0 * std::f32::consts::TAU;
            pts[i] = [cx + 14.0 * a.cos(), cy + 55.0 + 6.0 * a.sin(), 0.0];
        }
        for (k, &i) in LASH_RIGHT.iter().enumerate() {
            pts[i] = [cx - 55.0 + k as f32 * 4.0, cy - 25.0, 0.0];
        }
        for (k, &i) in LASH_LEFT.iter().enumerate() {
            pts[i] = [cx + 20.0 + k as f32 * 4.0, cy - 25.0, 0.0];
        }
        pts[205] = [cx - 55.0, cy + 20.0, 0.0];
        pts[425] = [cx + 55.0, cy + 20.0, 0.0];
        // bbox 확장용 극점
        pts[10] = [cx, cy - 110.0, 0.0];
        pts[152] = [cx, cy + 110.0, 0.0];
        pts
    }

    #[test]
    fn makeup_tints_lips() {
        let ov = rasterize(&synthetic_points(), &look()).expect("overlay");
        let sample = |fx: f32, fy: f32| {
            let gx = ((fx - ov.x0) / ov.w * MASK_SIZE as f32) as usize;
            let gy = ((fy - ov.y0) / ov.h * MASK_SIZE as f32) as usize;
            let idx = (gy.min(MASK_SIZE - 1) * MASK_SIZE + gx.min(MASK_SIZE - 1)) * 4;
            [ov.mul[idx], ov.mul[idx + 1], ov.mul[idx + 2], ov.mul[idx + 3]]
        };
        // 립 영역(입술 중앙 바로 위, outer-inner 사이) 알파 존재
        let lip = sample(200.0 + 20.0, 255.0);
        assert!(lip[3] > 20, "lip band should have alpha, got {}", lip[3]);
        // 얼굴 밖 코너는 비어야
        let corner = sample(200.0, 320.0);
        assert!(corner[3] < 40, "outside face should be near-empty");
    }

    // 다크 헤일로 회귀 가드 — 알파가 번진 가장자리 픽셀에 검정 RGB가 남으면
    // multiply 블렌드가 어두운 테두리를 만든다. premultiplied feather 후엔 없어야.
    #[test]
    fn feather_keeps_color_where_alpha_exists() {
        let ov = rasterize(&synthetic_points(), &look()).expect("overlay");
        for i in 0..MASK_SIZE * MASK_SIZE {
            let a = ov.mul[i * 4 + 3];
            if a > 12 {
                let rgb_sum =
                    ov.mul[i * 4] as u32 + ov.mul[i * 4 + 1] as u32 + ov.mul[i * 4 + 2] as u32;
                assert!(rgb_sum > 60, "alpha {a} at px {i} with near-black rgb (sum {rgb_sum})");
            }
        }
    }

    #[test]
    fn makeup_disabled_returns_none() {
        let mut mk = look();
        mk.enabled = false;
        assert!(rasterize(&synthetic_points(), &mk).is_none());
    }
}
