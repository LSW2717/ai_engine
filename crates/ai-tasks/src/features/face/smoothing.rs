//! OneEuroFilter — 랜드마크 지터 억제 (MediaPipe `LandmarksSmoothingCalculator`의
//! one_euro_filter 모드 등가 구조).
//!
//! 핵심: 저속에서는 강하게(지터 제거), 고속에서는 약하게(랙 제거) 저역통과.
//! MediaPipe는 속도를 **객체 크기로 정규화**한다 — 얼굴이 화면에서 클수록 같은
//! 픽셀 이동이 "느린" 움직임이라 더 강하게 눌린다. 그 규약(value_scale =
//! 1/object_scale)을 따른다.
//!
//! **파라미터 확정 (2026-08-14, MediaPipe face_landmarks_detector_graph.cc 원본
//! 대조)**: VIDEO/스트림 모드 smooth_landmarks = one_euro **min_cutoff 0.05 /
//! beta 80 / derivate_cutoff 1.0** (정지 시 α≈0.01, 고속 시 α≈0.94; num_faces==1
//! 일 때만 적용 — 우리도 lm 1명이라 동일). ⚠ 속도는 **픽셀 좌표계** 기준이다 —
//! LandmarksSmoothingCalculator가 정규화 랜드마크를 이미지 크기로 denormalize한
//! 뒤 필터하므로, 정규화 좌표를 그대로 미분하면 속도가 프레임 폭배만큼 작아져
//! 과잉 스무딩(랙)이 된다. object_scale도 MediaPipe 기본(랜드마크 bbox px의
//! (w+h)/2)을 따른다.

const TAU: f32 = 2.0 * std::f32::consts::PI;

fn smoothing_factor(te: f32, cutoff: f32) -> f32 {
    let r = TAU * cutoff * te;
    r / (r + 1.0)
}

/// 스칼라 하나의 One Euro 필터
#[derive(Clone, Debug)]
pub struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    state: Option<(f32, f32, f64)>, // (x_hat, dx_hat, t_ms)
}

impl OneEuro {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        OneEuro { min_cutoff, beta, d_cutoff, state: None }
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    /// value_scale: 속도 정규화 계수 (MediaPipe: 1/객체크기). 첫 샘플은 그대로.
    pub fn apply(&mut self, t_ms: f64, value_scale: f32, x: f32) -> f32 {
        let Some((x_prev, dx_prev, t_prev)) = self.state else {
            self.state = Some((x, 0.0, t_ms));
            return x;
        };
        let te = ((t_ms - t_prev) / 1e3) as f32;
        if te <= 0.0 {
            return x_prev; // 같은 타임스탬프 재호출 — 상태 불변
        }
        let dx = (x - x_prev) * value_scale / te;
        let dx_hat = dx_prev + smoothing_factor(te, self.d_cutoff) * (dx - dx_prev);
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let x_hat = x_prev + smoothing_factor(te, cutoff) * (x - x_prev);
        self.state = Some((x_hat, dx_hat, t_ms));
        x_hat
    }
}

/// 랜드마크 세트(N×[x,y,z]) 필터 — 좌표축마다 독립 OneEuro
pub struct LandmarkSmoother {
    filters: Vec<[OneEuro; 3]>,
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
}

impl LandmarkSmoother {
    /// face_landmarker VIDEO 확정값 (모듈 헤더 — MediaPipe 원본 대조)
    pub fn face_default() -> Self {
        LandmarkSmoother { filters: Vec::new(), min_cutoff: 0.05, beta: 80.0, d_cutoff: 1.0 }
    }

    pub fn reset(&mut self) {
        self.filters.clear();
    }

    /// pts: 원본 프레임 **정규화** [x,y,z]. img_w/img_h로 픽셀 속도를 계산한다
    /// (MediaPipe는 denormalize 후 필터 — 모듈 헤더 ⚠). object_scale은 랜드마크
    /// bbox px (w+h)/2 를 내부 계산. 랜드마크 수가 바뀌면 리셋.
    pub fn apply(&mut self, t_ms: f64, img_w: f32, img_h: f32, pts: &mut [[f32; 3]]) {
        if self.filters.len() != pts.len() {
            let f = OneEuro::new(self.min_cutoff, self.beta, self.d_cutoff);
            self.filters = vec![[f.clone(), f.clone(), f]; pts.len()];
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts.iter() {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        let object_scale = ((x1 - x0) * img_w + (y1 - y0) * img_h) * 0.5;
        let scale = if object_scale > 0.0 { 1.0 / object_scale } else { 1.0 };
        // 속도는 px 좌표계 — 축별 denormalize 계수 (z는 x축 스케일 관례)
        let dims = [img_w, img_h, img_w];
        for (p, fs) in pts.iter_mut().zip(&mut self.filters) {
            for c in 0..3 {
                p[c] = fs[c].apply(t_ms, dims[c] * scale, p[c]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_passthrough() {
        let mut f = OneEuro::new(1.0, 0.0, 1.0);
        assert_eq!(f.apply(0.0, 1.0, 3.5), 3.5);
    }

    #[test]
    fn jitter_is_damped_but_motion_tracks() {
        // 느린 지터: ±0.01 진동은 눌려야 한다
        let mut f = OneEuro::new(1.0, 0.0, 1.0);
        f.apply(0.0, 1.0, 0.0);
        let mut worst: f32 = 0.0;
        for i in 1..60 {
            let x = if i % 2 == 0 { 0.01 } else { -0.01 };
            worst = worst.max(f.apply(i as f64 * 33.0, 1.0, x).abs());
        }
        assert!(worst < 0.006, "지터 잔존 {worst}");
        // 큰 단차는 beta가 크면 빠르게 따라간다
        let mut fast = OneEuro::new(1.0, 500.0, 1.0);
        fast.apply(0.0, 1.0, 0.0);
        let y = fast.apply(33.0, 1.0, 1.0);
        assert!(y > 0.8, "랙 과다 {y}");
    }
}
