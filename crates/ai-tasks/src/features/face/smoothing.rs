//! OneEuroFilter — 랜드마크 지터 억제 (MediaPipe `LandmarksSmoothingCalculator`의
//! one_euro_filter 모드 등가 구조).
//!
//! 핵심: 저속에서는 강하게(지터 제거), 고속에서는 약하게(랙 제거) 저역통과.
//! MediaPipe는 속도를 **객체 크기로 정규화**한다 — 얼굴이 화면에서 클수록 같은
//! 픽셀 이동이 "느린" 움직임이라 더 강하게 눌린다. 그 규약(value_scale =
//! 1/object_scale)을 따른다.
//!
//! ⚠ 기본 파라미터는 MediaPipe face_landmarker의 것으로 추정한 값 — **VIDEO 모드
//! 파리티 게이트로 검증 전까지는 미확정**이다 (단일 프레임 게이트에는 영향 없음:
//! 첫 샘플은 그대로 통과한다).

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
    /// face_landmarker 추정 기본값 (미확정 — 모듈 헤더 참조)
    pub fn face_default() -> Self {
        LandmarkSmoother { filters: Vec::new(), min_cutoff: 0.1, beta: 40.0, d_cutoff: 1.0 }
    }

    pub fn reset(&mut self) {
        self.filters.clear();
    }

    /// object_scale: ROI 크기(절대 px, (w+h)/2). 랜드마크 수가 바뀌면 리셋.
    pub fn apply(&mut self, t_ms: f64, object_scale: f32, pts: &mut [[f32; 3]]) {
        if self.filters.len() != pts.len() {
            let f = OneEuro::new(self.min_cutoff, self.beta, self.d_cutoff);
            self.filters = vec![[f.clone(), f.clone(), f]; pts.len()];
        }
        let scale = if object_scale > 0.0 { 1.0 / object_scale } else { 1.0 };
        for (p, fs) in pts.iter_mut().zip(&mut self.filters) {
            for c in 0..3 {
                p[c] = fs[c].apply(t_ms, scale, p[c]);
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
