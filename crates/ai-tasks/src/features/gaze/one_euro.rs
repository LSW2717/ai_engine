//! OneEuro (웹 core/one-euro.ts 1:1 — freq 없는 변형: dt는 타임스탬프 차).
//! 상수: minCutoff 1.3 / beta 0.45 / dCutoff 1.0 (config.ts:52).
//! face/smoothing.rs와 별개 유지 — 파라미터화 축이 다르다.

pub struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
    t_prev: f64,
}

fn alpha(dt: f32, fc: f32) -> f32 {
    let r = std::f32::consts::TAU * fc * dt;
    r / (r + 1.0)
}

impl OneEuro {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        OneEuro { min_cutoff, beta, d_cutoff, x_prev: None, dx_prev: 0.0, t_prev: 0.0 }
    }

    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = 0.0;
    }

    pub fn filter(&mut self, x: f32, t_ms: f64) -> f32 {
        let t = t_ms / 1000.0;
        let Some(xp) = self.x_prev else {
            self.x_prev = Some(x);
            self.t_prev = t;
            self.dx_prev = 0.0;
            return x;
        };
        let dt = ((t - self.t_prev) as f32).max(1e-3);
        self.t_prev = t;
        let dx = (x - xp) / dt;
        let ad = alpha(dt, self.d_cutoff);
        let dx_hat = ad * dx + (1.0 - ad) * self.dx_prev;
        let fc = self.min_cutoff + self.beta * dx_hat.abs();
        let a = alpha(dt, fc);
        let x_hat = a * x + (1.0 - a) * xp;
        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        x_hat
    }
}

/// yaw/pitch 독립 2축 (웹 OneEuro2)
pub struct OneEuro2 {
    pub yaw: OneEuro,
    pub pitch: OneEuro,
}

impl Default for OneEuro2 {
    fn default() -> Self {
        OneEuro2 { yaw: OneEuro::new(1.3, 0.45, 1.0), pitch: OneEuro::new(1.3, 0.45, 1.0) }
    }
}

impl OneEuro2 {
    pub fn reset(&mut self) {
        self.yaw.reset();
        self.pitch.reset();
    }
}
