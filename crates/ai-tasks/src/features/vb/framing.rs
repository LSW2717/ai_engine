//! 인물 중앙 프레이밍 — 목표 계산 + 스무딩 (v-ai `_updateFraming` 1:1 이식,
//! INTEGRATION.md 계약 상수: 데드밴드 0.045/0.055 · 2s 지속이탈 커밋 · EMA 0.5s ·
//! 슬루 0.35/s).
//!
//! 순수 로직 — GPU·플랫폼 무관 (웹 wasm·모바일 ffi가 같은 코드). bbox는
//! 마스크 해상도 GPU 리덕션(stages/bbox.rs)의 비동기 리드백이 공급한다.
//!
//! 동작 원칙 ("촐싹거림 방지" UX 핵심 — 웹 주석 그대로):
//! - 매 프레임 목표를 쫓지 않는다. 순간 목표가 현재 크롭에서 데드밴드 이상 벗어난
//!   상태로 2초 유지될 때만 goal로 커밋 → 거기까지 한 번 활강 후 정지.
//! - 인물 소실: 2초 홀드(잠깐 놓친 것일 수 있음) 후 1x 복귀. 기능 off → 1x 수렴.
//! - 1x 복귀는 커밋 지연 없이 즉시 활강 목표.

/// 프레이밍 설정 (EffectsPatch `framing` — 웹 DEFAULT_VB_FRAMING_OPTIONS 기본값)
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FramingOptions {
    pub enabled: bool,
    pub zoom_max: f32,
    pub headroom: f32,
}

impl Default for FramingOptions {
    fn default() -> Self {
        FramingOptions { enabled: true, zoom_max: 1.7, headroom: 0.15 }
    }
}

/// 정규화 인물 bbox — [x0, x1, y0, y1] (y0 = 화면 위쪽)
pub type BBox = [f32; 4];

const COMMIT_MS: f64 = 2000.0;
const DEAD_CENTER: f32 = 0.045;
const DEAD_SCALE: f32 = 0.055;
const HOLD_MS: f64 = 2000.0;

#[derive(Clone, Copy, Debug)]
struct Goal {
    scale: f32,
    cx: f32,
    cy: f32,
}

pub struct Framing {
    scale: f32,
    cx: f32,
    cy: f32,
    last_seen_ms: f64,
    last_ms: f64,
    goal: Option<Goal>,
    pending_since_ms: f64,
}

impl Default for Framing {
    fn default() -> Self {
        Framing {
            scale: 1.0,
            cx: 0.5,
            cy: 0.5,
            last_seen_ms: 0.0,
            last_ms: 0.0,
            goal: None,
            pending_since_ms: 0.0,
        }
    }
}

impl Framing {
    pub fn reset(&mut self) {
        *self = Framing::default();
    }

    /// 현재 크롭 (scale, cx, cy) — scale 1 = 크롭 없음. compose uniform이 소비.
    pub fn current(&self) -> (f32, f32, f32) {
        (self.scale, self.cx, self.cy)
    }

    /// 크롭이 실제로 걸려 있나 (v-ai framingActive: scale < 0.999)
    pub fn active(&self) -> bool {
        self.scale < 0.999
    }

    /// 매 렌더 틱 호출. opts=None이면 기능 off. bbox=최신 스캔 결과
    /// (리드백이 아직이면 직전 값 유지 공급 — v-ai 스냅샷 재스캔 등가, None=인물 없음).
    pub fn update(&mut self, opts: Option<&FramingOptions>, bbox: Option<BBox>, ts_ms: f64) {
        let active = opts.map(|o| o.enabled).unwrap_or(false);
        let dt =
            if self.last_ms > 0.0 { ((ts_ms - self.last_ms) / 1000.0).min(0.1) } else { 0.033 }
                as f32;
        self.last_ms = ts_ms;

        let target: Goal;
        if active {
            let f = opts.unwrap();
            if let Some(b) = bbox {
                self.last_seen_ms = ts_ms;
                let zoom_max = f.zoom_max.clamp(1.0, 3.0);
                let headroom = f.headroom.max(0.0);
                let [x0, x1, y0, y1] = b;
                let person_h = y1 - y0;
                let top = (y0 - person_h * headroom).max(0.0);
                let bottom = (y1 + person_h * 0.05).min(1.0);
                let crop =
                    (bottom - top).max((x1 - x0) * 1.15).max(1.0 / zoom_max).min(1.0);
                let half = crop / 2.0;
                let cx = ((x0 + x1) / 2.0).clamp(half, 1.0 - half);
                let cy = (top + half).clamp(half, 1.0 - half);
                target = Goal { scale: crop, cx, cy };
            } else if self.last_seen_ms > 0.0 && ts_ms - self.last_seen_ms < HOLD_MS {
                return; // 홀드 — 현재 크롭 유지
            } else {
                target = Goal { scale: 1.0, cx: 0.5, cy: 0.5 }; // 인물 소실 — 복귀
            }
        } else {
            if self.scale >= 0.999 {
                return;
            }
            target = Goal { scale: 1.0, cx: 0.5, cy: 0.5 }; // 기능 off — 1x 수렴
        }

        // 지속 이탈 커밋
        let identity_target = target.scale >= 0.999;
        if identity_target {
            self.goal = Some(Goal { scale: 1.0, cx: 0.5, cy: 0.5 });
            self.pending_since_ms = 0.0;
        } else if self.goal.is_none() {
            let dc = (target.cx - self.cx).hypot(target.cy - self.cy);
            let ds = (target.scale - self.scale).abs();
            if dc > DEAD_CENTER || ds > DEAD_SCALE {
                if self.pending_since_ms == 0.0 {
                    self.pending_since_ms = ts_ms;
                }
                if ts_ms - self.pending_since_ms >= COMMIT_MS {
                    self.goal = Some(target); // 커밋 — 이동 중 갱신 안 함
                    self.pending_since_ms = 0.0;
                }
            } else {
                self.pending_since_ms = 0.0;
            }
        }

        let Some(goal) = self.goal else { return };

        // 활강 — 시간보정 EMA(시정수 ~0.5s) + 슬루 제한(초당 프레임 폭 35%)
        let alpha = 1.0 - (-dt * 2.0).exp();
        let max_step = 0.35 * dt;
        let step = |cur: f32, tgt: f32| cur + ((tgt - cur) * alpha).clamp(-max_step, max_step);
        self.cx = step(self.cx, goal.cx);
        self.cy = step(self.cy, goal.cy);
        self.scale = step(self.scale, goal.scale);

        let arrived = (goal.cx - self.cx).hypot(goal.cy - self.cy) < 0.004
            && (goal.scale - self.scale).abs() < 0.004;
        if arrived {
            if goal.scale >= 0.999 {
                self.scale = 1.0;
                self.cx = 0.5;
                self.cy = 0.5;
            }
            self.goal = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTS: FramingOptions =
        FramingOptions { enabled: true, zoom_max: 1.7, headroom: 0.15 };
    // 중앙 인물 bbox — crop = max(0.55·1.15, 세로항, 1/1.7) 계산 결과가 1 미만
    const BB: BBox = [0.3, 0.62, 0.2, 0.75];

    /// 30fps로 ms 동안 틱을 돌린다
    fn run(f: &mut Framing, opts: Option<&FramingOptions>, bbox: Option<BBox>, t0: f64, ms: f64) -> f64 {
        let mut t = t0;
        while t < t0 + ms {
            f.update(opts, bbox, t);
            t += 33.0;
        }
        t
    }

    #[test]
    fn deadband_ignores_small_motion() {
        let mut f = Framing::default();
        // 데드밴드 안쪽 목표(현재 1x에서 scale 이탈 0.05 < 0.055, 중심 이탈 작음)는 영원히 무시
        let bb: BBox = [0.03, 0.98, 0.0, 1.0]; // crop ≈ max(1.09,…)→1 → identity-ish
        run(&mut f, Some(&OPTS), Some(bb), 1000.0, 5000.0);
        assert_eq!(f.current().0, 1.0, "데드밴드 안 목표에 반응하면 안 됨");
    }

    #[test]
    fn sustained_deviation_commits_after_2s_then_glides() {
        let mut f = Framing::default();
        let t = run(&mut f, Some(&OPTS), Some(BB), 1000.0, 1900.0);
        assert_eq!(f.current().0, 1.0, "커밋 전(2s 미만)엔 정지");
        run(&mut f, Some(&OPTS), Some(BB), t, 4000.0);
        let (s, cx, _cy) = f.current();
        assert!(s < 0.75, "커밋 후 활강해야 함: scale={s}");
        assert!((cx - 0.46).abs() < 0.02, "중심 수렴: cx={cx}");
    }

    #[test]
    fn slew_limits_speed() {
        let mut f = Framing::default();
        // run()은 마지막 실행 틱 +33ms를 반환 — t에서 update하면 dt=33ms
        let t = run(&mut f, Some(&OPTS), Some(BB), 1000.0, 2100.0); // 커밋 직후
        let s0 = f.current().0;
        f.update(Some(&OPTS), Some(BB), t);
        let s1 = f.current().0;
        // 한 틱(33ms) 이동량 ≤ 슬루 0.35/s × 0.033s
        assert!((s1 - s0).abs() <= 0.35 * 0.0331 + 1e-6, "슬루 위반: {}", (s1 - s0).abs());
    }

    #[test]
    fn person_lost_holds_2s_then_returns() {
        let mut f = Framing::default();
        let t = run(&mut f, Some(&OPTS), Some(BB), 1000.0, 6000.0);
        let s_zoom = f.current().0;
        assert!(s_zoom < 0.75);
        // 소실 1.5s — 홀드 (변화 없음)
        let t = run(&mut f, Some(&OPTS), None, t, 1500.0);
        assert_eq!(f.current().0, s_zoom, "2s 미만 소실엔 홀드");
        // 소실 지속 — 즉시(커밋 지연 없이) 1x 복귀 활강
        run(&mut f, Some(&OPTS), None, t, 6000.0);
        assert_eq!(f.current().0, 1.0, "소실 지속 시 1x 복귀");
    }

    #[test]
    fn disable_converges_to_identity() {
        let mut f = Framing::default();
        let t = run(&mut f, Some(&OPTS), Some(BB), 1000.0, 6000.0);
        assert!(f.current().0 < 0.75);
        run(&mut f, None, None, t, 6000.0);
        let (s, cx, cy) = f.current();
        assert_eq!((s, cx, cy), (1.0, 0.5, 0.5), "off면 항등으로 수렴");
    }
}
