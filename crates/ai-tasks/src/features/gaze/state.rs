//! 집중 상태머신 + baseline — 웹 core/state-machine.ts·baseline.ts 1:1.
//!
//! 판정은 원뿔이 아니라 **비대칭 축정렬 박스**: |Δyaw|≤24°, Δpitch ∈ [−22°, +16°].
//! 히스테리시스: 집중→이탈 350ms / 이탈→집중 250ms / 무얼굴 600ms / 눈감김 450ms.
//! 함정(웹 검증 완료): ①무얼굴 600ms 미만은 **직전 상태 유지** ②MULTIPLE_FACES가
//! noFace/eyesClosed 타이머를 리셋하지 않음 ③score는 raw 집중값(표시값 아님) 누적.
//! 다중 모니터 투영 분류기는 레이아웃(호스트 제공)이 2개 이상일 때만 동작.

const YAW_HALF_DEG: f32 = 24.0;
const PITCH_UP_DEG: f32 = 16.0;
const PITCH_DOWN_DEG: f32 = 22.0;
const AWAY_ENTER_MS: f64 = 350.0;
const AWAY_EXIT_MS: f64 = 250.0;
const NO_FACE_MS: f64 = 600.0;
const BLINK_GRACE_MS: f64 = 450.0;
const SCORE_WINDOW_MS: f64 = 30_000.0;
const BASELINE_SAMPLES: usize = 24;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FocusStatus {
    Initializing,
    Focused,
    OtherMonitor,
    LookingAway,
    EyesClosed,
    NoFace,
    MultipleFaces,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gaze {
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct FocusResult {
    pub status: FocusStatus,
    pub attentive: bool,
    pub score: u32,
    pub monitor_index: i32,
}

/// 프레임 입력 (호스트/GazeTask가 채움)
#[derive(Clone, Copy, Debug)]
pub struct FocusFrame {
    pub ts_ms: f64,
    pub face_count: usize,
    pub gaze: Option<Gaze>,
    pub eyes_closed: bool,
}

/// 24샘플 독립 median baseline (웹: 짝수 upper-median = idx 12, 합성값 허용)
#[derive(Default)]
pub struct BaselineCollector {
    samples: Vec<Gaze>,
}

impl BaselineCollector {
    pub fn feed(&mut self, has_face: bool, gaze: Option<Gaze>) -> Option<Gaze> {
        let Some(g) = gaze.filter(|_| has_face) else {
            self.samples.clear();
            return None;
        };
        self.samples.push(g);
        if self.samples.len() < BASELINE_SAMPLES {
            return None;
        }
        let mut yaws: Vec<f32> = self.samples.iter().map(|s| s.yaw).collect();
        let mut pits: Vec<f32> = self.samples.iter().map(|s| s.pitch).collect();
        yaws.sort_by(|a, b| a.total_cmp(b));
        pits.sort_by(|a, b| a.total_cmp(b));
        self.samples.clear();
        Some(Gaze { yaw: yaws[BASELINE_SAMPLES / 2], pitch: pits[BASELINE_SAMPLES / 2] })
    }
}

pub struct FocusStateMachine {
    status: FocusStatus,
    display_attentive: bool,
    on_since: f64,
    off_since: f64,
    no_face_since: f64,
    eyes_closed_since: f64,
    history: std::collections::VecDeque<(f64, bool)>,
    pub baseline: Option<Gaze>,
}

impl Default for FocusStateMachine {
    fn default() -> Self {
        FocusStateMachine {
            status: FocusStatus::Initializing,
            display_attentive: false,
            on_since: 0.0,
            off_since: 0.0,
            no_face_since: 0.0,
            eyes_closed_since: 0.0,
            history: Default::default(),
            baseline: None,
        }
    }
}

impl FocusStateMachine {
    fn force(&mut self, s: FocusStatus) {
        self.status = s;
        self.display_attentive = false;
        self.on_since = 0.0;
        self.off_since = 0.0;
    }

    pub fn update(&mut self, f: FocusFrame) -> FocusResult {
        let now = f.ts_ms;
        let mut raw_attentive = false;
        let mut monitor_index = -1i32;
        let has_face = f.face_count >= 1 && f.gaze.is_some();

        if f.face_count >= 2 {
            self.force(FocusStatus::MultipleFaces);
        } else if !has_face {
            if self.no_face_since == 0.0 {
                self.no_face_since = now;
            }
            if now - self.no_face_since >= NO_FACE_MS {
                self.force(FocusStatus::NoFace);
            } // 600ms 미만: 직전 상태 유지 (웹 검증)
        } else {
            self.no_face_since = 0.0;
            if f.eyes_closed {
                if self.eyes_closed_since == 0.0 {
                    self.eyes_closed_since = now;
                }
            } else {
                self.eyes_closed_since = 0.0;
            }
            let eyes_ms =
                if self.eyes_closed_since != 0.0 { now - self.eyes_closed_since } else { 0.0 };

            let g = f.gaze.unwrap();
            let b = self.baseline.unwrap_or_default();
            let (dy, dp) = (g.yaw - b.yaw, g.pitch - b.pitch);
            let on_target =
                dy.abs() <= YAW_HALF_DEG && dp <= PITCH_UP_DEG && dp >= -PITCH_DOWN_DEG;

            if eyes_ms > BLINK_GRACE_MS {
                self.force(FocusStatus::EyesClosed);
            } else {
                raw_attentive = on_target;
                if on_target {
                    monitor_index = 0;
                    self.off_since = 0.0;
                    if self.on_since == 0.0 {
                        self.on_since = now;
                    }
                    if self.display_attentive || now - self.on_since >= AWAY_EXIT_MS {
                        self.display_attentive = true;
                        self.status = FocusStatus::Focused;
                    }
                } else {
                    self.on_since = 0.0;
                    if self.off_since == 0.0 {
                        self.off_since = now;
                    }
                    if !self.display_attentive || now - self.off_since >= AWAY_ENTER_MS {
                        self.display_attentive = false;
                        self.status = FocusStatus::LookingAway;
                    }
                }
            }
        }

        if f.face_count >= 1 {
            self.history.push_back((now, raw_attentive));
        }
        while self
            .history
            .front()
            .map(|(t, _)| *t < now - SCORE_WINDOW_MS)
            .unwrap_or(false)
        {
            self.history.pop_front();
        }
        let score = if self.history.is_empty() {
            100
        } else {
            let a = self.history.iter().filter(|(_, v)| *v).count();
            ((a as f64 / self.history.len() as f64) * 100.0).round() as u32
        };
        FocusResult {
            status: self.status,
            attentive: self.display_attentive,
            score,
            monitor_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ts: f64, yaw: f32, pitch: f32) -> FocusFrame {
        FocusFrame {
            ts_ms: ts,
            face_count: 1,
            gaze: Some(Gaze { yaw, pitch }),
            eyes_closed: false,
        }
    }

    #[test]
    fn hysteresis_enter_exit() {
        let mut m = FocusStateMachine::default();
        // 온타깃 250ms 이상 → FOCUSED
        let mut t = 0.0;
        for _ in 0..10 {
            m.update(frame(t, 0.0, 0.0));
            t += 100.0;
        }
        assert_eq!(m.update(frame(t, 0.0, 0.0)).status, FocusStatus::Focused);
        // 오프타깃 350ms 미만은 FOCUSED 유지
        t += 100.0;
        let r = m.update(frame(t, 40.0, 0.0));
        assert_eq!(r.status, FocusStatus::Focused, "350ms 미만 이탈은 래치 유지");
        // 350ms 경과 → LOOKING_AWAY
        t += 400.0;
        assert_eq!(m.update(frame(t, 40.0, 0.0)).status, FocusStatus::LookingAway);
    }

    #[test]
    fn asymmetric_pitch_box() {
        let mut m = FocusStateMachine::default();
        let mut t = 0.0;
        for _ in 0..10 {
            m.update(frame(t, 0.0, -20.0)); // 아래 20° — 허용(−22 하한)
            t += 100.0;
        }
        assert_eq!(m.update(frame(t, 0.0, -20.0)).status, FocusStatus::Focused);
        for _ in 0..10 {
            m.update(frame(t, 0.0, 20.0)); // 위 20° — 초과(+16 상한)
            t += 100.0;
        }
        assert_eq!(m.update(frame(t, 0.0, 20.0)).status, FocusStatus::LookingAway);
    }

    #[test]
    fn no_face_holds_status_before_600ms() {
        let mut m = FocusStateMachine::default();
        let mut t = 0.0;
        for _ in 0..10 {
            m.update(frame(t, 0.0, 0.0));
            t += 100.0;
        }
        let none = |ts| FocusFrame { ts_ms: ts, face_count: 0, gaze: None, eyes_closed: false };
        t += 100.0;
        assert_eq!(m.update(none(t)).status, FocusStatus::Focused, "600ms 미만 유지");
        t += 700.0;
        assert_eq!(m.update(none(t)).status, FocusStatus::NoFace);
    }

    #[test]
    fn baseline_median() {
        let mut b = BaselineCollector::default();
        for i in 0..23 {
            assert!(b.feed(true, Some(Gaze { yaw: i as f32, pitch: 0.0 })).is_none());
        }
        let r = b.feed(true, Some(Gaze { yaw: 23.0, pitch: 0.0 })).unwrap();
        assert_eq!(r.yaw, 12.0, "upper-median (idx 12)");
    }
}
