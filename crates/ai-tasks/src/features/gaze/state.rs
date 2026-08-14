//! 집중 상태머신 + baseline + 다중 모니터 분류기 — 웹 core/state-machine.ts·
//! baseline.ts 1:1.
//!
//! 판정은 원뿔이 아니라 **비대칭 축정렬 박스**: |Δyaw|≤24°, Δpitch ∈ [−22°, +16°].
//! 히스테리시스: 집중→이탈 350ms / 이탈→집중 250ms / 무얼굴 600ms / 눈감김 450ms.
//! 함정(웹 검증 완료): ①무얼굴 600ms 미만은 **직전 상태 유지** ②MULTIPLE_FACES가
//! noFace/eyesClosed 타이머를 리셋하지 않음 ③score는 raw 집중값(표시값 아님) 누적
//! ④모니터 매칭은 온타깃 박스 검사 **앞**에 온다 — 박스 안 시선도 인접 모니터가
//! 가로챌 수 있다 ⑤이미 이탈 상태에서 LOOKING_AWAY↔OTHER_MONITOR 전환은 즉시
//! (히스테리시스는 FOCUSED 래치에만) ⑥EYES_CLOSED도 monitor_index를 실어 나른다
//! (웹은 감김 판정 전에 시선을 분류) ⑦OTHER_MONITOR는 score에 비집중으로 쌓인다.

use serde::Deserialize;

const YAW_HALF_DEG: f32 = 24.0;
const PITCH_UP_DEG: f32 = 16.0;
const PITCH_DOWN_DEG: f32 = 22.0;
const AWAY_ENTER_MS: f64 = 350.0;
const AWAY_EXIT_MS: f64 = 250.0;
const NO_FACE_MS: f64 = 600.0;
const BLINK_GRACE_MS: f64 = 450.0;
const SCORE_WINDOW_MS: f64 = 30_000.0;
const BASELINE_SAMPLES: usize = 24;
/// 인접(맞닿은) 모니터의 정렬도 비례 문턱 완화 — 완전 정렬 시 24°→12°
const ADJACENT_SEAM_RELAX: f32 = 0.5;
/// 물리 각도(yawDeg) 모니터의 문턱 완화 계수 (60°에서 최대 40%)
const ANGLE_LENIENCY: f32 = 0.4;

/// 모니터 1개 — 웹 MonitorInfo 등가. 좌표는 가상 데스크톱 px (호스트
/// getScreenDetails). label·scaleFactor 등 표시/키잉 필드는 호스트 몫이라 뺐다.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MonitorInfo {
    pub index: i32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    /// 물리 yaw 각(도) — 레이아웃 오버라이드로만 설정 (기본 0)
    pub yaw_deg: f32,
}

/// 화면 레이아웃 — target_index는 **배열 위치** (웹 layout.targetIndex 규약:
/// 브라우저 창이 있는 모니터, isPrimary 아님).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ScreenLayout {
    pub monitors: Vec<MonitorInfo>,
    pub target_index: usize,
}

/// 사각형 인접 판정 — 공유 변(4px 허용) + 그 변 방향 겹침 (웹 monitorsAdjacent)
fn monitors_adjacent(a: &MonitorInfo, b: &MonitorInfo, tol: f32) -> bool {
    let (ar, ab) = (a.left + a.width, a.top + a.height);
    let (br, bb) = (b.left + b.width, b.top + b.height);
    let vert = (ar - b.left).abs() <= tol || (br - a.left).abs() <= tol;
    if vert && a.top < bb && b.top < ab {
        return true;
    }
    let hor = (ab - b.top).abs() <= tol || (bb - a.top).abs() <= tol;
    hor && a.left < br && b.left < ar
}

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

/// 시선 분류 결과 (웹 GazeClass)
struct GazeClass {
    on_target: bool,
    other_monitor: bool,
    monitor_index: i32,
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
    /// 다중 모니터 레이아웃 (호스트 제공, 2개 미만이면 분류기 비활성)
    pub layout: Option<ScreenLayout>,
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
            layout: None,
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

    /// 시선 → 온타깃/다른 모니터 분류. **모니터 매칭이 박스 검사보다 먼저다**
    /// (웹 classifyGaze — 박스 안 시선도 인접 모니터가 가로챌 수 있다).
    fn classify(&self, g: Gaze) -> GazeClass {
        let b = self.baseline.unwrap_or_default();
        let (dy, dp) = (g.yaw - b.yaw, g.pitch - b.pitch);
        let other = self.match_monitor(dy, dp);
        if other >= 0 {
            return GazeClass { on_target: false, other_monitor: true, monitor_index: other };
        }
        let on = dy.abs() <= YAW_HALF_DEG && dp <= PITCH_UP_DEG && dp >= -PITCH_DOWN_DEG;
        if on {
            GazeClass {
                on_target: true,
                other_monitor: false,
                monitor_index: self
                    .layout
                    .as_ref()
                    .map(|l| l.target_index as i32)
                    .unwrap_or(0),
            }
        } else {
            GazeClass { on_target: false, other_monitor: false, monitor_index: -1 }
        }
    }

    /// 시선 델타(도)를 타깃 모니터 중심 기준 각 모니터 방향에 투영해 매칭
    /// (웹 matchMonitorByGazeDelta). 단위 혼용은 설계다: 시선은 도(°), 모니터
    /// 방향은 px지만 /mMag로 단위 소거 → proj는 도 단위로 yawHalfDeg와 비교.
    /// 반환은 매칭된 모니터의 **index 필드** (타깃은 배열 위치 — 웹 그대로).
    fn match_monitor(&self, dy: f32, dp: f32) -> i32 {
        let Some(layout) = &self.layout else { return -1 };
        if layout.monitors.len() < 2 {
            return -1;
        }
        let Some(t) = layout.monitors.get(layout.target_index) else { return -1 };
        let cx = t.left + t.width / 2.0;
        let cy = t.top + t.height / 2.0;
        // 양의 pitch = 위 = 화면 y 음수
        let (gx, gy) = (dy, -dp);
        let g_mag = (gx * gx + gy * gy).sqrt();
        if g_mag < 0.001 {
            return -1;
        }
        let mut best_idx = -1i32;
        let mut best_proj = f32::NEG_INFINITY;
        for m in &layout.monitors {
            if m.index == t.index {
                continue;
            }
            let mx = m.left + m.width / 2.0 - cx;
            let my = m.top + m.height / 2.0 - cy;
            let m_mag = (mx * mx + my * my).sqrt();
            if m_mag < 1.0 {
                continue;
            }
            let proj = (gx * mx + gy * my) / m_mag;
            if proj <= 0.0 {
                continue; // 시선 반대편 모니터
            }
            let mut k = 1.0f32;
            if monitors_adjacent(t, m, 4.0) {
                // 완전 정렬(lateral 0)일수록 문턱 완화 — 이음새 넘김을 민감하게
                let cos_theta = proj / g_mag;
                let lateral = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let align = (1.0 - lateral).max(0.0);
                k = 1.0 - align * (1.0 - ADJACENT_SEAM_RELAX);
            }
            if m.yaw_deg != 0.0 {
                k *= 1.0 - (m.yaw_deg.abs().min(60.0) / 60.0) * ANGLE_LENIENCY;
            }
            if proj < YAW_HALF_DEG * k {
                continue;
            }
            if proj > best_proj {
                best_proj = proj;
                best_idx = m.index;
            }
        }
        best_idx
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

            // 웹 순서: 시선 분류가 감김 판정보다 먼저 — EYES_CLOSED도 index 유지
            let gz = self.classify(f.gaze.unwrap());
            monitor_index = gz.monitor_index;

            if eyes_ms > BLINK_GRACE_MS {
                self.force(FocusStatus::EyesClosed);
            } else {
                raw_attentive = gz.on_target;
                if gz.on_target {
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
                        self.status = if gz.other_monitor {
                            FocusStatus::OtherMonitor
                        } else {
                            FocusStatus::LookingAway
                        };
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

    /// 좌우 나란한 2모니터 레이아웃 (타깃=0, 오른쪽에 index 1)
    fn dual_layout() -> ScreenLayout {
        ScreenLayout {
            monitors: vec![
                MonitorInfo { index: 0, left: 0.0, top: 0.0, width: 1920.0, height: 1080.0, yaw_deg: 0.0 },
                MonitorInfo { index: 1, left: 1920.0, top: 0.0, width: 1920.0, height: 1080.0, yaw_deg: 0.0 },
            ],
            target_index: 0,
        }
    }

    #[test]
    fn other_monitor_classified_and_relabels_away() {
        let mut m = FocusStateMachine::default();
        m.layout = Some(dual_layout());
        let mut t = 0.0;
        for _ in 0..10 {
            m.update(frame(t, 0.0, 0.0));
            t += 100.0;
        }
        let r = m.update(frame(t, 0.0, 0.0));
        assert_eq!(r.status, FocusStatus::Focused);
        assert_eq!(r.monitor_index, 0, "온타깃은 target_index");
        // 오른쪽 30° — 인접 모니터 방향 (완전 정렬이라 문턱 12°). 350ms 히스테리시스
        // 지나면 OTHER_MONITOR (LOOKING_AWAY가 아니라)
        t += 100.0;
        m.update(frame(t, 30.0, 0.0));
        t += 400.0;
        let r = m.update(frame(t, 30.0, 0.0));
        assert_eq!(r.status, FocusStatus::OtherMonitor);
        assert_eq!(r.monitor_index, 1, "매칭 모니터의 index 필드");
        assert!(!r.attentive, "OTHER_MONITOR는 비집중");
        // 이미 이탈 상태 — 방향이 타깃 반대(왼쪽)로 바뀌면 즉시 LOOKING_AWAY
        t += 50.0;
        let r = m.update(frame(t, -40.0, 0.0));
        assert_eq!(r.status, FocusStatus::LookingAway);
        assert_eq!(r.monitor_index, -1);
    }

    #[test]
    fn adjacent_seam_relax_steals_in_box_gaze() {
        // 완전 정렬 시 문턱 24→12°: 박스 안(|Δyaw|≤24) 시선 15°도 인접 모니터가
        // 가로챈다 — "매칭이 박스 검사보다 먼저" 웹 순서의 검증
        let mut m = FocusStateMachine::default();
        m.layout = Some(dual_layout());
        let r = m.update(frame(0.0, 15.0, 0.0));
        assert_eq!(r.monitor_index, 1, "15°는 완화 문턱(12°)에 걸려야");
        // 비인접(간격 있는) 레이아웃이면 15°는 24° 문턱 미달 → 매칭 안 됨
        let mut far = dual_layout();
        far.monitors[1].left = 2400.0; // 480px 간격 — 인접 아님
        let mut m2 = FocusStateMachine::default();
        m2.layout = Some(far);
        let mut t = 0.0;
        for _ in 0..10 {
            m2.update(frame(t, 15.0, 0.0));
            t += 100.0;
        }
        let r = m2.update(frame(t, 15.0, 0.0));
        assert_eq!(r.status, FocusStatus::Focused, "비인접 15°는 온타깃 박스");
    }

    #[test]
    fn eyes_closed_carries_monitor_index() {
        let mut m = FocusStateMachine::default();
        m.layout = Some(dual_layout());
        let mut t = 0.0;
        let closed = |ts| FocusFrame {
            ts_ms: ts,
            face_count: 1,
            gaze: Some(Gaze { yaw: 0.0, pitch: 0.0 }),
            eyes_closed: true,
        };
        for _ in 0..8 {
            m.update(closed(t));
            t += 100.0;
        }
        let r = m.update(closed(t));
        assert_eq!(r.status, FocusStatus::EyesClosed);
        assert_eq!(r.monitor_index, 0, "감김에도 시선 분류 index 유지 (웹 순서)");
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
