//! 손 제스처 판정 — 웹 hand-detector(gestures/*.ts) 이식 + **clap 정확도 개선**.
//!
//! 웹 clap의 실패 모드 (사용자 불만 "잘 인식이 안 됨"의 원인 분석):
//!  ①박수 순간 두 손이 겹쳐 검출기가 **한 손으로 융합** → `hands≥2` 전제가
//!    정확히 접촉 프레임에 깨져 rising edge를 놓친다 (모바일이 1↔2손 전이
//!    판정을 따로 만든 이유 — vcxrust_ai 참조)
//!  ②handedness 오판(양쪽 다 Left 등) → distinct=false로 전면 무발화
//!  ③검출 드랍 1프레임에 pair_frames 리셋 — 빠른 박수는 2연속 양손을 못 채움
//!
//! 개선 (웹 상수는 유지하고 경로를 추가):
//!  - **트랙 A**: 웹과 동일한 접촉 rising edge (CLOSE 0.4 / APART 1.2 / 쿨다운)
//!  - **트랙 B (융합 브리지)**: 양손이 근접(FUSE_D 이내)+접근 중 상태에서 한 손으로
//!    줄면 접촉으로 간주해 발화 — 빠른 박수의 주 경로
//!  - handedness는 **보조 신호**로 강등: 다르면 통과, 같아도 팜 중심 거리가
//!    충분히 떨어져 있으면 서로 다른 손으로 인정 (오판 내성)
//!  - pair_frames는 1프레임 드랍을 용서 (GRACE)

/// MediaPipe Hands 21점 토폴로지 인덱스
pub mod lm {
    pub const WRIST: usize = 0;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_MCP: usize = 5;
    pub const INDEX_TIP: usize = 8;
    pub const MIDDLE_MCP: usize = 9;
    pub const MIDDLE_TIP: usize = 12;
    pub const RING_MCP: usize = 13;
    pub const RING_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_TIP: usize = 20;
    pub const FINGERTIPS: [usize; 5] = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
}

/// 정규화 이미지 좌표 (0..1, y 아래 방향) 21점
pub type HandLm = [[f32; 2]; 21];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Handedness {
    Left,
    Right,
    Unknown,
}

#[derive(Clone, Copy, Debug)]
pub struct HandSnapshot {
    pub landmarks: HandLm,
    pub handedness: Handedness,
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// 팜 폭 — index-MCP ↔ pinky-MCP (스케일 앵커, 웹 동일)
pub fn palm_width(l: &HandLm) -> f32 {
    dist(l[lm::INDEX_MCP], l[lm::PINKY_MCP])
}

/// 팜 중심 — 네 손가락 MCP 평균 (웹 동일)
pub fn palm_center(l: &HandLm) -> [f32; 2] {
    let ids = [lm::INDEX_MCP, lm::MIDDLE_MCP, lm::RING_MCP, lm::PINKY_MCP];
    let (mut x, mut y) = (0.0, 0.0);
    for i in ids {
        x += l[i][0];
        y += l[i][1];
    }
    [x / 4.0, y / 4.0]
}

/// 두 손 손끝 최소거리 (웹 동일 — 박수는 손끝이 먼저 만난다)
pub fn min_fingertip_dist(a: &HandLm, b: &HandLm) -> f32 {
    let mut best = f32::INFINITY;
    for &i in &lm::FINGERTIPS {
        for &j in &lm::FINGERTIPS {
            best = best.min(dist(a[i], b[j]));
        }
    }
    best
}

// ── Clap (개선판) ──────────────────────────────────────────────────────────

/// 웹 상수 (clap.ts)
const CLOSE_THRESHOLD: f32 = 0.4;
const APART_THRESHOLD: f32 = 1.2;
const MIN_DISTINCT_HAND_DISTANCE: f32 = 0.12;
const COOLDOWN_MS: f64 = 350.0;
/// 개선 상수 — 실카메라로 튜닝할 것 (기록: 초기값은 보수적)
const FUSE_D: f32 = 1.0; // 이 거리 안에서 한 손으로 줄면 접촉 간주 (팜폭 단위)
const FUSE_GRACE_FRAMES: u32 = 5; // 융합 브리지 유효 프레임 수
const APPROACH_V: f32 = -2.5; // 접근 판정 속도 하한 (팜폭/초, 음수=접근)
const PAIR_DROP_GRACE: u32 = 1; // 양손 프레임 카운트가 용서하는 드랍 수
const REQUIRED_PAIR_FRAMES: u32 = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct ClapResult {
    pub fired: bool,
    pub confidence: f32,
}

pub struct ClapDetector {
    close: bool,
    pair_frames: u32,
    drop_budget: u32,
    last_fire_ms: f64,
    /// 직전 양손 관측 (d, ts) — 속도 추정·융합 브리지
    last_pair: Option<(f32, f64)>,
    vel: f32,
    fused_frames: u32,
}

impl Default for ClapDetector {
    fn default() -> Self {
        ClapDetector {
            close: false,
            pair_frames: 0,
            drop_budget: 0,
            last_fire_ms: f64::NEG_INFINITY,
            last_pair: None,
            vel: 0.0,
            fused_frames: 0,
        }
    }
}

impl ClapDetector {
    pub fn reset(&mut self) {
        *self = ClapDetector::default();
    }

    pub fn update(&mut self, hands: &[HandSnapshot], ts_ms: f64) -> ClapResult {
        let cooled = ts_ms - self.last_fire_ms >= COOLDOWN_MS;
        if hands.len() >= 2 {
            self.fused_frames = 0;
            let (a, b) = (&hands[0], &hands[1]);
            let scale = palm_width(&a.landmarks).max(palm_width(&b.landmarks)).max(1e-6);
            let d = min_fingertip_dist(&a.landmarks, &b.landmarks) / scale;
            // 같은 손 중복검출 배제 — handedness는 보조 (오판 내성):
            // 다르면 통과, 같더라도 팜 중심이 충분히 떨어져 있으면 서로 다른 손
            let center_d = dist(palm_center(&a.landmarks), palm_center(&b.landmarks)) / scale;
            let distinct = if a.handedness != b.handedness
                && a.handedness != Handedness::Unknown
                && b.handedness != Handedness::Unknown
            {
                center_d >= MIN_DISTINCT_HAND_DISTANCE
            } else {
                center_d >= MIN_DISTINCT_HAND_DISTANCE * 4.0 // 같은 라벨이면 더 엄격
            };
            if !distinct {
                self.close = false;
                self.pair_frames = 0;
                self.last_pair = None;
                return ClapResult::default();
            }
            // 속도 (팜폭/초, 음수 = 접근)
            if let Some((pd, pt)) = self.last_pair {
                let dt = ((ts_ms - pt) / 1000.0).max(1e-3) as f32;
                self.vel = (d - pd) / dt;
            }
            self.last_pair = Some((d, ts_ms));
            self.pair_frames += 1;
            self.drop_budget = PAIR_DROP_GRACE;

            if d >= APART_THRESHOLD {
                self.close = false;
            }
            if !self.close && self.pair_frames >= REQUIRED_PAIR_FRAMES && d <= CLOSE_THRESHOLD && cooled
            {
                self.close = true;
                self.last_fire_ms = ts_ms;
                return ClapResult {
                    fired: true,
                    confidence: (1.0 - d / CLOSE_THRESHOLD).clamp(0.0, 1.0),
                };
            }
            return ClapResult::default();
        }

        // ── 트랙 B: 융합 브리지 — 양손 근접+접근 중 → 한 손 = 접촉 ──
        if hands.len() == 1 {
            if let Some((pd, _)) = self.last_pair {
                if self.fused_frames < FUSE_GRACE_FRAMES {
                    self.fused_frames += 1;
                    let approaching = self.vel <= APPROACH_V || pd <= CLOSE_THRESHOLD * 1.5;
                    if !self.close
                        && self.pair_frames >= REQUIRED_PAIR_FRAMES
                        && pd <= FUSE_D
                        && approaching
                        && cooled
                    {
                        self.close = true;
                        self.last_fire_ms = ts_ms;
                        // 신뢰도: 마지막 거리·속도 결합
                        let c_d = (1.0 - pd / FUSE_D).clamp(0.0, 1.0);
                        let c_v = ((-self.vel - 1.0) / 6.0).clamp(0.0, 0.5);
                        return ClapResult { fired: true, confidence: (c_d + c_v).min(1.0) };
                    }
                    return ClapResult::default();
                }
            }
            // 드랍 용서 — pair_frames 유지 (1프레임 노이즈에 리셋하지 않음)
            if self.drop_budget > 0 {
                self.drop_budget -= 1;
                return ClapResult::default();
            }
        }

        self.close = false;
        self.pair_frames = 0;
        self.last_pair = None;
        self.vel = 0.0;
        if hands.is_empty() {
            self.fused_frames = 0;
        }
        ClapResult::default()
    }
}

// ── ThumbsUp / HandRaise (웹 1:1 이식 — 상수는 config.ts) ──────────────────

const TU_MIN_EXTEND_RATIO: f32 = 0.9;
const TU_MAX_TILT_DEG: f32 = 50.0;
const TU_MAX_FINGER_EXT_COS: f32 = 0.4;
const TU_MIN_MCP_CLEARANCE: f32 = 0.7;
const HR_MAX_Y: f32 = 0.5;
const HR_MIN_TIPS_ABOVE_WRIST: usize = 3;

fn finger_extension_cos(l: &HandLm, mcp: usize, pip: usize, tip: usize) -> f32 {
    let (ax, ay) = (l[pip][0] - l[mcp][0], l[pip][1] - l[mcp][1]);
    let (bx, by) = (l[tip][0] - l[pip][0], l[tip][1] - l[pip][1]);
    let la = (ax * ax + ay * ay).sqrt().max(1e-6);
    let lb = (bx * bx + by * by).sqrt().max(1e-6);
    (ax * bx + ay * by) / (la * lb)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GestureResult {
    pub matched: bool,
    pub confidence: f32,
}

pub fn classify_thumbs_up(l: &HandLm) -> GestureResult {
    const THUMB_CMC: usize = 1;
    const THUMB_MCP: usize = 2;
    const THUMB_IP: usize = 3;
    let (tip, ip, mcp, cmc) = (l[lm::THUMB_TIP], l[THUMB_IP], l[THUMB_MCP], l[THUMB_CMC]);
    if !(tip[1] < ip[1] && ip[1] < mcp[1]) {
        return GestureResult::default();
    }
    let (dx, dy) = (tip[0] - cmc[0], tip[1] - cmc[1]);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let tilt_deg = (-dy / len).clamp(-1.0, 1.0).acos().to_degrees();
    if tilt_deg > TU_MAX_TILT_DEG {
        return GestureResult::default();
    }
    let palm = palm_width(l).max(1e-6);
    if len / palm < TU_MIN_EXTEND_RATIO {
        return GestureResult::default();
    }
    let cosines = [
        finger_extension_cos(l, lm::INDEX_MCP, 6, lm::INDEX_TIP),
        finger_extension_cos(l, lm::MIDDLE_MCP, 10, lm::MIDDLE_TIP),
        finger_extension_cos(l, lm::RING_MCP, 14, lm::RING_TIP),
        finger_extension_cos(l, lm::PINKY_MCP, 18, lm::PINKY_TIP),
    ];
    let max_cos = cosines.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_cos > TU_MAX_FINGER_EXT_COS {
        return GestureResult::default();
    }
    let mcp_avg_y =
        (l[lm::INDEX_MCP][1] + l[lm::MIDDLE_MCP][1] + l[lm::RING_MCP][1] + l[lm::PINKY_MCP][1])
            / 4.0;
    let clearance = (mcp_avg_y - tip[1]) / palm;
    if clearance < TU_MIN_MCP_CLEARANCE {
        return GestureResult::default();
    }
    let tilt_m = 1.0 - tilt_deg / TU_MAX_TILT_DEG;
    let curl_m = (TU_MAX_FINGER_EXT_COS - max_cos) / (TU_MAX_FINGER_EXT_COS + 1.0);
    let clr_m = ((clearance - TU_MIN_MCP_CLEARANCE) / (1.5 - TU_MIN_MCP_CLEARANCE)).min(1.0);
    GestureResult {
        matched: true,
        confidence: (0.35 * tilt_m + 0.35 * curl_m + 0.3 * clr_m).clamp(0.0, 1.0),
    }
}

pub fn classify_hand_raise(l: &HandLm) -> GestureResult {
    let wrist = l[lm::WRIST];
    let mid_mcp = l[lm::MIDDLE_MCP];
    if mid_mcp[1] > HR_MAX_Y {
        return GestureResult::default();
    }
    let tips = [lm::INDEX_TIP, lm::MIDDLE_TIP, lm::RING_TIP, lm::PINKY_TIP];
    let up = tips.iter().filter(|&&i| l[i][1] < wrist[1]).count();
    if up < HR_MIN_TIPS_ABOVE_WRIST {
        return GestureResult::default();
    }
    let position = 1.0 - mid_mcp[1] / HR_MAX_Y;
    let pointing = up as f32 / 4.0;
    GestureResult {
        matched: true,
        confidence: (0.6 * position + 0.4 * pointing).clamp(0.0, 1.0),
    }
}

// ── 오케스트레이터 (classifier.ts 이식 + clap 발행 게이트 완화) ─────────────
//
// ⚠ 웹과의 의도적 차이: 웹은 최종 발행에서도 `hands≥2`를 강제하는데, 이는
// 접촉 순간 융합(양손→한손)된 박수를 죽이는 바로 그 규칙이다. 개선판은 융합
// 브리지 발화를 허용한다 — 브리지 자체가 "직전 양손 근접+접근" 이력을 요구하므로
// 한 손 단독 오발화는 여전히 구조적으로 불가능하다 (테스트로 고정).

const GESTURE_COOLDOWN_MS: f64 = 700.0;
const THUMBS_UP_HOLD_FRAMES: u32 = 3;
const HAND_RAISE_HOLD_FRAMES: u32 = 5;
const HAND_RAISE_RELEASE_FRAMES: u32 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gesture {
    ThumbsUp,
    HandRaise,
    Clap,
}

#[derive(Clone, Copy, Debug)]
pub struct GestureEvent {
    pub gesture: Gesture,
    pub confidence: f32,
    pub handedness: Handedness,
    pub ts_ms: f64,
}

#[derive(Clone, Copy, Default)]
struct RaiseState {
    up: u32,
    down: u32,
    fired: bool,
}

#[derive(Default)]
pub struct GestureClassifier {
    thumbs_hold: [u32; 3],  // Handedness 인덱스
    raise: [RaiseState; 3],
    last_emit: [f64; 3], // Gesture 인덱스
    clap: ClapDetector,
}

fn hidx(h: Handedness) -> usize {
    match h {
        Handedness::Left => 0,
        Handedness::Right => 1,
        Handedness::Unknown => 2,
    }
}

impl GestureClassifier {
    pub fn reset(&mut self) {
        *self = GestureClassifier::default();
    }

    pub fn classify(&mut self, hands: &[HandSnapshot], ts_ms: f64) -> Vec<GestureEvent> {
        let mut out = Vec::new();
        let mut seen = [false; 3];
        if self.last_emit == [0.0; 3] {
            self.last_emit = [f64::NEG_INFINITY; 3];
        }
        for h in hands {
            let i = hidx(h.handedness);
            seen[i] = true;
            // thumbsUp — N연속 유지 시 1회 (정확히 N번째 프레임)
            let r = classify_thumbs_up(&h.landmarks);
            self.thumbs_hold[i] = if r.matched { self.thumbs_hold[i] + 1 } else { 0 };
            if self.thumbs_hold[i] == THUMBS_UP_HOLD_FRAMES
                && ts_ms - self.last_emit[0] >= GESTURE_COOLDOWN_MS
            {
                self.last_emit[0] = ts_ms;
                out.push(GestureEvent {
                    gesture: Gesture::ThumbsUp,
                    confidence: r.confidence,
                    handedness: h.handedness,
                    ts_ms,
                });
            }
            // handRaise — rising edge + 내림 재무장
            let r = classify_hand_raise(&h.landmarks);
            let s = &mut self.raise[i];
            if r.matched {
                s.up += 1;
                s.down = 0;
                if !s.fired
                    && s.up >= HAND_RAISE_HOLD_FRAMES
                    && ts_ms - self.last_emit[1] >= GESTURE_COOLDOWN_MS
                {
                    s.fired = true;
                    self.last_emit[1] = ts_ms;
                    out.push(GestureEvent {
                        gesture: Gesture::HandRaise,
                        confidence: r.confidence,
                        handedness: h.handedness,
                        ts_ms,
                    });
                }
            } else {
                s.up = 0;
                s.down += 1;
                if s.down >= HAND_RAISE_RELEASE_FRAMES {
                    s.fired = false;
                }
            }
        }
        for i in 0..3 {
            if !seen[i] {
                self.thumbs_hold[i] = 0;
                let s = &mut self.raise[i];
                s.up = 0;
                s.down = HAND_RAISE_RELEASE_FRAMES;
                s.fired = false;
            }
        }
        // clap — 쿨다운은 ClapDetector 소유. 융합 브리지 발화 허용 (웹과 다름 —
        // 헤더 주석 참조)
        let c = self.clap.update(hands, ts_ms);
        if c.fired {
            out.push(GestureEvent {
                gesture: Gesture::Clap,
                confidence: c.confidence,
                handedness: Handedness::Unknown,
                ts_ms,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 손끝 최소거리가 d(팜폭 단위)가 되는 현실적 좌우 손 배치 —
    /// 손끝은 서로를 향하고 팜(MCP)은 뒤에 (팜폭 0.1, 팜 중심 간격 = d+1 팜폭)
    fn pair_at(d_norm: f32) -> [HandSnapshot; 2] {
        let pw = 0.1f32;
        let mk = |cx: f32, dir: f32, hd: Handedness| {
            let mut l = [[0f32; 2]; 21];
            for p in l.iter_mut() {
                *p = [cx, 0.5];
            }
            // 손끝: 팜에서 dir 방향으로 반 팜폭 앞
            for &i in &lm::FINGERTIPS {
                l[i] = [cx + dir * pw * 0.5, 0.5];
            }
            l[lm::INDEX_MCP] = [cx - pw / 2.0, 0.55];
            l[lm::PINKY_MCP] = [cx + pw / 2.0, 0.55];
            l[lm::MIDDLE_MCP] = [cx, 0.55];
            l[lm::RING_MCP] = [cx, 0.55];
            HandSnapshot { landmarks: l, handedness: hd }
        };
        // 팜 중심 간격: 손끝 갭 d·pw + 양쪽 손끝 오프셋 0.5pw×2
        let centers = d_norm * pw + pw;
        [
            mk(0.5 - centers / 2.0, 1.0, Handedness::Left),
            mk(0.5 + centers / 2.0, -1.0, Handedness::Right),
        ]
    }

    #[test]
    fn slow_clap_contact_edge_fires_once() {
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        for d in [2.0f32, 1.6, 1.2, 0.8] {
            assert!(!c.update(&pair_at(d), t).fired);
            t += 33.0;
        }
        assert!(c.update(&pair_at(0.3), t).fired, "접촉 rising edge 발화");
        t += 33.0;
        assert!(!c.update(&pair_at(0.2), t).fired, "붙인 채 유지 — 연사 금지");
        // 벌렸다 다시 접촉 (쿨다운 지나서) → 재발화
        t += 400.0;
        c.update(&pair_at(1.5), t);
        t += 33.0;
        assert!(c.update(&pair_at(0.3), t).fired, "재접촉 발화");
    }

    #[test]
    fn fast_clap_fusion_bridge_fires() {
        // 빠른 박수: 접근 중 접촉 직전에 한 손으로 융합 — 웹은 놓치는 케이스
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        for d in [2.4f32, 1.8, 1.0, 0.7] {
            c.update(&pair_at(d), t);
            t += 33.0;
        }
        let one = [pair_at(0.5)[0]];
        let r = c.update(&one, t);
        assert!(r.fired, "융합 브리지 발화 (vel {:.1})", c.vel);
        t += 33.0;
        assert!(!c.update(&one, t).fired, "융합 유지 중 연사 금지");
    }

    #[test]
    fn single_hand_never_fires_without_history() {
        let mut c = ClapDetector::default();
        let one = [pair_at(0.3)[0]];
        for i in 0..10 {
            assert!(!c.update(&one, i as f64 * 33.0).fired);
        }
    }

    #[test]
    fn same_handedness_still_fires_when_clearly_two_hands() {
        // handedness 오판(둘 다 Left) — 팜 중심이 충분히 떨어져 있으면 발화해야 한다
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        let mislabel = |d: f32| {
            let mut p = pair_at(d);
            p[1].handedness = Handedness::Left;
            p
        };
        for d in [2.0f32, 1.5, 1.0] {
            c.update(&mislabel(d), t);
            t += 33.0;
        }
        assert!(c.update(&mislabel(0.3), t).fired, "오판 내성 발화");
    }
}
