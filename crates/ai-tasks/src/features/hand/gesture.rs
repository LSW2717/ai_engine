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
    /// 손 존재 확률 [0,1] — 검출 문턱을 낮춰도 발화는 자신 있는 관측만 하도록
    /// 게이트한다. 모를 때는 1.0 (게이트 통과).
    pub presence: f32,
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

/// 손 축의 수직(위쪽) 대비 기울기 (deg, 0=손끝 위) — wrist→middle-MCP
pub fn hand_axis_deg(l: &HandLm) -> f32 {
    hand_axis_signed_deg(l).abs()
}

/// 부호 있는 손 축 기울기 (deg, 0=손끝 위, 시계방향 +) — 두 손 평행성 판정용
pub fn hand_axis_signed_deg(l: &HandLm) -> f32 {
    let (w, m) = (l[lm::WRIST], l[lm::MIDDLE_MCP]);
    (m[0] - w[0]).atan2(-(m[1] - w[1])).to_degrees()
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

// ── Clap (융합 브리지 주 경로) ─────────────────────────────────────────────
//
// 실측(박수 영상 4종, 15fps) 신호 구조:
//  - 접촉 순간 검출기는 거의 항상 두 손을 한 손으로 융합한다(2→1). 같은 손을
//    두 번 잡는 중복은 task.rs dedup_tracked가 1손으로 정리한다. 즉 "양손이
//    검출된 채 d≈0"인 진짜 접촉 관측은 드물다 — 손끝거리 rising edge만으로는
//    빠른 박수를 구조적으로 놓친다.
//  - 따라서 주 경로는 융합 브리지: 양손 접근 이력 + 한 손 전이 = 접촉.
//    중점 게이트(융합된 손이 직전 양손의 가운데 있어야)가 한 손 퇴장/드랍을
//    접촉으로 오인하는 것을 막는다.
//  - 트랙 A(양손 유지 접촉 rising edge)는 느린 기도자세 박수용으로 유지.
//  - 프레임 수 대신 시간창 — detectFps가 바뀌어도 판정이 흔들리지 않는다.

// 임계값 — 박수 영상 16종(정답 10/20/0/3/0/4/3/0/0/0/0/0/23/10/5/15,
// 지붕·X팔짱·갈퀴·머리짚기·근접비접촉·한손들기·한손흔들기 네거티브 포함)
// 15fps 그리드 서치(ai-bench clap_tune, 덤프 리플레이) 결과:
// 15/20/0/3/0/5/4/0/0/0/0/0/22/0/5/14, 오발화 0.
// 박수14(얼굴 앞 초근접 기도자세 연타)는 팜 검출기가 손을 아예 못 봐 0 —
// 검출 문턱을 낮추면(set_thresholds) 6~7회 잡히지만 한손 영상(11·12)의
// 유령 이중피팅 오발화가 어떤 판정 조합으로도 안 죽어 검출 기본값을 유지했다.
const CLOSE_THRESHOLD: f32 = 0.3;
/// 발화 래치 해제(재무장) 손끝거리 — 연타의 분리 위상이 얕아(실측 d 0.4~0.8)
/// 크게 잡으면 재무장이 안 된다. CLOSE와의 밴드는 쿨다운이 채터를 막는다.
const REARM_D: f32 = 0.4;
const MIN_DISTINCT_HAND_DISTANCE: f32 = 0.12;
/// 연타 실측 주기 ~270ms(박수13) — 이보다 길면 물리적으로 가능한 박수를 깎는다
const COOLDOWN_MS: f64 = 200.0;
/// 마지막 양손 관측(진짜/의사 쌍) 후 융합을 접촉으로 인정하는 시간창
const FUSE_WINDOW_MS: f64 = 600.0;
/// 융합 손 중심 ↔ 직전 양손 중점 허용 거리 (팜폭) — 한 손 퇴장 오인 방지
const MID_TOL: f32 = 1.3;
/// 위상 의존 분리 밴드: 래치 중(방금 접촉) 한 손이 중점에서 이만큼 벗어나면
/// 분리로 해석해 재무장한다. 같은 md라도 비래치면 융합(발화) — 접촉→분리→접촉이
/// 전부 한손 틱으로만 관측되는 초연타 구간의 유일한 재무장 신호.
const SEP_MD: f32 = 0.4;
/// 약한(경계) 분리 증거의 유효 시간창 — 발화 후 이 안에 온 분리만 재무장한다.
/// 진짜 박수는 타악이라 반동 분리가 빨리 온다(박수13 실측 ~334ms — 250이면
/// 초연타를 절반 놓친다). 홀드 지터도 이 창 안이면 재무장될 수 있어 붙인 채
/// 유지 중 +1 과발화가 생기지만(박수1 +5/박수7 +1), 연타 회수 손실보다 싸다.
const REBOUND_MS: f64 = 350.0;
/// 강한 분리 증거 — 리바운드 창과 무관하게 언제든 재무장:
/// 양손 손끝거리 ≥ STRONG_D (진짜로 벌림), 한손 중점 이탈 ≥ STRONG_MD,
/// 한손 홉, 손 전멸. 홀드 후 다시 치는 박수는 반드시 이 경로로 재무장된다
/// (박수7 실측: 재박수 전 분리 d 3~10).
const STRONG_D: f32 = 1.2;
const STRONG_MD: f32 = 1.4;
/// 약분리(d 0.4~1.2)라도 팜 중심이 이만큼 벌어졌으면 진짜 분리로 재무장 —
/// 얕게 벌리는 박수(실측 분리 d 0.55~0.77, cd 1.4~3.0)와 홀드 지터
/// (d 스파이크는 커도 cd 낮음)의 분리선.
const SEP_CD: f32 = 1.35;
/// 한손 홉 판정 점프 거리 (팜폭) — 연속 한손 틱에서 이만큼 점프 + 라벨 플립이면
/// 트래커가 분리된 두 손 사이를 오간 것: 두 위치로 의사 쌍을 재구성한다
const HOP_D: f32 = 1.1;
/// 손 축(wrist→middle-MCP)의 수직 대비 최대 기울기 — 박수는 손끝이 위를 향한
/// 두 손바닥의 접촉이다. 실측: 진짜 박수 |축|≤19°(박수13), 머리 위 지붕/내림
/// 48~72°. 화면 가장자리에서 치면 팔이 기울어 15°를 넘으므로 캡을 30°로 두고,
/// 홀드 자세(X팔짱 등) 배제는 아래 평행성 게이트가 담당한다.
const AXIS_UP_MAX_DEG: f32 = 30.0;
/// 두 손 축의 평행성 — 부호 있는 축각 차 |Δ|의 상한. 박수는 위치와 무관하게
/// 두 손바닥이 평행(실측 Δ≤17°)이고, X팔짱·지붕은 축이 교차/수렴한다(Δ 31~126).
/// 절대각 캡을 30°로 키우고도 네거티브 7종 오발화 0을 유지시키는 방어선.
const AXIS_PAIR_DIFF_MAX: f32 = 15.0;
/// 발화 최소 presence — 약검출(트래킹 유지용)로 상태는 굴리되, 발화·신뢰 시드는
/// 자신 있는 관측만. 0 = 비활성. 검출 문턱 완화(HandTask::set_thresholds)와
/// 짝으로 쓴다: 유령 이중피팅은 약검출에서 온다.
const FIRE_MIN_PRESENCE: f32 = 0.0;
/// 같은 라벨 근접쌍은 수평 배치일 때만 신뢰 — 한 손 위 이중피팅(유령 쌍)은
/// 수직 스택 + 라벨 동일이 시그니처다 (흔들기 실측 100% 수직). 진짜 박수의
/// 라벨 오판 쌍은 손이 좌우로 나란해 수평이 지배적이다.
const SAME_LABEL_NEED_HORIZONTAL: bool = false;
/// 한손 홉 점프는 수평 지배여야 인정 — 박수의 두 손은 좌우로 나란하므로
/// 트래커 홉도 수평 점프다. 흔드는 한 손의 라벨 노이즈 점프는 수직 성분이
/// 커서 (실측 dx 0.11 vs dy 0.24) 여기서 걸러진다.
const HOP_NEED_HORIZONTAL: bool = false;

// ── 근접쌍 신뢰 게이트 — 한 손 위 이중피팅(유령 쌍) 배제 ────────────────────
// 접촉 순간의 이중피팅은 진짜 접촉과 유령이 틱 단위로 동일하다(라벨·팜폭비·
// presence 전부 겹침, 실측). 아래 둘은 그중 유령 쪽에서만 관측된 특성:
/// 근접쌍(d ≤ 이 값)에만 신뢰 게이트를 적용한다 — 원거리 접근 쌍은 면제
const SEED_TRUST_MAX_D: f32 = 2.0;
/// 근접쌍 발화/시드 요건: 축 게이트(캡+평행성) 통과 + 절대 팜중심거리 ≤ 이 값
/// (랜드마크 스프롤 오핏 vs 진짜 접촉의 분리선. 0.30 — 근접비접촉 스윙(박수9)의
/// 최소 접근 0.25는 축·무장 게이트가 막아, 0.20→0.30 완화로도 오발화 0 유지)
const SEED_ABS_CD_MAX: f32 = 0.30;

// ── 발화 무장(arm) — "접촉 관측"만으로는 부족, 맥락 증거 필요 ────────────────
// 진짜 박수의 선행 맥락 4종 중 하나가 있어야 발화한다. 배회하는 한 손의
// 유령 쌍은 넷 다 없다.
/// ① 접근: 양손이 중거리에서 빠르게 수렴(직전 2h 대비 d가 이만큼 감소).
///   호버(안 닿고 유지, Δ≤0.42 실측)와 구분된다. 절대거리 하한이 유령
///   "수렴" (절대거리 0.05 미만 미소 요동)을 거른다.
const APPROACH_D_LO: f32 = 0.8;
const APPROACH_D_HI: f32 = 3.5;
const APPROACH_CONV_MIN: f32 = 0.5;
const APPROACH_ABS_CD_MIN: f32 = 0.10;
const APPROACH_WIN_MS: f64 = 1800.0;
/// ② 진입: 빈 화면 → 손 등장 직후 (기도자세 박수 — 손을 모아 올려 바로 침).
///   실측 진입→발화 200~400ms, 유령 배회 첫 발화는 1000ms+.
const ENTRY_WIN_MS: f64 = 700.0;
/// ③ 세션: 직전 발화 후 이 시간 안의 연타는 맥락 재증명 면제.
///   길수록 박수 사이 뜸(pause)에 관대하지만, 발화 직후 유령 자세 노출창도
///   같이 길어진다 — 13영상 검증 무영향 상한(12s) 아래의 중간값.
const SESSION_MS: f64 = 5000.0;
// (홉은 무장이 아니다 — 흔드는 한 손도 라벨 플립+점프를 만든다(실측 확신도
// 0.83~0.98로 진짜와 미분리). 홉의 쌍 재구성·재무장 역할만 유지하고, 무장은
// 접근/진입/세션 세 경로가 담당 — 연타 홉 구간은 세션이 이미 커버한다.)
/// 진입 인정 최소 공백 — 짧은 검출 깜빡임(흔들기 드랍아웃 실측 67~467ms)은
/// 진입이 아니다. 진짜 진입(손 올려 등장)은 공백 ≥800ms 실측.
const ENTRY_GAP_MS: f64 = 600.0;

// ── 진입 무장 전용 추가 증거 — 진입 직후는 맥락이 얇아 접촉 관측 1개로는
// 스침(sweep-through) 유령과 구분 불가. 셋 중 하나 필요:
/// (a) 지속: 근접쌍 관측 2회가 이 시간 이상에 걸침 (기도자세 유지 접촉).
/// 200이면 흔드는 한 손의 이중피팅 유령 쌍(66ms 간격 연속 관측)이 채워버린다
/// (박수12 실측) — 400은 유령 쌍의 수명(<350ms)보다 길어 진짜 유지 접촉만 남는다.
const ENTRY_PERSIST_MS: f64 = 400.0;
/// (b) 클린 쌍: 라벨 반대 + 팜중심 수평 배치 (이중피팅은 수직 스택)
/// (c) 반동: 접촉 시점 전후 이동 방향 반전 (박수 충돌 반동; 스침은 통과 지속)
///   — 발화를 이 시간만큼 보류하고 반전 확인 시 발화 (연타 재접촉이 267ms에
///   오므로 그보다 길게). 반전은 접촉 전 진동이 없고(흔들기 배제) 후행 변위가
///   작을 때만(반동=제자리 소진동 ≤0.05, 흔들기 턴=스윕 0.26+) 인정.
const ENTRY_CONFIRM_MS: f64 = 300.0;
const REBOUND_DISP_MAX: f32 = 0.15;

#[derive(Clone, Copy, Debug, Default)]
pub struct ClapResult {
    pub fired: bool,
    pub confidence: f32,
}

/// 판정 임계 묶음 — 기본값이 프로덕션 값(위 상수들)이다.
/// 오프라인 그리드 서치(ai-bench clap_tune)가 이 구조체로 스윕한다.
#[derive(Clone, Copy, Debug)]
pub struct ClapTuning {
    pub close_threshold: f32,
    pub rearm_d: f32,
    pub min_distinct: f32,
    pub cooldown_ms: f64,
    pub fuse_window_ms: f64,
    pub mid_tol: f32,
    pub sep_md: f32,
    pub rebound_ms: f64,
    pub strong_d: f32,
    pub strong_md: f32,
    pub sep_cd: f32,
    pub hop_d: f32,
    pub axis_up_max_deg: f32,
    pub axis_pair_diff_max: f32,
    pub fire_min_presence: f32,
    pub same_label_need_horizontal: bool,
    pub hop_need_horizontal: bool,
    pub seed_trust_max_d: f32,
    pub seed_abs_cd_max: f32,
    pub approach_d_lo: f32,
    pub approach_d_hi: f32,
    pub approach_conv_min: f32,
    pub approach_abs_cd_min: f32,
    pub approach_win_ms: f64,
    pub entry_win_ms: f64,
    pub session_ms: f64,
    pub entry_gap_ms: f64,
    pub entry_persist_ms: f64,
    pub entry_confirm_ms: f64,
    pub rebound_disp_max: f32,
}

impl Default for ClapTuning {
    fn default() -> Self {
        ClapTuning {
            close_threshold: CLOSE_THRESHOLD,
            rearm_d: REARM_D,
            min_distinct: MIN_DISTINCT_HAND_DISTANCE,
            cooldown_ms: COOLDOWN_MS,
            fuse_window_ms: FUSE_WINDOW_MS,
            mid_tol: MID_TOL,
            sep_md: SEP_MD,
            rebound_ms: REBOUND_MS,
            strong_d: STRONG_D,
            strong_md: STRONG_MD,
            sep_cd: SEP_CD,
            hop_d: HOP_D,
            axis_up_max_deg: AXIS_UP_MAX_DEG,
            axis_pair_diff_max: AXIS_PAIR_DIFF_MAX,
            fire_min_presence: FIRE_MIN_PRESENCE,
            same_label_need_horizontal: SAME_LABEL_NEED_HORIZONTAL,
            hop_need_horizontal: HOP_NEED_HORIZONTAL,
            seed_trust_max_d: SEED_TRUST_MAX_D,
            seed_abs_cd_max: SEED_ABS_CD_MAX,
            approach_d_lo: APPROACH_D_LO,
            approach_d_hi: APPROACH_D_HI,
            approach_conv_min: APPROACH_CONV_MIN,
            approach_abs_cd_min: APPROACH_ABS_CD_MIN,
            approach_win_ms: APPROACH_WIN_MS,
            entry_win_ms: ENTRY_WIN_MS,
            session_ms: SESSION_MS,
            entry_gap_ms: ENTRY_GAP_MS,
            entry_persist_ms: ENTRY_PERSIST_MS,
            entry_confirm_ms: ENTRY_CONFIRM_MS,
            rebound_disp_max: REBOUND_DISP_MAX,
        }
    }
}

/// 마지막 양손 관측 (진짜 쌍 또는 한손 홉 의사 쌍) — 융합 브리지 판정 재료
#[derive(Clone, Copy, Debug)]
struct PairObs {
    ts: f64,
    /// 두 팜 중심의 중점 (정규화 이미지 좌표)
    mid: [f32; 2],
    /// 스케일 앵커 — max 팜폭 (정규화 단위)
    pw: f32,
}

/// 직전 틱의 한 손 관측 — 홉 판정 재료
#[derive(Clone, Copy, Debug)]
struct SingleObs {
    ts: f64,
    center: [f32; 2],
    pw: f32,
    handedness: Handedness,
}

/// 진입 무장 전용 — 반동 확인 대기 중인 발화
#[derive(Clone, Copy, Debug)]
struct PendingFire {
    ts: f64,
    confidence: f32,
    /// 접촉 전 이동 방향 반전이 이미 있었나 — 흔들기 중 접촉은 반동 증거 무효
    oscillated: bool,
}

/// 근접쌍 관측 이력 — 진입 무장 증거(지속/클린) 재료
#[derive(Clone, Copy, Debug)]
struct ClosePairObs {
    ts: f64,
    opposite: bool,
    horizontal: bool,
}

pub struct ClapDetector {
    /// 판정 임계 — 기본은 프로덕션 상수, 튜너만 바꾼다
    t: ClapTuning,
    /// 발화 래치 — 재무장 조건: 양손 d≥REARM_D / 한 손이 중점을 벗어남(분리) /
    /// 홉 관측 / 손 전멸
    latched: bool,
    last_fire_ms: f64,
    pair: Option<PairObs>,
    prev_single: Option<SingleObs>,
    // 무장 상태
    last_approach_ms: f64,
    entry_ms: f64,
    /// 빈 화면 시작 시각 — 진입 공백 판정용 (None = 손 보이는 중)
    empty_since: Option<f64>,
    prev_pair_d: Option<f32>,
    // 진입 증거 이력 (고정 소용량 링 — 틱당 O(1), 힙 재할당 없음)
    close_pairs: Vec<ClosePairObs>,
    centers: Vec<(f64, [f32; 2])>,
    pending: Option<PendingFire>,
}

impl Default for ClapDetector {
    fn default() -> Self {
        ClapDetector {
            t: ClapTuning::default(),
            latched: false,
            last_fire_ms: f64::NEG_INFINITY,
            pair: None,
            prev_single: None,
            last_approach_ms: f64::NEG_INFINITY,
            entry_ms: f64::NEG_INFINITY,
            empty_since: Some(f64::NEG_INFINITY),
            prev_pair_d: None,
            close_pairs: Vec::with_capacity(16),
            centers: Vec::with_capacity(24),
            pending: None,
        }
    }
}

impl ClapDetector {
    pub fn reset(&mut self) {
        *self = ClapDetector { t: self.t, ..ClapDetector::default() };
    }

    /// 튜너 전용 — 임계 묶음을 바꿔 생성 (프로덕션은 Default)
    pub fn with_tuning(t: ClapTuning) -> Self {
        ClapDetector { t, ..ClapDetector::default() }
    }

    /// 무장 여부 — 발화에는 접촉 관측 + 선행 맥락(접근/홉/진입/세션)이 필요하다
    fn armed(&self, ts_ms: f64) -> bool {
        ts_ms - self.last_approach_ms <= self.t.approach_win_ms
            || ts_ms - self.entry_ms <= self.t.entry_win_ms
            || ts_ms - self.last_fire_ms <= self.t.session_ms
    }

    /// 진입 무장 단독 — 추가 증거(지속/클린/반동) 요구 대상
    fn entry_only(&self, ts_ms: f64) -> bool {
        ts_ms - self.entry_ms <= self.t.entry_win_ms
            && ts_ms - self.last_approach_ms > self.t.approach_win_ms
            && ts_ms - self.last_fire_ms > self.t.session_ms
    }

    /// 창 [lo, hi] 내 손 중심 변위 (반동 반전 판정용)
    fn displacement(&self, lo: f64, hi: f64) -> Option<[f32; 2]> {
        let mut first = None;
        let mut last = None;
        for &(t, c) in &self.centers {
            if t >= lo && t <= hi {
                if first.is_none() {
                    first = Some(c);
                }
                last = Some(c);
            }
        }
        match (first, last) {
            (Some(f), Some(l)) if f != l => Some([l[0] - f[0], l[1] - f[1]]),
            _ => None,
        }
    }

    fn persist_now(&self) -> bool {
        self.close_pairs.len() >= 2
            && self.close_pairs[self.close_pairs.len() - 1].ts - self.close_pairs[0].ts
                >= self.t.entry_persist_ms
    }

    fn clean_pair_now(&self) -> bool {
        self.close_pairs.last().is_some_and(|cp| cp.opposite && cp.horizontal)
    }

    /// 진입 무장 발화 시도 — 증거 있으면 즉발, 없으면 반동 확인 보류
    /// 접촉 전 600ms 창에 이동 방향 반전이 이미 있었나 — 흔들기 판별
    fn pre_oscillated(&self, ts_ms: f64) -> bool {
        let pts: Vec<&(f64, [f32; 2])> =
            self.centers.iter().filter(|(t, _)| ts_ms - t <= 600.0 && *t <= ts_ms).collect();
        if pts.len() < 3 {
            return false;
        }
        for w in pts.windows(3) {
            let d1 = [w[1].1[0] - w[0].1[0], w[1].1[1] - w[0].1[1]];
            let d2 = [w[2].1[0] - w[1].1[0], w[2].1[1] - w[1].1[1]];
            if d1[0] * d2[0] + d1[1] * d2[1] < 0.0 {
                return true;
            }
        }
        false
    }

    fn entry_fire(&mut self, ts_ms: f64, confidence: f32) -> ClapResult {
        self.latched = true;
        if self.persist_now() || self.clean_pair_now() {
            self.last_fire_ms = ts_ms;
            return ClapResult { fired: true, confidence };
        }
        self.pending =
            Some(PendingFire { ts: ts_ms, confidence, oscillated: self.pre_oscillated(ts_ms) });
        ClapResult::default()
    }

    pub fn update(&mut self, hands: &[HandSnapshot], ts_ms: f64) -> ClapResult {
        // 보류 발화 처리 — 반동 반전 확인 / 만료. 확인 발화 틱은 즉시 반환
        // (결과 1개/틱 — 진행 중 접촉은 다음 틱의 브리지/트랙A가 이어받는다)
        let pending_result = ClapResult::default();
        if let Some(p) = self.pending {
            if ts_ms - p.ts > self.t.entry_confirm_ms {
                self.pending = None;
                self.latched = false; // 만료 — 후속 접촉 재평가 허용
            } else {
                let pre = self.displacement(p.ts - 250.0, p.ts);
                let post = self.displacement(p.ts, ts_ms);
                if let (Some(pre), Some(post)) = (pre, post) {
                    let post_mag = (post[0] * post[0] + post[1] * post[1]).sqrt();
                    if pre[0] * post[0] + pre[1] * post[1] < 0.0
                        && !p.oscillated
                        && post_mag <= self.t.rebound_disp_max
                    {
                        // 쿨다운 기준은 발화(방출) 틱 — 접촉 시각(p.ts)으로
                        // 백데이트하면 확정 직후 접촉이 쿨다운을 건너뛴다
                        self.last_fire_ms = ts_ms;
                        self.pending = None;
                        self.latched = false;
                        return ClapResult { fired: true, confidence: p.confidence };
                    }
                }
            }
        }
        let cooled = ts_ms - self.last_fire_ms >= self.t.cooldown_ms;

        if hands.is_empty() {
            // 손이 완전히 사라짐 — 이력 폐기 + 재무장 (진입 무장은 재등장 시)
            if self.empty_since.is_none() {
                self.empty_since = Some(ts_ms);
            }
            self.latched = false;
            self.pair = None;
            self.prev_single = None;
            self.prev_pair_d = None;
            return pending_result;
        }
        if let Some(es) = self.empty_since {
            // 충분한 공백 후 등장만 진입 — 검출 깜빡임은 진입이 아니다
            if ts_ms - es >= self.t.entry_gap_ms {
                self.entry_ms = ts_ms;
            }
            self.empty_since = None;
        }

        if hands.len() >= 2 {
            let (a, b) = (&hands[0], &hands[1]);
            let scale = palm_width(&a.landmarks).max(palm_width(&b.landmarks)).max(1e-6);
            let d = min_fingertip_dist(&a.landmarks, &b.landmarks) / scale;
            let ca = palm_center(&a.landmarks);
            let cb = palm_center(&b.landmarks);
            let abs_cd = dist(ca, cb);
            let center_d = abs_cd / scale;
            self.centers.push((ts_ms, ca));
            self.centers.retain(|&(t, _)| ts_ms - t <= 600.0);
            // 같은 손 중복검출 배제 — handedness는 보조 (오판 내성):
            // 다르면 통과, 같더라도 팜 중심이 충분히 떨어져 있으면 서로 다른 손
            let distinct = if a.handedness != b.handedness
                && a.handedness != Handedness::Unknown
                && b.handedness != Handedness::Unknown
            {
                center_d >= self.t.min_distinct
            } else {
                center_d >= self.t.min_distinct * 4.0 // 같은 라벨이면 더 엄격
            };
            if !distinct {
                // dedup 이후 드문 잔재 — 관측만 무시하고 상태는 보존
                return pending_result;
            }
            // 접근 무장 관측 — 중거리에서 빠른 수렴 + 절대거리 실존.
            // 원거리 재이탈은 접근 취소 (호버 후 유령 접촉 차단).
            if d > self.t.approach_d_hi {
                self.last_approach_ms = f64::NEG_INFINITY;
            }
            if let Some(prev_d) = self.prev_pair_d {
                if (self.t.approach_d_lo..=self.t.approach_d_hi).contains(&d)
                    && abs_cd >= self.t.approach_abs_cd_min
                    && prev_d - d >= self.t.approach_conv_min
                {
                    self.last_approach_ms = ts_ms;
                }
            }
            self.prev_pair_d = Some(d);
            // 근접쌍 신뢰 게이트 — 실패 시 관측 무시 (재무장 증거는 위에서 처리됨)
            // 축 판정: 절대각 캡 + 두 손 평행성(|Δ|) — 가장자리 기운 박수는
            // 두 손이 같이 기울므로 Δ가 작다 (X팔짱·지붕은 교차/수렴)
            let axis_sa = hand_axis_signed_deg(&a.landmarks);
            let axis_sb = hand_axis_signed_deg(&b.landmarks);
            let axis_ok = axis_sa.abs().max(axis_sb.abs()) <= self.t.axis_up_max_deg
                && (axis_sa - axis_sb).abs() <= self.t.axis_pair_diff_max;
            // 발화/신뢰 시드는 자신 있는 관측만 — 약검출 유령 쌍 배제
            let confident = a.presence.min(b.presence) >= self.t.fire_min_presence;
            // 같은 라벨 + 수직 스택 = 한 손 이중피팅 시그니처 (옵션 게이트)
            let same_label = a.handedness == b.handedness;
            let stacked = (ca[1] - cb[1]).abs() > (ca[0] - cb[0]).abs();
            let ghost_stack = self.t.same_label_need_horizontal && same_label && stacked;
            let trusted = confident
                && !ghost_stack
                && (d > self.t.seed_trust_max_d
                    || (axis_ok && abs_cd <= self.t.seed_abs_cd_max));
            if d >= self.t.rearm_d
                && (d >= self.t.strong_d
                    || ts_ms - self.last_fire_ms <= self.t.rebound_ms
                    || center_d >= self.t.sep_cd)
            {
                self.latched = false;
            }
            if !trusted {
                return pending_result;
            }
            self.pair = Some(PairObs {
                ts: ts_ms,
                mid: [(ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0],
                pw: scale,
            });
            self.prev_single = None;
            if d <= 0.35 {
                self.close_pairs.push(ClosePairObs {
                    ts: ts_ms,
                    opposite: a.handedness != b.handedness,
                    horizontal: (ca[1] - cb[1]).abs() <= (ca[0] - cb[0]).abs(),
                });
                self.close_pairs.retain(|cp| ts_ms - cp.ts <= 800.0);
                // 보류 발화가 지속 증거로 확정 — 이번 틱은 확정 발화만 반환하고
                // 현 접촉은 다음 틱(브리지/트랙A)이 이어받는다 (결과 1개/틱 제약)
                if self.pending.is_some() && self.persist_now() {
                    let p = self.pending.take().unwrap();
                    // 쿨다운 기준은 방출 틱 (위 반동 확정과 동일한 이유)
                    self.last_fire_ms = ts_ms;
                    self.latched = false;
                    return ClapResult { fired: true, confidence: p.confidence };
                }
            }
            // 트랙 A — 양손이 유지된 채 접촉 (느린 박수·기도자세).
            // 축 게이트: 절대 캡 + 평행성 (X팔짱·지붕 자세 배제, 위치 불변).
            let both_up = axis_ok;
            if !self.latched
                && self.pending.is_none()
                && d <= self.t.close_threshold
                && cooled
                && both_up
                && self.armed(ts_ms)
            {
                let confidence = (1.0 - d / self.t.close_threshold).clamp(0.0, 1.0);
                if self.entry_only(ts_ms) {
                    let r = self.entry_fire(ts_ms, confidence);
                    if r.fired {
                        return r;
                    }
                } else {
                    self.latched = true;
                    self.last_fire_ms = ts_ms;
                    return ClapResult { fired: true, confidence };
                }
            }
            return pending_result;
        }

        // hands.len() == 1
        let c = palm_center(&hands[0].landmarks);
        let pw1 = palm_width(&hands[0].landmarks);
        self.centers.push((ts_ms, c));
        self.centers.retain(|&(t, _)| ts_ms - t <= 600.0);
        // 한손 홉 — 직전 한손 틱에서 위치가 크게 점프 + 라벨 플립 = 트래커가
        // 분리된 두 손 사이를 오간 것. 두 위치로 의사 쌍을 재구성한다
        // (연타 중 반대 손을 오래 못 되찾는 구간의 유일한 분리/접촉 신호).
        if let Some(s) = self.prev_single {
            if ts_ms - s.ts <= self.t.fuse_window_ms {
                let pw = pw1.max(s.pw).max(1e-6);
                let jump = dist(c, s.center) / pw;
                let horizontal = (c[0] - s.center[0]).abs() >= (c[1] - s.center[1]).abs();
                if jump >= self.t.hop_d
                    && hands[0].handedness != s.handedness
                    && (horizontal || !self.t.hop_need_horizontal)
                {
                    self.pair = Some(PairObs {
                        ts: ts_ms,
                        mid: [(c[0] + s.center[0]) / 2.0, (c[1] + s.center[1]) / 2.0],
                        pw,
                    });
                    self.latched = false; // 분리 관측 — 재무장 (무장은 아니다)
                }
            }
        }
        // 트랙 B — 융합 브리지. 한 손 틱의 중점 거리(md)는 위상에 따라 해석:
        //  · 래치 중(방금 접촉): md ≥ SEP_MD = 분리 위상 — 재무장
        //  · 비래치: md ≤ MID_TOL = 융합 접촉 — 발화 (축 게이트 적용)
        // 같은 md 값이라도 접촉 직후면 분리, 접근 직후면 접촉이다 — 초연타
        // 구간은 접촉·분리가 전부 한손 틱으로만 관측된다.
        match self.pair {
            Some(p) if ts_ms - p.ts <= self.t.fuse_window_ms => {
                let mid_dist = dist(c, p.mid) / p.pw;
                if self.latched {
                    if mid_dist >= self.t.sep_md
                        && (mid_dist >= self.t.strong_md
                            || ts_ms - self.last_fire_ms <= self.t.rebound_ms)
                    {
                        self.latched = false; // 분리 관측 — 재무장
                    }
                } else if self.pending.is_none()
                    && mid_dist <= self.t.mid_tol
                    && cooled
                    && hand_axis_deg(&hands[0].landmarks) <= self.t.axis_up_max_deg
                    && hands[0].presence >= self.t.fire_min_presence
                    && self.armed(ts_ms)
                {
                    let confidence = (1.0 - mid_dist / self.t.mid_tol).clamp(0.3, 1.0);
                    if self.entry_only(ts_ms) {
                        let r = self.entry_fire(ts_ms, confidence);
                        if r.fired {
                            self.prev_single = Some(SingleObs {
                                ts: ts_ms,
                                center: c,
                                pw: pw1,
                                handedness: hands[0].handedness,
                            });
                            return r;
                        }
                    } else {
                        self.latched = true;
                        self.last_fire_ms = ts_ms;
                        self.prev_single = Some(SingleObs {
                            ts: ts_ms,
                            center: c,
                            pw: pw1,
                            handedness: hands[0].handedness,
                        });
                        return ClapResult { fired: true, confidence };
                    }
                }
            }
            // 쌍 이력 만료는 재무장이 아니다 — 붙인 채 한 손으로만 보이는
            // 홀드에서 풀리면 다음 접촉 틱이 홀드 재발화가 된다 (박수7 6.6s).
            Some(_) | None => {}
        }
        self.prev_single =
            Some(SingleObs { ts: ts_ms, center: c, pw: pw1, handedness: hands[0].handedness });
        pending_result
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

    /// 튜너 전용 — clap 임계 묶음을 바꿔 생성 (프로덕션은 Default)
    pub fn with_clap_tuning(t: ClapTuning) -> Self {
        GestureClassifier { clap: ClapDetector::with_tuning(t), ..Default::default() }
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
            // 손끝 위 방향 — wrist가 MCP 아래 (축 게이트 통과)
            l[lm::WRIST] = [cx, 0.68];
            HandSnapshot { landmarks: l, handedness: hd, presence: 1.0 }
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
        assert!(c.update(&pair_at(0.25), t).fired, "접촉 rising edge 발화");
        t += 33.0;
        assert!(!c.update(&pair_at(0.2), t).fired, "붙인 채 유지 — 연사 금지");
        // 벌렸다 다시 접촉 (쿨다운 지나서) → 재발화
        t += 400.0;
        c.update(&pair_at(1.5), t);
        t += 33.0;
        assert!(c.update(&pair_at(0.25), t).fired, "재접촉 발화");
    }

    #[test]
    fn sustained_hold_with_jitter_never_refires() {
        // 붙인 채 유지 — 검출 지터(경계 d 스파이크)가 리바운드 창(REBOUND_MS)
        // **밖에** 오면 재무장하면 안 된다. 창 안 지터의 +1 재발화는 연타 회수
        // 회복(박수13)과 맞바꾼 의도된 트레이드오프 — 여기선 창 밖만 고정한다.
        // 진짜 재박수는 강한 분리(d≥STRONG_D) 또는 팜 중심 분리(SEP_CD) 후에만.
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        c.update(&pair_at(2.0), t);
        t += 66.0;
        assert!(c.update(&pair_at(0.25), t).fired, "첫 접촉 발화");
        // 발화 후 REBOUND_MS 동안은 조용한 홀드 (d < REARM_D — 재무장 불가),
        // 그 뒤 경계 지터 — 손끝 핏만 튀고 팜 중심(cd=1.2 < SEP_CD)은 붙어 있다
        for d in [0.1f32, 0.15, 0.1, 0.12, 0.1, 0.1, 0.5, 0.1, 0.8, 0.05, 1.0, 0.1, 0.9, 0.08]
        {
            t += 66.0;
            // pair_at(0.2)의 팜 배치(cd=1.2)를 유지한 채 손끝만 벌린다
            let mut p = pair_at(0.2);
            let spread = (d - 0.2).max(0.0) * 0.1 / 2.0;
            for &i in &lm::FINGERTIPS {
                p[0].landmarks[i][0] -= spread;
                p[1].landmarks[i][0] += spread;
            }
            assert!(!c.update(&p, t).fired, "홀드 지터 재발화 금지 (d={d})");
        }
        // 강한 분리 후 재접촉 → 발화
        t += 66.0;
        c.update(&pair_at(3.0), t);
        t += 66.0;
        assert!(c.update(&pair_at(0.25), t).fired, "진짜 분리 후 재발화");
    }

    #[test]
    fn tracker_hop_reconstructs_pair_and_fires() {
        // 연타 중 트래커가 분리된 두 손 사이를 오가는 구간(한손 틱만 연속) —
        // 라벨 플립 + 큰 점프 = 의사 쌍, 이어지는 중점 융합 = 박수.
        // 홉은 무장이 아니므로 접근 맥락(양손 수렴 관측)이 선행해야 한다.
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        for d in [3.0f32, 2.2] {
            c.update(&pair_at(d), t); // 수렴 Δ0.8 — 접근 무장
            t += 66.0;
        }
        let far = pair_at(3.0);
        c.update(&[far[0]], t); // Left @ 좌측
        c.update(&[far[1]], t + 66.0); // Right @ 우측 — 홉 (의사 쌍 + 재무장)
        // 중점(0.5) 부근으로 융합된 한 손 — far[0]의 팜 중심 x(0.3)를 0.5로 이동
        let mut mid = far[0];
        for p in mid.landmarks.iter_mut() {
            p[0] += 0.2;
        }
        let r = c.update(&[mid], t + 133.0);
        assert!(r.fired, "홉 의사 쌍 → 중점 융합 발화");
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
        assert!(r.fired, "융합 브리지 발화");
        t += 33.0;
        assert!(!c.update(&one, t).fired, "융합 유지 중 연사 금지");
    }

    #[test]
    fn hand_exit_is_not_a_clap() {
        // 양손이 벌어진 채 한 손이 사라짐(퇴장/드랍) — 남은 손이 중점에서 멀어
        // 융합으로 오인하면 안 된다
        let mut c = ClapDetector::default();
        let mut t = 0.0;
        for d in [2.2f32, 2.1, 2.0] {
            c.update(&pair_at(d), t);
            t += 66.0;
        }
        let one = [pair_at(2.0)[0]];
        assert!(!c.update(&one, t).fired, "퇴장은 박수가 아니다");
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
        for d in [2.4f32, 1.7, 1.0] {
            c.update(&mislabel(d), t);
            t += 33.0;
        }
        assert!(c.update(&mislabel(0.25), t).fired, "오판 내성 발화");
    }
}

