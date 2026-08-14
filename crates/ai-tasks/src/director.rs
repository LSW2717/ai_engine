//! Director — **호스트 1개가 쓰는 전체 파이프라인 오케스트레이터.**
//!
//! studio.js(웹 데모)에서 검증한 배선 — 세그 파이프라인 + FaceTask(GPU 텍스처
//! 입력) + 터치업/메이크업 fx + 3D 아이템(크롭 동행·광원 프로브 8틱) + 손 제스처
//! (detectFps 페이싱, 이벤트 큐) + 집중도(비전 틱·num_faces=2) + 지연 로드 —
//! 를 Rust로 내린 것. `ai-ffi`(모바일)·`ai-wasm`(웹 워커)은 이 타입을 **접착만**
//! 한다: "바인딩에 분기가 생기면 로직이고 ai-tasks로 내려간다" 규칙의 이행.
//!
//! 계약:
//! - 설정은 **단일 JSON** — EffectsPatch(배경/블러/밝기/흑백/미러/회전/조명/
//!   프레이밍/터치업/메이크업) + 태스크 키(faceItems/handDetection/
//!   focusDetection). 머지 규약 동일: 없음=유지 / null=해제 / 값=설정.
//! - 모델 바이트 조달은 호스트 몫(`set_model`) — 세션 생성은 **지연**(켜질 때).
//! - 프레임: 호스트가 `with_frame`으로 프레임 텍스처를 채운 뒤 `frame()`.
//!   `needs_render()`면 target 필수, analyzer-only(손/집중도만)면 target 없이
//!   태스크만 돈다(세그·합성 생략 — 저사양 고속경로). `passthrough()`면 아무
//!   일도 없다.
//! - 결과는 우리 타입 그대로 JSON: `focus_json()`(FocusResult 7상태 전체),
//!   `poll_gesture_json()`(GestureEvent, 16슬롯 FIFO).
//! - `detach()` = 웜 리셋(세션·모델 유지 — 웜 워커), `reset()` = 세션 드랍
//!   (GPU 메모리 반납, 모델 바이트는 유지 — 다음 프레임 지연 재로드).

use std::collections::{HashMap, HashSet, VecDeque};

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use serde::Deserialize;

use crate::detect::gpu::GpuPre;
use crate::error::TaskError;
use crate::features::face::items3d::ItemsOverlay;
use crate::features::face::FaceTask;
use crate::features::gaze::GazeTask;
use crate::features::hand::gesture::{
    Gesture, GestureClassifier, GestureEvent, Handedness, HandSnapshot,
};
use crate::features::hand::HandTask;
use crate::features::vb::VideoPipeline;
use crate::session::gpu::GpuSession;

/// 존재=Some(값 or 해제) — params.rs `double`과 같은 세미antics
fn double<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Some(Option::<T>::deserialize(d)?))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FaceItems {
    pub enabled: bool,
    pub hat: String,
    pub eyewear: String,
    pub beard: String,
}

impl Default for FaceItems {
    fn default() -> Self {
        FaceItems { enabled: false, hat: "none".into(), eyewear: "none".into(), beard: "none".into() }
    }
}

impl FaceItems {
    fn any(&self) -> bool {
        self.enabled && (self.hat != "none" || self.eyewear != "none" || self.beard != "none")
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct HandDetection {
    pub enabled: bool,
    pub detect_fps: f32,
    /// 발화 허용 제스처 이름("thumbsUp"|"clap"|"handRaise") — 비면 전부
    pub gestures: Vec<String>,
}

impl Default for HandDetection {
    fn default() -> Self {
        HandDetection { enabled: false, detect_fps: 10.0, gestures: Vec::new() }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FocusDetection {
    pub enabled: bool,
    pub detect_fps: f32,
}

impl Default for FocusDetection {
    fn default() -> Self {
        FocusDetection { enabled: false, detect_fps: 10.0 }
    }
}

/// 태스크 토글 패치 — EffectsPatch와 같은 JSON에서 파싱 (서로 미지 키 무시)
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct TasksPatch {
    #[serde(deserialize_with = "double")]
    face_items: Option<Option<FaceItems>>,
    #[serde(deserialize_with = "double")]
    hand_detection: Option<Option<HandDetection>>,
    #[serde(deserialize_with = "double")]
    focus_detection: Option<Option<FocusDetection>>,
}

/// 모델 종류 키 (`set_model`) — 문서화된 유일한 이름들
pub const MODEL_KINDS: [&str; 7] =
    ["seg", "face_det", "face_lm", "gaze", "gaze_bs", "hand_det", "hand_lm"];

const GESTURE_QUEUE_CAP: usize = 16;

fn gesture_name(g: Gesture) -> &'static str {
    match g {
        Gesture::ThumbsUp => "thumbsUp",
        Gesture::Clap => "clap",
        Gesture::HandRaise => "handRaise",
    }
}

fn handedness_name(h: Handedness) -> &'static str {
    match h {
        Handedness::Left => "left",
        Handedness::Right => "right",
        Handedness::Unknown => "unknown",
    }
}

pub struct Director {
    pub pipeline: VideoPipeline,
    target_format: wgpu::TextureFormat,
    pre: Option<GpuPre>,
    /// 호스트가 주입한 모델 바이트 (세션은 지연 생성)
    models: HashMap<&'static str, Vec<u8>>,
    seg: Option<GpuSession>,
    face: Option<(FaceTask, GpuSession, GpuSession)>, // (task, det, lm)
    items: Option<ItemsOverlay>,
    items_loaded: HashSet<String>,
    /// GLB 바이트 조달 (ffi: 디렉터리 fs 로더 / wasm: set_item_glb로 선주입)
    glb_loader: Option<Box<dyn Fn(&str) -> Option<Vec<u8>> + Send>>,
    hand: Option<(HandTask, GpuSession, GpuSession)>,
    gestures: GestureClassifier,
    gesture_queue: VecDeque<GestureEvent>,
    gaze: Option<(GazeTask, GpuSession, Option<GpuSession>)>,
    face_items: Option<FaceItems>,
    hand_cfg: Option<HandDetection>,
    focus_cfg: Option<FocusDetection>,
    last_face: Option<Vec<[f32; 3]>>,
    last_focus_ms: f64,
    last_hand_ms: f64,
    probe_tick: u32,
    last_focus: Option<crate::features::gaze::FocusResult>,
}

impl Director {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        Director {
            pipeline: VideoPipeline::new(ctx, target_format),
            target_format,
            pre: None,
            models: HashMap::new(),
            seg: None,
            face: None,
            items: None,
            items_loaded: HashSet::new(),
            glb_loader: None,
            hand: None,
            gestures: GestureClassifier::default(),
            gesture_queue: VecDeque::new(),
            gaze: None,
            face_items: None,
            hand_cfg: None,
            focus_cfg: None,
            last_face: None,
            last_focus_ms: f64::NEG_INFINITY,
            last_hand_ms: f64::NEG_INFINITY,
            probe_tick: 0,
            last_focus: None,
        }
    }

    /// 단일 JSON 설정 — EffectsPatch + 태스크 키. 머지 규약: 없음=유지/null=해제
    pub fn apply_json(&mut self, json: &str) -> Result<(), String> {
        self.pipeline.apply_json(json)?;
        let t: TasksPatch =
            serde_json::from_str(json).map_err(|e| format!("TasksPatch 파싱: {e}"))?;
        if let Some(v) = t.face_items {
            self.face_items = v.filter(|f| f.any());
            if let Some(items) = &mut self.items {
                let f = self.face_items.clone().unwrap_or_default();
                items.set_items(&f.hat, &f.eyewear, &f.beard);
            }
        }
        if let Some(v) = t.hand_detection {
            let on = v.as_ref().map(|h| h.enabled).unwrap_or(false);
            self.hand_cfg = v.filter(|h| h.enabled);
            if !on {
                if let Some((task, ..)) = &mut self.hand {
                    task.reset();
                }
                self.gestures.reset();
                self.gesture_queue.clear();
            }
        }
        if let Some(v) = t.focus_detection {
            let on = v.as_ref().map(|f| f.enabled).unwrap_or(false);
            self.focus_cfg = v.filter(|f| f.enabled);
            if !on {
                if let Some((task, ..)) = &mut self.gaze {
                    task.reset(); // 스트림 파기 규약 — 재켬은 백지에서 (studio 동일)
                }
                self.last_focus = None;
            }
        }
        Ok(())
    }

    /// 다중 모니터 레이아웃 (JSON | "null") — GazeTask::set_layout_json 전달
    pub fn set_focus_layout_json(&mut self, json: &str) -> Result<(), String> {
        if let Some((task, ..)) = &mut self.gaze {
            task.set_layout_json(json)
        } else {
            // 게이즈 미기동이면 다음 기동 때 반영되도록 보관할 수도 있으나,
            // 레이아웃은 호스트가 focus 켠 뒤 보내는 게 자연스러워 단순 무시하지
            // 않고 에러로 알린다 (배선 실수 조기 발견).
            Err("집중도 미기동 — focusDetection을 먼저 켜고 한 프레임 이후 호출".into())
        }
    }

    /// 배경 이미지 (RGBA8) — EffectsPatch background:"image"가 이걸 쓴다
    pub fn set_background_image(&mut self, ctx: &GpuContext, rgba: &[u8], w: u32, h: u32) {
        self.pipeline.set_background_image(ctx, rgba, w, h);
    }

    /// 모델 바이트 주입 — kind는 `MODEL_KINDS` 참조. 같은 kind 재주입 = 교체
    /// (기존 세션 드랍 → 다음 사용 시 재로드).
    pub fn set_model(&mut self, kind: &str, bytes: Vec<u8>) -> Result<(), String> {
        let Some(&k) = MODEL_KINDS.iter().find(|&&k| k == kind) else {
            return Err(format!("미지 모델 kind: {kind} (가능: {MODEL_KINDS:?})"));
        };
        self.models.insert(k, bytes);
        match k {
            "seg" => {
                self.seg = None;
                self.pipeline.invalidate();
            }
            "face_det" | "face_lm" => self.face = None,
            "gaze" | "gaze_bs" => self.gaze = None,
            "hand_det" | "hand_lm" => self.hand = None,
            _ => {}
        }
        Ok(())
    }

    /// GLB 바이트 조달자 (ffi: 디렉터리 fs 로더). wasm은 대신 set_item_glb로 선주입
    pub fn set_glb_loader(&mut self, f: Box<dyn Fn(&str) -> Option<Vec<u8>> + Send>) {
        self.glb_loader = Some(f);
    }

    /// GLB 선주입 (종류당 1회 — 호스트 fetch)
    pub fn set_item_glb(
        &mut self,
        ctx: &GpuContext,
        kind: &str,
        bytes: &[u8],
    ) -> Result<(), TaskError> {
        if self.items.is_none() {
            self.items = Some(ItemsOverlay::new(ctx)?);
        }
        self.items.as_mut().unwrap().preload_glb(ctx, kind, bytes)?;
        self.items_loaded.insert(kind.to_string());
        Ok(())
    }

    // ── 활성 판정 (호스트의 passthrough/타깃 결정 근거) ──

    /// 세그+합성(비디오 경로)이 필요한가 — 3D 아이템도 합성 타깃이 필요하다
    pub fn needs_render(&self) -> bool {
        self.pipeline.state.any_active() || self.face_items.is_some()
    }

    /// analyzer 태스크(손/집중도)가 켜져 있는가
    pub fn tasks_active(&self) -> bool {
        self.hand_cfg.is_some() || self.focus_cfg.is_some()
    }

    /// 완전 무가공 — 호스트는 원본 프레임을 그대로 내보내면 된다 (제로카피)
    pub fn passthrough(&self) -> bool {
        !self.needs_render() && !self.tasks_active()
    }

    /// 손/집중도 CNN이 u8 RGB 픽셀을 요구하는 틱인가 — 호스트가 이때만 픽셀을
    /// 뽑으면 된다 (웹 getImageData 절약 — studio.js와 같은 규약)
    pub fn wants_pixels(&self, t_ms: f64) -> bool {
        let focus_due = self.focus_cfg.as_ref().map(|c| {
            t_ms - self.last_focus_ms >= 1000.0 / c.detect_fps.max(1.0) as f64
        });
        let hand_due = self.hand_cfg.as_ref().map(|c| {
            t_ms - self.last_hand_ms >= 1000.0 / c.detect_fps.max(1.0) as f64
        });
        let probe_due = self.face_items.is_some() && self.probe_tick % 8 == 0;
        focus_due.unwrap_or(false) || hand_due.unwrap_or(false) || probe_due
    }

    // ── 프레임 ──

    /// 프레임 텍스처 확보+채움 — 비디오 경로면 세그 세션에 묶인 ensure,
    /// analyzer-only면 세션 없는 ensure. 채우기(write_texture/외부 임포트)는
    /// 호출자 몫 (플랫폼별로 진짜 다른 유일한 부분).
    pub async fn with_frame<R>(
        &mut self,
        ctx: &GpuContext,
        w: u32,
        h: u32,
        f: impl FnOnce(&wgpu::Texture) -> R,
    ) -> Result<R, TaskError> {
        if self.needs_render() {
            self.ensure_seg(ctx).await?;
            let seg = self
                .seg
                .as_ref()
                .ok_or_else(|| TaskError::Other("seg 모델 미주입 (set_model \"seg\")".into()))?;
            self.pipeline.with_frame_texture(ctx, seg, w, h, f)
        } else {
            // analyzer-only: 마스크 경로를 안 쓰므로 EMA 해상도는 형식값
            self.pipeline.with_frame_texture_nogpu(ctx, w, h, (2, 2), f)
        }
    }

    /// with_frame의 B티어판 (frame_mask 짝) — 세그 세션 없이 프레임 텍스처 확보.
    /// mask_dims = 주입할 외부 마스크 해상도 (EMA 해상도).
    pub fn with_frame_mask<R>(
        &mut self,
        ctx: &GpuContext,
        w: u32,
        h: u32,
        mask_dims: (u32, u32),
        f: impl FnOnce(&wgpu::Texture) -> R,
    ) -> Result<R, TaskError> {
        self.pipeline.with_frame_texture_nogpu(ctx, w, h, mask_dims, f)
    }

    /// 한 프레임 (A티어 — GPU 추론): 비디오 경로(세그→스택→합성→아이템) +
    /// 태스크(face fx·집중도·손 제스처). target은 needs_render()일 때 필수.
    /// rgb = 원본 프레임 u8 RGB (wants_pixels 틱에만 필요 — None이면 해당 틱은
    /// 조용히 건너뛴다).
    pub async fn frame(
        &mut self,
        ctx: &GpuContext,
        fw: u32,
        fh: u32,
        target: Option<&wgpu::TextureView>,
        rgb: Option<&[u8]>,
        t_ms: f64,
    ) -> Result<(), TaskError> {
        if self.pipeline.state.any_active() || self.face_items.is_some() {
            let target = target.ok_or_else(|| {
                TaskError::Other("needs_render()인데 target 없음".into())
            })?;
            self.ensure_seg(ctx).await?;
            if self.seg.is_none() {
                return Err(TaskError::Other("seg 모델 미주입 (set_model \"seg\")".into()));
            }
            let seg = self.seg.as_mut().unwrap();
            self.pipeline.process_gpu(ctx, seg, fw, fh, target).await?;
        }
        self.run_tasks(ctx, fw, fh, target, rgb, t_ms).await
    }

    /// 한 프레임 (B티어 — CPU 추론 마스크 주입): 세그 모델·세션 없이 외부
    /// 마스크로 이펙트 스택을 돌린다 (process_mask_nogpu). 태스크는 A티어와 동일.
    /// 주의: 프레임 텍스처는 with_frame으로 이미 채워져 있어야 한다.
    #[allow(clippy::too_many_arguments)]
    pub async fn frame_mask(
        &mut self,
        ctx: &GpuContext,
        mask: &[f32],
        ch: u32,
        mask_w: u32,
        mask_h: u32,
        fw: u32,
        fh: u32,
        target: &wgpu::TextureView,
        rgb: Option<&[u8]>,
        t_ms: f64,
    ) -> Result<(), TaskError> {
        self.pipeline
            .process_mask_nogpu(ctx, mask, ch, mask_w, mask_h, true, fw, fh, target)?;
        self.run_tasks(ctx, fw, fh, Some(target), rgb, t_ms).await
    }

    /// 태스크 공용부 — face fx·아이템·집중도·손 제스처 (frame/frame_mask 공유)
    async fn run_tasks(
        &mut self,
        ctx: &GpuContext,
        fw: u32,
        fh: u32,
        target: Option<&wgpu::TextureView>,
        rgb: Option<&[u8]>,
        t_ms: f64,
    ) -> Result<(), TaskError> {
        // 얼굴 소비자 (아이템·터치업/메이크업·집중도)
        let fx_on = self.pipeline.state.touch_up.is_some() || self.pipeline.state.makeup.is_some();
        let focus_due = match &self.focus_cfg {
            Some(c) => t_ms - self.last_focus_ms >= 1000.0 / c.detect_fps.max(1.0) as f64,
            None => false,
        };
        let want_face = self.face_items.is_some() || fx_on || focus_due;
        let mut face_count = 0usize;
        if want_face && self.ensure_face(ctx).await? {
            let view = self
                .pipeline
                .frame_view()
                .ok_or_else(|| TaskError::Other("프레임 텍스처 없음 — with_frame 먼저".into()))?
                .0;
            let pre = self.pre.as_ref().unwrap();
            let (task, det, lm) = self.face.as_mut().unwrap();
            // MULTIPLE_FACES 감시는 집중도 켰을 때만 (디텍터 상시 비용 제거)
            task.set_num_faces(if self.focus_cfg.is_some() { 2 } else { 1 });
            let r = task.process_tex(ctx, pre, &view, det, lm, fw, fh, t_ms).await?;
            face_count = task.face_count();
            self.last_face = r.map(|f| f.points);
            // 터치업/메이크업 오버레이 (off·소실이면 내부 no-op/해제)
            self.pipeline.update_face_fx(ctx, self.last_face.as_deref());
        }

        // 3) 3D 아이템 — 합성 위 오버레이 (프레이밍 크롭 동행)
        if let Some(fi) = self.face_items.clone() {
            if let Some(target) = target {
                self.ensure_items(ctx, &fi)?;
                if let Some(items) = &mut self.items {
                    items.set_items(&fi.hat, &fi.eyewear, &fi.beard);
                    items.set_pose(self.last_face.clone());
                    let (s, cx, cy) = self.pipeline.framing_current();
                    items.set_view_crop(s, cx, cy);
                    self.probe_tick = self.probe_tick.wrapping_add(1);
                    if let Some(rgb) = rgb {
                        if self.probe_tick % 8 == 1 {
                            items.renderer.probe_scene_light_rgb_now(
                                rgb,
                                fw as usize,
                                fh as usize,
                            );
                        }
                    }
                    items.draw(ctx, target, self.target_format, fw, fh);
                }
            }
        }

        // 4) 집중도 (비전 틱 — rgb 필요: 게이즈 CNN 크롭)
        if focus_due {
            if let Some(rgb) = rgb {
                if self.ensure_gaze(ctx).await? {
                    self.last_focus_ms = t_ms;
                    let pts2: Option<Vec<[f32; 2]>> = self
                        .last_face
                        .as_ref()
                        .map(|p| p.iter().map(|q| [q[0], q[1]]).collect());
                    let (task, gaze_s, bs_s) = self.gaze.as_mut().unwrap();
                    let r = task
                        .process_gpu(
                            ctx,
                            gaze_s,
                            bs_s.as_mut(),
                            rgb,
                            fw as usize,
                            fh as usize,
                            pts2.as_deref(),
                            face_count,
                            t_ms,
                        )
                        .await?;
                    self.last_focus = Some(r);
                }
            }
        }

        // 5) 손 제스처 (detectFps 페이싱 — rgb 필요)
        let hand_due = match &self.hand_cfg {
            Some(c) => t_ms - self.last_hand_ms >= 1000.0 / c.detect_fps.max(1.0) as f64,
            None => false,
        };
        if hand_due {
            if let Some(rgb) = rgb {
                if self.ensure_hand(ctx).await? {
                    self.last_hand_ms = t_ms;
                    let (task, det, lm) = self.hand.as_mut().unwrap();
                    let hands = task.process_gpu(ctx, det, lm, rgb, fw, fh, t_ms).await?;
                    let snaps: Vec<HandSnapshot> = hands
                        .iter()
                        .map(|h| {
                            let mut landmarks = [[0f32; 2]; 21];
                            for (i, p) in h.points.iter().take(21).enumerate() {
                                landmarks[i] = [p[0], p[1]];
                            }
                            let handedness = if h.handedness > 0.5 {
                                Handedness::Left
                            } else {
                                Handedness::Right
                            };
                            HandSnapshot { landmarks, handedness }
                        })
                        .collect();
                    let filter = self.hand_cfg.as_ref().map(|c| c.gestures.clone()).unwrap_or_default();
                    for ev in self.gestures.classify(&snaps, t_ms) {
                        if !filter.is_empty()
                            && !filter.iter().any(|g| g == gesture_name(ev.gesture))
                        {
                            continue;
                        }
                        if self.gesture_queue.len() >= GESTURE_QUEUE_CAP {
                            self.gesture_queue.pop_front(); // 가장 오래된 것 드랍
                        }
                        self.gesture_queue.push_back(ev);
                    }
                }
            }
        }
        Ok(())
    }

    // ── 결과 (풀 방식 — 호스트 폴링) ──

    /// 마지막 집중도 — FocusResult 전체 필드 JSON (미기동이면 INITIALIZING)
    pub fn focus_json(&self) -> String {
        match &self.last_focus {
            Some(r) => {
                let (yaw, pitch) = self
                    .gaze
                    .as_ref()
                    .and_then(|(t, ..)| t.last_filtered)
                    .map(|g| (Some(g.yaw), Some(g.pitch)))
                    .unwrap_or((None, None));
                format!(
                    concat!(
                        "{{\"status\":\"{}\",\"attentive\":{},\"score\":{},",
                        "\"monitorIndex\":{},\"yaw\":{},\"pitch\":{}}}"
                    ),
                    r.status.as_str(),
                    r.attentive,
                    r.score,
                    r.monitor_index,
                    yaw.map(|v| format!("{v:.2}")).unwrap_or("null".into()),
                    pitch.map(|v| format!("{v:.2}")).unwrap_or("null".into()),
                )
            }
            None => concat!(
                "{\"status\":\"INITIALIZING\",\"attentive\":false,\"score\":100,",
                "\"monitorIndex\":-1,\"yaw\":null,\"pitch\":null}"
            )
            .into(),
        }
    }

    /// 제스처 이벤트 하나 꺼내기 (FIFO) — 없으면 None
    pub fn poll_gesture_json(&mut self) -> Option<String> {
        self.gesture_queue.pop_front().map(|ev| {
            format!(
                "{{\"gesture\":\"{}\",\"confidence\":{:.3},\"handedness\":\"{}\",\"tsMs\":{:.1}}}",
                gesture_name(ev.gesture),
                ev.confidence,
                handedness_name(ev.handedness),
                ev.ts_ms,
            )
        })
    }

    // ── 수명 ──

    /// 웜 리셋 — 스트림 리소스(파이프라인 res)와 시간 상태만 버리고 세션·모델·
    /// 컴파일 결과는 유지 (v-ai destroy 규약: 재활성화가 즉시 뜬다)
    pub fn detach(&mut self) {
        self.pipeline.invalidate();
        if let Some((task, ..)) = &mut self.face {
            task.reset();
        }
        if let Some((task, ..)) = &mut self.hand {
            task.reset();
        }
        if let Some((task, ..)) = &mut self.gaze {
            task.reset();
        }
        self.gestures.reset();
        self.gesture_queue.clear();
        self.last_face = None;
        self.last_focus = None;
        self.last_focus_ms = f64::NEG_INFINITY;
        self.last_hand_ms = f64::NEG_INFINITY;
    }

    /// 하드 리셋 — detach + 세션 전부 드랍 (GPU 메모리 반납). 모델 바이트는
    /// 유지 — 다음 프레임에 지연 재로드 (vcxrust destroy=리셋 규약의 우리판)
    pub fn reset(&mut self) {
        self.detach();
        self.seg = None;
        self.face = None;
        self.hand = None;
        self.gaze = None;
        self.pre = None;
        self.items = None;
        self.items_loaded.clear();
    }

    // ── 지연 로드 ──

    async fn ensure_seg(&mut self, ctx: &GpuContext) -> Result<(), TaskError> {
        if self.seg.is_none() {
            if let Some(bytes) = self.models.get("seg") {
                self.seg = Some(GpuSession::load(ctx, bytes).await?);
            }
        }
        Ok(())
    }

    /// face det+lm 준비 — 모델 미주입이면 false (조용히 스킵, 배선 진단은 호스트)
    async fn ensure_face(&mut self, ctx: &GpuContext) -> Result<bool, TaskError> {
        if self.face.is_none() {
            let (Some(det_b), Some(lm_b)) =
                (self.models.get("face_det"), self.models.get("face_lm"))
            else {
                return Ok(false);
            };
            let det = GpuSession::load(ctx, det_b).await?;
            let lm = GpuSession::load(ctx, lm_b).await?;
            self.face = Some((FaceTask::new(false), det, lm));
        }
        if self.pre.is_none() {
            self.pre = Some(GpuPre::new(ctx));
        }
        Ok(true)
    }

    async fn ensure_gaze(&mut self, ctx: &GpuContext) -> Result<bool, TaskError> {
        if self.gaze.is_none() {
            let Some(gaze_b) = self.models.get("gaze") else { return Ok(false) };
            let gaze_s = GpuSession::load(ctx, gaze_b).await?;
            let bs_s = match self.models.get("gaze_bs") {
                Some(b) => Some(GpuSession::load(ctx, b).await?),
                None => None, // bs 없으면 blink는 EAR 절반만 — 동작엔 지장 없음
            };
            self.gaze = Some((GazeTask::default(), gaze_s, bs_s));
        }
        Ok(true)
    }

    async fn ensure_hand(&mut self, ctx: &GpuContext) -> Result<bool, TaskError> {
        if self.hand.is_none() {
            let (Some(det_b), Some(lm_b)) =
                (self.models.get("hand_det"), self.models.get("hand_lm"))
            else {
                return Ok(false);
            };
            let det = GpuSession::load(ctx, det_b).await?;
            let lm = GpuSession::load(ctx, lm_b).await?;
            self.hand = Some((HandTask::new(2), det, lm));
        }
        Ok(true)
    }

    fn ensure_items(&mut self, ctx: &GpuContext, fi: &FaceItems) -> Result<(), TaskError> {
        if self.items.is_none() {
            self.items = Some(ItemsOverlay::new(ctx)?);
        }
        // GLB 지연 조달 (로더가 있을 때 — wasm은 set_item_glb 선주입이라 로더 없음)
        if let Some(loader) = &self.glb_loader {
            for kind in [&fi.hat, &fi.eyewear, &fi.beard] {
                if kind != "none" && !self.items_loaded.contains(kind.as_str()) {
                    // 실패도 기록 — 없는 에셋 재시도 스팸 방지 (vcxrust 규약)
                    if let Some(bytes) = loader(kind) {
                        self.items.as_mut().unwrap().preload_glb(ctx, kind, &bytes)?;
                    }
                    self.items_loaded.insert(kind.clone());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_patch_merge_semantics() {
        let ctx = match GpuContext::new_blocking() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skip: GPU 없음");
                return;
            }
        };
        let mut d = Director::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
        assert!(d.passthrough());
        d.apply_json(
            r##"{"handDetection":{"enabled":true,"detectFps":10},
                 "focusDetection":{"enabled":true},
                 "faceItems":{"enabled":true,"hat":"hat1"}}"##,
        )
        .unwrap();
        assert!(d.tasks_active());
        assert!(d.needs_render(), "아이템은 합성 타깃 필요");
        // 없음=유지
        d.apply_json(r#"{"blur":0.5}"#).unwrap();
        assert!(d.tasks_active());
        // null=해제
        d.apply_json(r#"{"handDetection":null,"faceItems":null,"focusDetection":null}"#)
            .unwrap();
        assert!(!d.tasks_active());
        assert!(d.needs_render(), "blur 0.5는 비디오 경로");
        d.apply_json(r#"{"blur":0}"#).unwrap();
        assert!(d.passthrough());
    }

    #[test]
    fn result_json_shapes() {
        let ctx = match GpuContext::new_blocking() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skip: GPU 없음");
                return;
            }
        };
        let mut d = Director::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
        let j = d.focus_json();
        assert!(j.contains("\"status\":\"INITIALIZING\""), "{j}");
        assert!(d.poll_gesture_json().is_none());
        d.gesture_queue.push_back(GestureEvent {
            gesture: Gesture::Clap,
            confidence: 0.9734,
            handedness: Handedness::Unknown,
            ts_ms: 123.45,
        });
        let g = d.poll_gesture_json().unwrap();
        assert_eq!(
            g,
            "{\"gesture\":\"clap\",\"confidence\":0.973,\"handedness\":\"unknown\",\"tsMs\":123.5}"
        );
    }
}
