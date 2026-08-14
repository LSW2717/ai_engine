//! GazeTask — 웹 focus-tracker analyze.ts 등가 오케스트레이션.
//! 호스트가 비전 틱(~10fps)마다 호출: 랜드마크(FaceTask 결과) + 프레임 RGB.
//! CNN은 내부 페이싱(83.3ms 하한 — 웹 GAZE_MODEL.minIntervalMs)으로만 돈다.
//! 웹과 달리 추론이 수 ms(GPU 3.9ms)라 비동기 오프루프 대신 동기 실행 —
//! 페이싱 규약(마지막 완료값 사용)은 동일하게 유지된다.

use ai_gpu::GpuContext;

use crate::error::TaskError;
use crate::features::face::blendshapes;
use crate::session::gpu::GpuSession;
use super::one_euro::OneEuro2;
use super::preprocess::{crop_resize_rgb, decode_bins, ears_closed, face_crop_box, imagenet_normalize};
use super::state::{BaselineCollector, FocusFrame, FocusResult, FocusStateMachine, Gaze, ScreenLayout};

const CNN_MIN_INTERVAL_MS: f64 = 1000.0 / 12.0;
const INPUT: usize = 448;

pub struct GazeTask {
    filter: OneEuro2,
    pub machine: FocusStateMachine,
    baseline: BaselineCollector,
    buf: Vec<f32>,
    last_cnn_ms: f64,
    last_gaze: Option<Gaze>,
    /// 마지막 틱의 필터 후 각도 — HUD/게이트 노출용 (얼굴 소실 틱은 None)
    pub last_filtered: Option<Gaze>,
    pub auto_baseline: bool,
}

impl Default for GazeTask {
    fn default() -> Self {
        GazeTask {
            filter: OneEuro2::default(),
            machine: FocusStateMachine::default(),
            baseline: BaselineCollector::default(),
            buf: vec![0f32; INPUT * INPUT * 3],
            last_cnn_ms: f64::NEG_INFINITY,
            last_gaze: None,
            last_filtered: None,
            auto_baseline: true,
        }
    }
}

impl GazeTask {
    pub fn reset(&mut self) {
        self.filter.reset();
        self.last_gaze = None;
        self.last_filtered = None;
        self.machine = FocusStateMachine::default();
        self.baseline = BaselineCollector::default();
        self.last_cnn_ms = f64::NEG_INFINITY;
    }

    /// 마지막 CNN 원시 각도 (필터 전) — 게이트/진단용
    pub fn last_cnn(&self) -> Option<Gaze> {
        self.last_gaze
    }

    /// JSON 레이아웃 설정 — {monitors:[{index,left,top,width,height,yawDeg}],
    /// targetIndex} 또는 null (바인딩용 — 웹 getScreenDetails 산출물)
    pub fn set_layout_json(&mut self, json: &str) -> Result<(), String> {
        let layout: Option<ScreenLayout> =
            serde_json::from_str(json).map_err(|e| format!("레이아웃 파싱: {e}"))?;
        self.set_layout(layout);
        Ok(())
    }

    /// 다중 모니터 레이아웃 설정 (None = 단일 모니터 폴백).
    /// 타깃 모니터가 바뀌면 baseline을 리셋한다 (웹 tracker.ts 규약 —
    /// 시선 원점이 물리적으로 이동했으므로 재수집).
    pub fn set_layout(&mut self, layout: Option<ScreenLayout>) {
        let prev = self.machine.layout.as_ref().map(|l| l.target_index);
        let next = layout.as_ref().map(|l| l.target_index);
        self.machine.layout = layout;
        if prev.is_some() && next.is_some() && prev != next {
            self.machine.baseline = None;
            self.baseline = BaselineCollector::default();
        }
    }

    /// 비전 틱 1회. landmarks = 정규화 478점 (없으면 얼굴 소실 프레임).
    /// bs = face_blendshapes 세션 (Some이면 blink가 EAR ∨ 블렌드셰이프 —
    /// 웹 규약; None이면 EAR 절반만으로 동작).
    #[allow(clippy::too_many_arguments)]
    pub async fn process_gpu(
        &mut self,
        ctx: &GpuContext,
        gaze: &mut GpuSession,
        mut bs: Option<&mut GpuSession>,
        rgb: &[u8],
        w: usize,
        h: usize,
        landmarks: Option<&[[f32; 2]]>,
        face_count: usize,
        ts_ms: f64,
    ) -> Result<FocusResult, TaskError> {
        let Some(pts) = landmarks else {
            // 얼굴 소실 — 웹 analyze.ts:106: 필터 리셋 (cnnGaze는 유지)
            self.filter.reset();
            self.last_filtered = None;
            return Ok(self.machine.update(FocusFrame {
                ts_ms,
                face_count: 0,
                gaze: None,
                eyes_closed: false,
            }));
        };

        // CNN 페이싱 (83.3ms 하한 — 웹은 visionFps=10이라 실효 미바인딩, 규약 유지)
        if ts_ms - self.last_cnn_ms >= CNN_MIN_INTERVAL_MS {
            if let Some(cb) = face_crop_box(pts, w as f32, h as f32) {
                self.last_cnn_ms = ts_ms;
                crop_resize_rgb(rgb, w, h, cb, INPUT, &mut self.buf);
                imagenet_normalize(&mut self.buf);
                gaze.upload(ctx, &self.buf)?;
                gaze.infer(ctx).await?;
                // 출력은 이름 매칭 (웹 gazeModel.ts — 순서 무관)
                let yaw_l = gaze.read_output(ctx, "yaw").await?;
                let pitch_l = gaze.read_output(ctx, "pitch").await?;
                // 리드백이 이미 동기화 — finish_frame은 프레임타임·frames 기록용
                // (안 부르면 stats가 0으로 남아 감사·강등 판정 입력이 죽는다)
                gaze.finish_frame(ctx).await?;
                self.last_gaze = Some(Gaze {
                    yaw: decode_bins(&yaw_l),
                    pitch: decode_bins(&pitch_l),
                });
            }
        }

        let gaze_v = self.last_gaze.map(|g| Gaze {
            yaw: self.filter.yaw.filter(g.yaw, ts_ms),
            pitch: self.filter.pitch.filter(g.pitch, ts_ms),
        });
        self.last_filtered = gaze_v;
        // baseline 자동 수집 (미설정 시)
        if self.auto_baseline && self.machine.baseline.is_none() {
            if let Some(b) = self.baseline.feed(true, gaze_v) {
                self.machine.baseline = Some(b);
            }
        }
        // blink 블렌드셰이프 절반 — 웹 blink.ts: (bsL≥0.55 AND bsR≥0.55) OR EAR.
        // 입력 규약은 blendshapes.rs 헤더 (146 서브셋 × 프레임 px).
        let mut bs_closed = false;
        if let Some(s) = bs.as_deref_mut() {
            if let Some(input) = blendshapes::input_from_landmarks(pts, w as f32, h as f32) {
                s.upload(ctx, &input)?;
                s.infer(ctx).await?;
                let out_name =
                    s.model().sw.tensors[s.model().sw.outputs[0] as usize].name.clone();
                let coeffs = s.read_output(ctx, &out_name).await?;
                s.finish_frame(ctx).await?;
                bs_closed = blendshapes::blink_closed(&coeffs);
            }
        }
        Ok(self.machine.update(FocusFrame {
            ts_ms,
            face_count,
            gaze: gaze_v,
            eyes_closed: ears_closed(pts) || bs_closed,
        }))
    }
}
