//! GazeTask — 웹 focus-tracker analyze.ts 등가 오케스트레이션.
//! 호스트가 비전 틱(~10fps)마다 호출: 랜드마크(FaceTask 결과) + 프레임 RGB.
//! CNN은 내부 페이싱(83.3ms 하한 — 웹 GAZE_MODEL.minIntervalMs)으로만 돈다.
//! 웹과 달리 추론이 수 ms(GPU 3.9ms)라 비동기 오프루프 대신 동기 실행 —
//! 페이싱 규약(마지막 완료값 사용)은 동일하게 유지된다.

use ai_gpu::GpuContext;

use crate::error::TaskError;
use crate::session::gpu::GpuSession;
use super::one_euro::OneEuro2;
use super::preprocess::{crop_resize_rgb, decode_bins, ears_closed, face_crop_box, imagenet_normalize};
use super::state::{BaselineCollector, FocusFrame, FocusResult, FocusStateMachine, Gaze};

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

    /// 비전 틱 1회. landmarks = 정규화 478점 (없으면 얼굴 소실 프레임).
    #[allow(clippy::too_many_arguments)]
    pub async fn process_gpu(
        &mut self,
        ctx: &GpuContext,
        gaze: &mut GpuSession,
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
        Ok(self.machine.update(FocusFrame {
            ts_ms,
            face_count,
            gaze: gaze_v,
            eyes_closed: ears_closed(pts),
        }))
    }
}
