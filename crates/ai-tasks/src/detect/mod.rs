//! 디텍터 후처리 — 날 앵커 회귀/로짓 → 원본 프레임 좌표의 검출.
//!
//! MediaPipe calculator 그래프의 텐서 이후 구간(SsdAnchors → TensorsToDetections →
//! WeightedNMS → 레터박스 역투영)을 순수 Rust로 옮긴 것. 프리셋 상수는
//! mediapipe/modules의 .pbtxt에서 그대로 가져왔다 — 여기 값이 하나라도 다르면
//! 좌표 diff 게이트(web/demo/face-ab.html)가 잡는다.
//!
//! 백엔드(GPU/CPU)와 무관: 입력은 모델 출력 텐서(f32 슬라이스)뿐이다.

pub mod anchors;
pub mod decode;
pub mod letterbox;
pub mod nms;
pub mod roi;

use std::sync::OnceLock;

use crate::error::TaskError;
use anchors::{generate, Anchor, AnchorConfig};
use decode::DecodeOpts;
use letterbox::Letterbox;

/// 검출 하나 — 좌표는 정규화([0,1], 원점 좌상단). 키포인트 의미는 모델별
/// (face: 오른눈·왼눈·코끝·입·오른귀·왼귀 / palm: 손목 등 7점).
#[derive(Clone, Debug)]
pub struct Detection {
    pub score: f32,
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub keypoints: Vec<[f32; 2]>,
}

impl Detection {
    /// 레터박스 공간 → 원본 프레임 공간
    fn unletterbox(&mut self, lb: &Letterbox) {
        let (x0, y0) = lb.unproject(self.xmin, self.ymin);
        let (x1, y1) = lb.unproject(self.xmax, self.ymax);
        (self.xmin, self.ymin, self.xmax, self.ymax) = (x0, y0, x1, y1);
        for kp in &mut self.keypoints {
            let (x, y) = lb.unproject(kp[0], kp[1]);
            *kp = [x, y];
        }
    }
}

/// BlazeFace short-range (face_detection_short_range.pbtxt, 입력 128²)
pub const FACE_ANCHORS: AnchorConfig = AnchorConfig {
    input_w: 128,
    input_h: 128,
    min_scale: 0.1484375,
    max_scale: 0.75,
    strides: &[8, 16, 16, 16],
    aspect_ratios: &[1.0],
    anchor_offset_x: 0.5,
    anchor_offset_y: 0.5,
    interpolated_scale_aspect_ratio: 1.0,
    fixed_anchor_size: true,
};

/// palm_detection_full (입력 192²)
pub const PALM_ANCHORS: AnchorConfig = AnchorConfig {
    input_w: 192,
    input_h: 192,
    min_scale: 0.1484375,
    max_scale: 0.75,
    strides: &[8, 16, 16, 16],
    aspect_ratios: &[1.0],
    anchor_offset_x: 0.5,
    anchor_offset_y: 0.5,
    interpolated_scale_aspect_ratio: 1.0,
    fixed_anchor_size: true,
};

/// 디텍터 한 종의 후처리 전체 (앵커 + 디코드 옵션 + NMS 문턱)
pub struct DetectorPost {
    anchors: Vec<Anchor>,
    opts: DecodeOpts,
    /// IoU가 이걸 넘는 후보끼리 가중 평균 (MediaPipe min_suppression_threshold)
    nms_threshold: f32,
    input_w: u32,
    input_h: u32,
    /// 모델 입력 픽셀 범위 [lo, hi] — .pbtxt output_tensor_float_range.
    /// **디텍터마다 다르다**: face short-range [-1,1] / palm full [0,1].
    input_range: [f32; 2],
}

impl DetectorPost {
    /// BlazeFace short-range — face_landmarker.task의 디텍터와 동일 모델
    pub fn face_short_range() -> Self {
        DetectorPost {
            anchors: generate(&FACE_ANCHORS),
            opts: DecodeOpts {
                num_coords: 16,
                num_keypoints: 6,
                keypoint_offset: 4,
                x_scale: 128.0,
                y_scale: 128.0,
                w_scale: 128.0,
                h_scale: 128.0,
                score_clip: 100.0,
                min_score: 0.5,
            },
            nms_threshold: 0.3,
            input_w: 128,
            input_h: 128,
            input_range: [-1.0, 1.0],
        }
    }

    /// palm_detection_full — hand_landmarker.task의 디텍터
    pub fn palm_full() -> Self {
        DetectorPost {
            anchors: generate(&PALM_ANCHORS),
            opts: DecodeOpts {
                num_coords: 18,
                num_keypoints: 7,
                keypoint_offset: 4,
                x_scale: 192.0,
                y_scale: 192.0,
                w_scale: 192.0,
                h_scale: 192.0,
                score_clip: 100.0,
                min_score: 0.5,
            },
            nms_threshold: 0.3,
            input_w: 192,
            input_h: 192,
            input_range: [0.0, 1.0],
        }
    }

    /// 모델 입력 크기 (호스트가 레터박스 캔버스를 이 크기로 만든다)
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_w, self.input_h)
    }

    /// 모델 입력 픽셀 범위 [lo, hi] — 레터박스 정규화·패딩 값이 이걸 따른다
    pub fn input_range(&self) -> [f32; 2] {
        self.input_range
    }

    /// 모델 출력들(순서 무관)에서 박스/점수 텐서를 **원소 수로** 식별해 디코드+NMS.
    /// 이름 결합을 피하는 이유: tf2onnx가 출력 이름을 임의로 바꾼다
    /// (face는 regressors/classificators, hand는 Identity/Identity_1).
    pub fn run(&self, outputs: &[&[f32]]) -> Result<Vec<Detection>, TaskError> {
        let n = self.anchors.len();
        let boxes = outputs
            .iter()
            .find(|o| o.len() == n * self.opts.num_coords)
            .ok_or_else(|| {
                TaskError::Other(format!(
                    "박스 텐서({}×{}) 없음 — 출력 크기: {:?}",
                    n,
                    self.opts.num_coords,
                    outputs.iter().map(|o| o.len()).collect::<Vec<_>>()
                ))
            })?;
        let scores = outputs.iter().find(|o| o.len() == n).ok_or_else(|| {
            TaskError::Other(format!("점수 텐서({n}) 없음"))
        })?;
        let dets = decode::decode(&self.anchors, &self.opts, boxes, scores);
        Ok(nms::weighted_nms(dets, self.nms_threshold))
    }

    /// `run` + 레터박스 역투영 — 원본 프레임(src_w×src_h) 정규화 좌표로 반환
    pub fn run_projected(
        &self,
        outputs: &[&[f32]],
        src_w: f32,
        src_h: f32,
    ) -> Result<Vec<Detection>, TaskError> {
        let lb = Letterbox::fit(src_w, src_h, self.input_w as f32, self.input_h as f32);
        let mut dets = self.run(outputs)?;
        for d in &mut dets {
            d.unletterbox(&lb);
        }
        Ok(dets)
    }
}

/// 프리셋 캐시 — 바인딩이 프레임마다 앵커를 재생성하지 않게 한다
pub fn preset(name: &str) -> Result<&'static DetectorPost, TaskError> {
    static FACE: OnceLock<DetectorPost> = OnceLock::new();
    static PALM: OnceLock<DetectorPost> = OnceLock::new();
    match name {
        "face" => Ok(FACE.get_or_init(DetectorPost::face_short_range)),
        "palm" | "hand" => Ok(PALM.get_or_init(DetectorPost::palm_full)),
        _ => Err(TaskError::Other(format!("미지 디텍터 프리셋: {name} (face|palm)"))),
    }
}
