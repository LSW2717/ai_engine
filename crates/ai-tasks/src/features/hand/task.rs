//! HandTask — MediaPipe hand_landmarker 그래프 등가의 프레임 루프 (num_hands≤2).
//!
//! FaceTask와 같은 구조에 **다중 손 트래킹**이 얹힌다 (hand_landmarker_graph.cc):
//!   트래킹 ROI 수 == num_hands → 디텍터 **생략** / 부족 → 팜 검출 →
//!   HandAssociation(기존 ROI와 축정렬 IoU > 0.5인 검출은 버림) → num_hands 클립
//!   → ROI별: 회전 크롭(224², [0,1]) → hand_landmarks → presence 게이트 →
//!   역투영 → 다음 ROI를 랜드마크로 갱신 (roi.rs — MediaPipe quirk 2개 보존).
//!
//! 출력 4텐서 식별 (tf2onnx가 이름을 바꾸므로 크기·순서·크기규모로):
//!   63짜리 2개 = 화면 lm(px 좌표, max|·|가 큰 쪽) + 월드 lm(미터) /
//!   1짜리 2개 = 순서대로 presence, handedness — 모델이 sigmoid를 내장하므로
//!   **로짓이 아니라 확률**이다 (face와 다름 — 그래프의 TensorsToFloats가
//!   activation 없이 그대로 쓰고 ThresholdingCalculator로 자른다).
//!   ⚠ handedness **원시값은 P(Right)** — TensorsToClassification binary가
//!   index 0에 raw score를 주고 label_items[0]="Right"라서. HandResult 필드는
//!   1−raw = P(Left)로 통일한다 (hand-ab 게이트가 라벨 뒤집힘을 잡았다).

use ai_gpu::GpuContext;

use crate::detect::letterbox::letterbox_u8_rgb;
use crate::detect::roi::{crop_u8_rgb, project_landmarks, Roi};
use crate::detect::{self, Detection, DetectorPost};
use crate::error::TaskError;
use crate::session::cpu::CpuSession;
use crate::session::gpu::GpuSession;
use super::roi::{iou_axis_aligned, roi_from_hand_landmarks, roi_from_palm_detection};

/// 한 손의 결과 — 좌표는 원본 프레임 정규화 [x,y,z] (z는 ROI 폭 스케일)
#[derive(Clone, Debug)]
pub struct HandResult {
    pub points: Vec<[f32; 3]>,
    /// 월드 랜드마크 (미터, 손 중심 원점) — ROI 회전만 역적용
    pub world: Vec<[f32; 3]>,
    /// 손 존재 확률 [0,1] (모델 sigmoid 내장)
    pub presence: f32,
    /// P(Left) [0,1] — >0.5면 Left (MediaPipe 라벨 규약)
    pub handedness: f32,
    pub roi: Roi,
}

/// min_hand_presence_confidence 기본값 (tasks-vision)
const PRESENCE_MIN: f32 = 0.5;
/// HandAssociation min_similarity_threshold = min_tracking_confidence 기본값
const ASSOC_IOU: f32 = 0.5;

pub struct HandTask {
    post: &'static DetectorPost,
    num_hands: usize,
    prev_rois: Vec<Roi>,
}

impl HandTask {
    pub fn new(num_hands: usize) -> Self {
        HandTask {
            post: detect::preset("palm").expect("palm 프리셋"),
            num_hands: num_hands.max(1),
            prev_rois: Vec::new(),
        }
    }

    /// 트래킹 상태 폐기 — 다음 프레임은 검출부터
    pub fn reset(&mut self) {
        self.prev_rois.clear();
    }

    /// 트래킹 중인 손 수 — 진단용
    pub fn tracking(&self) -> usize {
        self.prev_rois.len()
    }

    /// 검출 결과를 기존 트래킹 ROI에 합류 (HandAssociation + 클립).
    /// dets는 점수 내림차순으로 소비한다.
    fn merge_detections(&self, rois: &mut Vec<Roi>, dets: &[Detection], w: f32, h: f32) {
        let mut order: Vec<&Detection> = dets.iter().collect();
        order.sort_by(|a, b| b.score.total_cmp(&a.score));
        for d in order {
            if rois.len() >= self.num_hands {
                break;
            }
            let r = roi_from_palm_detection(d, w, h);
            if rois.iter().all(|e| iou_axis_aligned(e, &r, w, h) <= ASSOC_IOU) {
                rois.push(r);
            }
        }
    }

    /// 랜드마크 원시 출력 → 결과 1손 (presence 게이트 통과 시).
    fn finish_one(
        outputs: &[Vec<f32>],
        lm_input: f32,
        roi: Roi,
        img_w: f32,
        img_h: f32,
    ) -> Result<Option<HandResult>, TaskError> {
        let lms: Vec<&Vec<f32>> =
            outputs.iter().filter(|o| o.len() >= 63 && o.len() % 3 == 0).collect();
        let scalars: Vec<f32> =
            outputs.iter().filter(|o| o.len() == 1).map(|o| o[0]).collect();
        if lms.len() != 2 || scalars.len() != 2 {
            return Err(TaskError::Other(format!(
                "hand lm 출력 식별 실패 (63×{} / 1×{})",
                lms.len(),
                scalars.len()
            )));
        }
        // handedness 원시값 = P(Right): TensorsToClassification binary가 index 0에
        // raw score를 주고 label_items[0]="Right"다 (게이트가 잡은 뒤집힘).
        // 필드는 P(Left)로 통일 — MediaPipe categoryName과 >0.5 판정이 일치한다.
        let (presence, handedness) = (scalars[0], 1.0 - scalars[1]);
        if presence < PRESENCE_MIN {
            return Ok(None);
        }
        let amp = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let (screen, world_raw) = if amp(lms[0]) >= amp(lms[1]) {
            (lms[0], lms[1])
        } else {
            (lms[1], lms[0])
        };
        let mut points: Vec<[f32; 3]> = screen
            .chunks_exact(3)
            .map(|p| [p[0] / lm_input, p[1] / lm_input, p[2] / lm_input])
            .collect();
        project_landmarks(&mut points, &roi, img_w, img_h);
        // WorldLandmarkProjectionCalculator — 회전만 적용, z 불변
        let (sinr, cosr) = roi.rotation.sin_cos();
        let world: Vec<[f32; 3]> = world_raw
            .chunks_exact(3)
            .map(|p| [cosr * p[0] - sinr * p[1], sinr * p[0] + cosr * p[1], p[2]])
            .collect();
        Ok(Some(HandResult { points, world, presence, handedness, roi }))
    }

    /// 이번 프레임에 쓸 ROI 목록 (검출은 클로저 — 트래킹이 충분하면 안 불린다)
    fn frame_rois<E>(
        &mut self,
        w: f32,
        h: f32,
        detect: impl FnOnce() -> Result<Vec<Detection>, E>,
    ) -> Result<Vec<Roi>, E> {
        let mut rois = self.prev_rois.clone();
        if rois.len() < self.num_hands {
            self.merge_detections(&mut rois, &detect()?, w, h);
        }
        Ok(rois)
    }

    /// CPU 한 프레임 (동기): u8 RGB 프레임 → 손 목록 (0~num_hands개)
    pub fn process_cpu(
        &mut self,
        det: &mut CpuSession,
        lm: &mut CpuSession,
        frame: &[u8],
        img_w: u32,
        img_h: u32,
        _t_ms: f64,
    ) -> Result<Vec<HandResult>, TaskError> {
        let (w, h) = (img_w as f32, img_h as f32);
        let (iw, ih) = self.post.input_size();
        let [lo, hi] = self.post.input_range();
        let post = self.post;
        let rois = self.frame_rois(w, h, || {
            let input = letterbox_u8_rgb(
                frame, img_w as usize, img_h as usize, iw as usize, ih as usize, lo, hi,
            );
            det.detect(post, &input, img_w, img_h)
        })?;
        let (lm_input, names) = {
            let sw = lm.model().sw();
            let names: Vec<String> =
                sw.outputs.iter().map(|&o| sw.tensors[o as usize].name.clone()).collect();
            (sw.tensors[sw.inputs[0] as usize].h as f32, names)
        };
        let mut results = Vec::new();
        let mut next = Vec::new();
        for roi in rois {
            let crop =
                crop_u8_rgb(frame, img_w as usize, img_h as usize, &roi, lm_input as usize);
            lm.infer_frame(&crop)?;
            let outputs: Vec<Vec<f32>> =
                names.iter().map(|n| lm.read_output(n)).collect::<Result<_, _>>()?;
            if let Some(r) = Self::finish_one(&outputs, lm_input, roi, w, h)? {
                next.push(roi_from_hand_landmarks(&r.points, w, h));
                results.push(r);
            }
        }
        self.prev_rois = next;
        Ok(results)
    }

    /// GPU 한 프레임 (비동기) — process_cpu와 같은 단계, 세션 호출만 다르다
    pub async fn process_gpu(
        &mut self,
        ctx: &GpuContext,
        det: &mut GpuSession,
        lm: &mut GpuSession,
        frame: &[u8],
        img_w: u32,
        img_h: u32,
        _t_ms: f64,
    ) -> Result<Vec<HandResult>, TaskError> {
        let (w, h) = (img_w as f32, img_h as f32);
        // 검출 (async라 frame_rois 클로저로 못 넘긴다 — 같은 순서를 펼쳐 쓴다)
        let mut rois = self.prev_rois.clone();
        if rois.len() < self.num_hands {
            let (iw, ih) = self.post.input_size();
            let [lo, hi] = self.post.input_range();
            let input = letterbox_u8_rgb(
                frame, img_w as usize, img_h as usize, iw as usize, ih as usize, lo, hi,
            );
            let dets = det.detect(ctx, self.post, &input, img_w, img_h).await?;
            self.merge_detections(&mut rois, &dets, w, h);
        }
        let lm_input = lm.model().sw.tensors[lm.model().sw.inputs[0] as usize].h as f32;
        let names: Vec<String> = lm
            .model()
            .sw
            .outputs
            .iter()
            .map(|&o| lm.model().sw.tensors[o as usize].name.clone())
            .collect();
        let mut results = Vec::new();
        let mut next = Vec::new();
        for roi in rois {
            let crop =
                crop_u8_rgb(frame, img_w as usize, img_h as usize, &roi, lm_input as usize);
            lm.upload(ctx, &crop)?;
            lm.infer(ctx).await?;
            let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
            for n in &names {
                outputs.push(lm.read_output(ctx, n).await?);
            }
            lm.finish_frame(ctx).await?;
            if let Some(r) = Self::finish_one(&outputs, lm_input, roi, w, h)? {
                next.push(roi_from_hand_landmarks(&r.points, w, h));
                results.push(r);
            }
        }
        self.prev_rois = next;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lm63(scale: f32) -> Vec<f32> {
        // 21점: 크롭 좌표계에서 손 모양 대충 — 값 크기(px)가 월드와 구분되게
        (0..63).map(|i| (i as f32 % 21.0) * scale + 30.0 * scale).collect()
    }

    #[test]
    fn finish_identifies_outputs_and_gates() {
        let roi = Roi { cx: 50.0, cy: 50.0, w: 50.0, h: 50.0, rotation: 0.0 };
        // 순서: [화면 lm(px), presence, handedness, 월드(작은 값)] — .sw 출력 순서
        let outputs =
            vec![lm63(1.0), vec![0.93], vec![0.8], lm63(0.001)];
        let r = HandTask::finish_one(&outputs, 224.0, roi, 100.0, 100.0)
            .unwrap()
            .expect("presence 통과");
        assert_eq!(r.points.len(), 21);
        assert_eq!(r.world.len(), 21);
        assert!((r.presence - 0.93).abs() < 1e-6);
        // 원시 0.8 = P(Right) → 필드는 P(Left) = 0.2
        assert!((r.handedness - 0.2).abs() < 1e-6);
        // 월드가 화면과 뒤바뀌지 않았다 (크기 규모)
        assert!(r.world[0][0].abs() < 1.0);

        // presence 미달 → None
        let outputs = vec![lm63(1.0), vec![0.2], vec![0.8], lm63(0.001)];
        assert!(HandTask::finish_one(&outputs, 224.0, roi, 100.0, 100.0).unwrap().is_none());
    }

    #[test]
    fn association_drops_overlapping_detection() {
        let task = HandTask::new(2);
        let mk = |x: f32, score: f32| Detection {
            score,
            xmin: x,
            ymin: 0.4,
            xmax: x + 0.2,
            ymax: 0.6,
            keypoints: vec![[x + 0.1, 0.58], [0.0, 0.0], [x + 0.1, 0.42], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]],
        };
        // 기존 트래킹 1손 (검출 0.3 위치와 겹치게)
        let tracked = roi_from_palm_detection(&mk(0.3, 1.0), 100.0, 100.0);
        let mut rois = vec![tracked];
        // 겹치는 검출(0.3) + 떨어진 검출(0.7): 겹치는 쪽만 버려져야 한다
        task.merge_detections(&mut rois, &[mk(0.3, 0.9), mk(0.7, 0.8)], 100.0, 100.0);
        assert_eq!(rois.len(), 2);
        assert!((rois[1].cx - 80.0).abs() < 5.0, "cx={}", rois[1].cx);
    }
}
