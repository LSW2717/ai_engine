//! FaceTask — 첫 **파이프라인 레벨 태스크** (이름 규칙: `~Session`=모델 인스턴스,
//! `~Task`=모델 여러 개+전·후처리+트래킹을 묶은 도메인 기능).
//!
//! MediaPipe face_landmarker 그래프 등가의 프레임 루프:
//!   이전 프레임 ROI 있음 → 디텍터 **생략**(트래킹) / 없음 → 검출→ROI
//!   → 회전 정규화 크롭(256², [0,1]) → face_landmarks → presence 게이트
//!   → 역투영 → (옵션) OneEuroFilter → 다음 프레임 ROI를 랜드마크로 갱신.
//!
//! 세션(det/lm)은 **밖에서 주입**받는다: 모델 수명·백엔드 선택은 호스트/워커
//! 몫이고 (GPU 강등 정책), 이 타입은 상태(prev ROI, 필터)와 수학만 소유한다.
//! CPU(동기)/GPU(비동기) 드라이버가 같은 내부 단계를 공유하므로 웹·모바일·
//! 네이티브가 한 벌이다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::detect::gpu::GpuPre;
use crate::session::cpu::CpuSession;
use crate::detect::letterbox::letterbox_u8_rgb;
use crate::detect::roi::{crop_u8_rgb, project_landmarks, roi_from_detection, roi_from_landmarks, Roi};
use crate::detect::{self, Detection, DetectorPost};
use crate::error::TaskError;
use crate::features::face::smoothing::LandmarkSmoother;
use crate::session::gpu::GpuSession;

/// 한 프레임의 얼굴 결과 — 좌표는 원본 프레임 정규화 [x,y,z] (z는 ROI 폭 스케일)
#[derive(Clone, Debug)]
pub struct FaceResult {
    pub points: Vec<[f32; 3]>,
    pub presence: f32,
    /// 이번 랜드마크 추론에 쓴 크롭 (절대 px — 디버그·시각화용)
    pub roi: Roi,
}

/// face_landmarker 그래프 상수 (mediapipe/modules/face_landmark 기준)
const ROI_SCALE: f32 = 1.5;
/// 검출 → 회전: 키포인트 0,1 = 양눈
const DET_ROT_KP: (usize, usize) = (0, 1);
/// 랜드마크 → 다음 ROI 회전: 33,263 = 양눈 바깥꼬리
const LM_ROT_KP: (usize, usize) = (33, 263);
const PRESENCE_MIN: f32 = 0.5;

pub struct FaceTask {
    post: &'static DetectorPost,
    prev_roi: Option<Roi>,
    smoother: Option<LandmarkSmoother>,
    /// 감시할 최대 얼굴 수 — 1이면 트래킹 중 디텍터 생략(기존 동작), ≥2면
    /// **매 프레임 디텍터로 얼굴 수 감시** (MediaPipe FaceLandmarker가
    /// tracked<num_faces인 동안 검출을 계속 돌리는 것과 같은 규약 —
    /// 집중도 MULTIPLE_FACES의 근거). 랜드마크·결과는 최고점 1명 유지
    /// (웹 analyze.ts도 faces[0]만 분석한다).
    num_faces: usize,
    last_count: usize,
}

impl FaceTask {
    /// smoothing: OneEuroFilter 적용 여부 (파라미터는 VIDEO 게이트 검증 전 — filter.rs)
    pub fn new(smoothing: bool) -> Self {
        FaceTask {
            post: detect::preset("face").expect("face 프리셋"),
            prev_roi: None,
            smoother: smoothing.then(LandmarkSmoother::face_default),
            num_faces: 1,
            last_count: 0,
        }
    }

    /// 얼굴 수 감시 설정 (집중도 켤 때 2 — 끄면 1로 되돌려 디텍터 비용 제거)
    pub fn set_num_faces(&mut self, n: usize) {
        self.num_faces = n.max(1);
    }

    /// 최근 프레임의 얼굴 수 — num_faces==1이면 0/1(트래킹 결과), ≥2면 디텍터
    /// 검출 수(post-NMS, score≥0.5). 웹은 landmarker 통과 수를 세지만 우리는
    /// 랜드마크가 1명뿐이라 디텍터 수로 근사 (문턱·NMS는 동일 프리셋).
    pub fn face_count(&self) -> usize {
        self.last_count
    }

    /// 트래킹 상태 폐기 — 다음 프레임은 검출부터
    pub fn reset(&mut self) {
        self.prev_roi = None;
        self.last_count = 0;
        if let Some(s) = &mut self.smoother {
            s.reset();
        }
    }

    /// 검출 결과(점수 내림차순)에서 이번 프레임 ROI — 최고점 1명 (num_faces=1)
    fn roi_from_dets(&self, dets: &[Detection], img_w: f32, img_h: f32) -> Option<Roi> {
        let d = dets.first()?;
        Some(roi_from_detection(d, DET_ROT_KP.0, DET_ROT_KP.1, ROI_SCALE, img_w, img_h))
    }

    /// 랜드마크 원시 출력 → 최종 결과 + 트래킹 상태 갱신 (백엔드 공통 후반부).
    /// outputs: lm 모델 전 출력 (순서 보존) — 최대 길이=랜드마크(px 좌표,
    /// /입력크기), 그 외 첫 1원소=presence 로짓(sigmoid 전) — tf2onnx가 이름을
    /// 바꾸므로 크기·순서로 식별한다.
    fn finish(
        &mut self,
        outputs: &[Vec<f32>],
        lm_input: f32,
        roi: Roi,
        img_w: f32,
        img_h: f32,
        t_ms: f64,
    ) -> Result<Option<FaceResult>, TaskError> {
        let lm_raw = outputs
            .iter()
            .max_by_key(|o| o.len())
            .filter(|o| o.len() >= 9 && o.len() % 3 == 0)
            .ok_or_else(|| TaskError::Other("랜드마크 텐서(3N) 없음".into()))?;
        let presence_raw = outputs
            .iter()
            .find(|o| o.len() == 1)
            .ok_or_else(|| TaskError::Other("presence 텐서(1) 없음".into()))?[0];
        let presence = 1.0 / (1.0 + (-presence_raw).exp());
        if presence < PRESENCE_MIN {
            // 얼굴 놓침 — 트래킹 폐기, 다음 프레임은 검출부터
            self.reset();
            return Ok(None);
        }
        let mut points: Vec<[f32; 3]> = lm_raw
            .chunks_exact(3)
            .map(|p| [p[0] / lm_input, p[1] / lm_input, p[2] / lm_input])
            .collect();
        project_landmarks(&mut points, &roi, img_w, img_h);
        if let Some(s) = &mut self.smoother {
            s.apply(t_ms, img_w, img_h, &mut points);
        }
        self.prev_roi = Some(roi_from_landmarks(
            &points, LM_ROT_KP.0, LM_ROT_KP.1, ROI_SCALE, img_w, img_h,
        ));
        Ok(Some(FaceResult { points, presence, roi }))
    }

    /// CPU 한 프레임 (동기): u8 RGB 프레임 → 얼굴 랜드마크.
    /// None = 얼굴 없음 (검출 실패 또는 presence 미달 — 다음 프레임은 검출부터).
    pub fn process_cpu(
        &mut self,
        det: &mut CpuSession,
        lm: &mut CpuSession,
        frame: &[u8],
        img_w: u32,
        img_h: u32,
        t_ms: f64,
    ) -> Result<Option<FaceResult>, TaskError> {
        let (w, h) = (img_w as f32, img_h as f32);
        // 검출 — ROI 재획득(트래킹 끊김) 또는 얼굴 수 감시(num_faces≥2)
        let mut acquired: Option<Roi> = None;
        let mut det_count: Option<usize> = None;
        if self.prev_roi.is_none() || self.num_faces >= 2 {
            let (iw, ih) = self.post.input_size();
            let [lo, hi] = self.post.input_range();
            let input = letterbox_u8_rgb(
                frame, img_w as usize, img_h as usize, iw as usize, ih as usize, lo, hi,
            );
            let dets = det.detect(self.post, &input, img_w, img_h)?;
            det_count = Some(dets.len());
            if self.prev_roi.is_none() {
                acquired = self.roi_from_dets(&dets, w, h);
                if acquired.is_none() {
                    self.last_count = 0;
                    return Ok(None);
                }
            }
        }
        let roi = self.prev_roi.or(acquired).unwrap();
        let (lm_input, names) = {
            let sw = lm.model().sw();
            let names: Vec<String> =
                sw.outputs.iter().map(|&o| sw.tensors[o as usize].name.clone()).collect();
            (sw.tensors[sw.inputs[0] as usize].h as f32, names)
        };
        let crop = crop_u8_rgb(frame, img_w as usize, img_h as usize, &roi, lm_input as usize);
        lm.infer_frame(&crop)?;
        let outputs: Vec<Vec<f32>> =
            names.iter().map(|n| lm.read_output(n)).collect::<Result<_, _>>()?;
        let result = self.finish(&outputs, lm_input, roi, w, h, t_ms)?;
        self.last_count = det_count.unwrap_or(0).max(result.is_some() as usize);
        Ok(result)
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
        t_ms: f64,
    ) -> Result<Option<FaceResult>, TaskError> {
        let (w, h) = (img_w as f32, img_h as f32);
        let mut acquired: Option<Roi> = None;
        let mut det_count: Option<usize> = None;
        if self.prev_roi.is_none() || self.num_faces >= 2 {
            let (iw, ih) = self.post.input_size();
            let [lo, hi] = self.post.input_range();
            let input = letterbox_u8_rgb(
                frame, img_w as usize, img_h as usize, iw as usize, ih as usize, lo, hi,
            );
            let dets = det.detect(ctx, self.post, &input, img_w, img_h).await?;
            det_count = Some(dets.len());
            if self.prev_roi.is_none() {
                acquired = self.roi_from_dets(&dets, w, h);
                if acquired.is_none() {
                    self.last_count = 0;
                    return Ok(None);
                }
            }
        }
        let roi = self.prev_roi.or(acquired).unwrap();
        let lm_input = lm.model().sw.tensors[lm.model().sw.inputs[0] as usize].h as f32;
        let crop = crop_u8_rgb(frame, img_w as usize, img_h as usize, &roi, lm_input as usize);
        lm.upload(ctx, &crop)?;
        lm.infer(ctx).await?;
        let names: Vec<String> = lm
            .model()
            .sw
            .outputs
            .iter()
            .map(|&o| lm.model().sw.tensors[o as usize].name.clone())
            .collect();
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
        for n in &names {
            outputs.push(lm.read_output(ctx, n).await?);
        }
        lm.finish_frame(ctx).await?;
        let result = self.finish(&outputs, lm_input, roi, w, h, t_ms)?;
        self.last_count = det_count.unwrap_or(0).max(result.is_some() as usize);
        Ok(result)
    }

    /// GPU 텍스처 한 프레임 — process_gpu와 같은 단계지만 **픽셀이 CPU를 거치지
    /// 않는다**: 레터박스·크롭이 `GpuPre` 컴퓨트 커널로 모델 입력 버퍼에 직결된다
    /// (남는 CPU 전송은 uniform 몇십 B + 소형 출력 리드백 — 후자는 ROI 제어
    /// 흐름상 구조적으로 불가피, NEXT.md "GPU↔CPU 왕복" 문단).
    /// `frame`은 img_w×img_h Rgba8Unorm 텍스처 뷰 — 스탠드얼론이면 `pre.frame`,
    /// studio면 파이프라인 프레임 텍스처를 넘긴다.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_tex(
        &mut self,
        ctx: &GpuContext,
        pre: &GpuPre,
        frame: &wgpu::TextureView,
        det: &mut GpuSession,
        lm: &mut GpuSession,
        img_w: u32,
        img_h: u32,
        t_ms: f64,
    ) -> Result<Option<FaceResult>, TaskError> {
        let (w, h) = (img_w as f32, img_h as f32);
        let mut acquired: Option<Roi> = None;
        let mut det_count: Option<usize> = None;
        if self.prev_roi.is_none() || self.num_faces >= 2 {
            let [lo, hi] = self.post.input_range();
            {
                let name = det.input_name().to_string();
                let (buf, desc) = det.input_storage(&name).ok_or_else(|| {
                    TaskError::Other(format!("det 입력 버퍼 없음: {name}"))
                })?;
                pre.letterbox_into(ctx, frame, img_w, img_h, buf, &desc, lo, hi)?;
            }
            let dets = det.detect_uploaded(ctx, self.post, img_w, img_h).await?;
            det_count = Some(dets.len());
            if self.prev_roi.is_none() {
                acquired = self.roi_from_dets(&dets, w, h);
                if acquired.is_none() {
                    self.last_count = 0;
                    return Ok(None);
                }
            }
        }
        let roi = self.prev_roi.or(acquired).unwrap();
        let lm_input = {
            let name = lm.input_name().to_string();
            let (buf, desc) = lm
                .input_storage(&name)
                .ok_or_else(|| TaskError::Other(format!("lm 입력 버퍼 없음: {name}")))?;
            pre.crop_into(ctx, frame, img_w, img_h, &roi, buf, &desc, 0.0, 1.0)?;
            desc.h as f32
        };
        lm.infer(ctx).await?;
        let names: Vec<String> = lm
            .model()
            .sw
            .outputs
            .iter()
            .map(|&o| lm.model().sw.tensors[o as usize].name.clone())
            .collect();
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
        for n in &names {
            outputs.push(lm.read_output(ctx, n).await?);
        }
        lm.finish_frame(ctx).await?;
        let result = self.finish(&outputs, lm_input, roi, w, h, t_ms)?;
        self.last_count = det_count.unwrap_or(0).max(result.is_some() as usize);
        Ok(result)
    }

    /// 트래킹 중인가 (다음 프레임에 디텍터를 건너뛰는가) — 진단용
    pub fn is_tracking(&self) -> bool {
        self.prev_roi.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// finish()의 출력 식별·presence 게이트·투영을 합성 데이터로 검증
    #[test]
    fn finish_gates_on_presence_and_projects() {
        let mut task = FaceTask::new(false);
        let roi = Roi { cx: 50.0, cy: 50.0, w: 50.0, h: 50.0, rotation: 0.0 };
        // 랜드마크 478개: 전부 크롭 중앙(128,128,0) — 원본 정규화 (0.5,0.5)
        let mut lm = vec![0.0f32; 1434];
        for p in lm.chunks_exact_mut(3) {
            (p[0], p[1]) = (128.0, 128.0);
        }
        let outputs = vec![lm, vec![10.0], vec![0.0]]; // presence 로짓 10 → ~1.0
        let r = task
            .finish(&outputs, 256.0, roi, 100.0, 100.0, 0.0)
            .unwrap()
            .expect("얼굴 있음");
        assert_eq!(r.points.len(), 478);
        assert!((r.points[0][0] - 0.5).abs() < 1e-5);
        assert!(r.presence > 0.99);
        assert!(task.is_tracking());

        // presence 미달 → None + 트래킹 폐기
        let outputs = vec![vec![0.0f32; 1434], vec![-10.0], vec![0.0]];
        let r = task.finish(&outputs, 256.0, roi, 100.0, 100.0, 33.0).unwrap();
        assert!(r.is_none());
        assert!(!task.is_tracking());
    }
}
