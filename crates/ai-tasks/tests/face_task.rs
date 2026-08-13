//! FaceTask e2e (CPU) — 실프레임에서 검출→ROI→크롭→랜드마크→역투영 전체와
//! **트래킹 계약**(2프레임째는 디텍터 생략, 얼굴 놓치면 검출로 복귀)을 검증한다.
//!
//! 좌표의 정확한 파리티는 브라우저 게이트(face-ab.html의 landmarks 스테이지 —
//! MediaPipe FaceLandmarker와 478점 px diff)가 담당. 모델 없으면 스킵.

use ai_tasks::{CpuSession, FaceTask};

const DET: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_detector.sw");
const LM: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_landmarks.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn face_task_e2e_and_tracking() {
    let (Ok(det_b), Ok(lm_b), Ok(frame)) =
        (std::fs::read(DET), std::fs::read(LM), std::fs::read(FRAME))
    else {
        eprintln!("skip: 모델/프레임 없음 (make convert-mediapipe)");
        return;
    };
    let mut det = CpuSession::load(&det_b).expect("det 로드");
    let mut lm = CpuSession::load(&lm_b).expect("lm 로드");
    let mut task = FaceTask::new(false);

    // 프레임 1: 검출 경로
    let r1 = task
        .process_cpu(&mut det, &mut lm, &frame, 256, 144, 0.0)
        .expect("process")
        .expect("얼굴 있음");
    assert_eq!(r1.points.len(), 478);
    assert!(r1.presence > 0.9, "presence {}", r1.presence);
    assert_eq!(det.stats().frames, 1);
    assert!(task.is_tracking());
    // 구조 검증: 전 점이 프레임 근방, 코끝(1)이 디텍터 박스 부근(대략 중앙 우측),
    // 눈꼬리(33, 263)가 입(13)보다 위
    for p in &r1.points {
        assert!(p[0] > -0.1 && p[0] < 1.1 && p[1] > -0.1 && p[1] < 1.1, "{p:?}");
    }
    let (nose, mouth, eye_l, eye_r) = (r1.points[1], r1.points[13], r1.points[33], r1.points[263]);
    assert!(nose[0] > 0.4 && nose[0] < 0.7 && nose[1] > 0.3 && nose[1] < 0.7, "코 {nose:?}");
    assert!(eye_l[1] < mouth[1] && eye_r[1] < mouth[1], "눈이 입보다 아래?");
    assert!(eye_l[0] < eye_r[0], "33(왼눈꼬리)이 263(오른눈꼬리)보다 오른쪽?");

    // 프레임 2 (같은 그림): 트래킹 경로 — 디텍터 안 돎, 좌표는 거의 동일
    let r2 = task
        .process_cpu(&mut det, &mut lm, &frame, 256, 144, 33.0)
        .expect("process")
        .expect("얼굴 있음");
    assert_eq!(det.stats().frames, 1, "트래킹인데 디텍터가 돌았다");
    assert_eq!(lm.stats().frames, 2);
    let max_d = r1
        .points
        .iter()
        .zip(&r2.points)
        .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
        .fold(0f32, f32::max);
    // ROI가 검출 기반 → 랜드마크 기반으로 바뀌므로 완전 동일하진 않다
    assert!(max_d < 0.02, "트래킹 프레임 좌표 튐: {max_d}");

    // 프레임 3: 빈 화면 — presence 미달 → None + 트래킹 폐기
    let blank = vec![0u8; 256 * 144 * 3];
    let r3 = task.process_cpu(&mut det, &mut lm, &blank, 256, 144, 66.0).expect("process");
    assert!(r3.is_none(), "빈 화면에서 얼굴?");
    assert!(!task.is_tracking());

    // 프레임 4: 빈 화면 — 검출부터 다시 (디텍터 가동), 검출 실패 → None
    let r4 = task.process_cpu(&mut det, &mut lm, &blank, 256, 144, 99.0).expect("process");
    assert!(r4.is_none());
    assert_eq!(det.stats().frames, 2, "재검출 경로에서 디텍터가 안 돌았다");

    // 프레임 5: 얼굴 복귀 — 검출→랜드마크 재획득
    let r5 = task
        .process_cpu(&mut det, &mut lm, &frame, 256, 144, 132.0)
        .expect("process")
        .expect("얼굴 복귀");
    assert!(r5.presence > 0.9);
    assert_eq!(det.stats().frames, 3);
    println!(
        "코끝 ({:.3},{:.3}) presence {:.3} roi {:?}",
        r5.points[1][0], r5.points[1][1], r5.presence, r5.roi
    );
}
