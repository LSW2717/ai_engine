//! FaceTask GPU 텍스처 입력 e2e — **process_tex(GPU 전처리 직결)가 process_gpu
//! (CPU 레터박스·크롭 + 업로드)와 같은 랜드마크를 내는지** 실모델로 대조한다.
//!
//! 커널 자체의 픽셀 파리티는 tests/tex_input.rs(1e-4)가, MediaPipe 대비 절대
//! 좌표는 브라우저 게이트(face-ab.html)가 담당 — 여기는 두 GPU 경로의 등가와
//! 트래킹 계약(2프레임째 디텍터 생략)을 본다. 모델 없으면 스킵.

use ai_tasks::{FaceTask, GpuPre, GpuSession};

use ai_gpu::GpuContext;

const DET: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_detector.sw");
const LM: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_landmarks.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn process_tex_matches_process_gpu() {
    let (Ok(det_b), Ok(lm_b), Ok(frame)) =
        (std::fs::read(DET), std::fs::read(LM), std::fs::read(FRAME))
    else {
        eprintln!("skip: 모델/프레임 없음 (make convert-mediapipe)");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let mut det = pollster::block_on(GpuSession::load(&ctx, &det_b)).expect("det 로드");
    let mut lm = pollster::block_on(GpuSession::load(&ctx, &lm_b)).expect("lm 로드");

    // 기준: 기존 GPU 경로 (CPU 픽셀 전처리 + 업로드)
    let mut task_cpu_in = FaceTask::new(false);
    let base = pollster::block_on(
        task_cpu_in.process_gpu(&ctx, &mut det, &mut lm, &frame, w, h, 0.0),
    )
    .expect("process_gpu")
    .expect("얼굴 있음");

    // 대상: GPU 텍스처 경로 (전처리 커널 직결)
    let mut pre = GpuPre::new(&ctx);
    pre.frame.upload_rgb(&ctx, &frame, w, h);
    let view = pre.frame.view().unwrap().0.clone();
    let mut task_tex = FaceTask::new(false);
    let r1 = pollster::block_on(
        task_tex.process_tex(&ctx, &pre, &view, &mut det, &mut lm, w, h, 0.0),
    )
    .expect("process_tex")
    .expect("얼굴 있음 (tex)");
    assert_eq!(r1.points.len(), 478);
    assert!(r1.presence > 0.9, "presence {}", r1.presence);

    // 두 경로 좌표 대조 — 입력 픽셀이 f32 연산순서 수준(1e-5)으로만 다르므로
    // 랜드마크도 서브픽셀로 일치해야 한다 (px 단위 0.5 여유)
    let max_px = base
        .points
        .iter()
        .zip(&r1.points)
        .map(|(a, b)| {
            ((a[0] - b[0]) * w as f32).abs().max(((a[1] - b[1]) * h as f32).abs())
        })
        .fold(0f32, f32::max);
    println!("process_tex vs process_gpu 478pt max {max_px:.3}px");
    assert!(max_px < 0.5, "GPU 텍스처 경로 좌표 이탈: {max_px}px");

    // 트래킹 계약: 2프레임째는 디텍터 생략 (process_gpu 1 + process_tex 1 = 2)
    let det_frames = det.stats().frames;
    let r2 = pollster::block_on(
        task_tex.process_tex(&ctx, &pre, &view, &mut det, &mut lm, w, h, 33.0),
    )
    .expect("process_tex 2")
    .expect("얼굴 있음 (트래킹)");
    assert_eq!(det.stats().frames, det_frames, "트래킹인데 디텍터가 돌았다");
    let drift = r1
        .points
        .iter()
        .zip(&r2.points)
        .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
        .fold(0f32, f32::max);
    assert!(drift < 0.02, "트래킹 프레임 좌표 튐: {drift}");

    // 빈 프레임 → 소실 → None + 재검출 경로 (레터박스 커널이 다시 돈다)
    let blank = vec![0u8; (w * h * 3) as usize];
    pre.frame.upload_rgb(&ctx, &blank, w, h);
    let view = pre.frame.view().unwrap().0.clone();
    let r3 = pollster::block_on(
        task_tex.process_tex(&ctx, &pre, &view, &mut det, &mut lm, w, h, 66.0),
    )
    .expect("process_tex 3");
    assert!(r3.is_none(), "빈 화면에서 얼굴?");
    let r4 = pollster::block_on(
        task_tex.process_tex(&ctx, &pre, &view, &mut det, &mut lm, w, h, 99.0),
    )
    .expect("process_tex 4");
    assert!(r4.is_none());
    assert_eq!(det.stats().frames, det_frames + 1, "재검출 경로에서 디텍터가 안 돌았다");
}
