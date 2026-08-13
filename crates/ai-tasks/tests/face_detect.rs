//! face_detector e2e — 실제 프레임을 레터박스해 .sw(CPU 백엔드)로 추론하고
//! 앵커 디코드+가중 NMS까지 통과시켜 "얼굴 1개, 말이 되는 기하"를 확인한다.
//!
//! 좌표의 **정확한** 파리티는 브라우저 게이트(web/demo/face-ab.html — MediaPipe
//! wasm과 같은 프레임 좌표 diff)가 담당한다. 이 테스트는 CI에서 GPU·브라우저
//! 없이 전체 후처리 경로가 살아있는지를 지키는 구조 게이트다.
//!
//! 모델(.sw)이 없으면 스킵 — `make convert-mediapipe`로 생성된다.

use ai_tasks::{CpuSession, DetectorPost};

const SW: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_detector.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn face_detect_e2e_cpu() {
    let (Ok(sw), Ok(frame)) = (std::fs::read(SW), std::fs::read(FRAME)) else {
        eprintln!("skip: face_detector.sw 또는 프레임 없음 (make convert-mediapipe)");
        return;
    };
    assert_eq!(frame.len(), 256 * 144 * 3, "프레임은 256×144 RGB u8");

    let mut seg = CpuSession::load(&sw).expect("모델 로드");
    let post = DetectorPost::face_short_range();
    let (iw, ih) = post.input_size();
    let input =
        ai_tasks::detect::letterbox::letterbox_u8_rgb(&frame, 256, 144, iw as usize, ih as usize);

    let dets = seg.detect(&post, &input, 256, 144).expect("detect");
    for d in &dets {
        println!(
            "score {:.3} box [{:.3},{:.3}]–[{:.3},{:.3}] kp {:?}",
            d.score, d.xmin, d.ymin, d.xmax, d.ymax, d.keypoints
        );
    }
    assert_eq!(dets.len(), 1, "이 프레임엔 얼굴이 정확히 1개다");
    let d = &dets[0];
    assert!(d.score > 0.5 && d.score <= 1.0);
    // 박스·키포인트가 프레임 안의 그럴듯한 위치인지 (정밀 좌표는 브라우저 게이트)
    assert!(d.xmin > -0.05 && d.ymin > -0.05 && d.xmax < 1.05 && d.ymax < 1.05);
    let (w, h) = (d.xmax - d.xmin, d.ymax - d.ymin);
    assert!(w > 0.05 && w < 0.9 && h > 0.05 && h < 0.9, "박스 크기 비정상: {w}×{h}");
    assert_eq!(d.keypoints.len(), 6);
    for kp in &d.keypoints {
        assert!(
            kp[0] > d.xmin - 0.1 && kp[0] < d.xmax + 0.1
                && kp[1] > d.ymin - 0.1 && kp[1] < d.ymax + 0.1,
            "키포인트가 박스 밖: {kp:?}"
        );
    }
}
