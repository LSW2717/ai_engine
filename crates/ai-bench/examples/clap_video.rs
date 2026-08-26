//! clap_video — 실영상 박수 판정 오프라인 러너.
//!
//! 프로덕션 경로(director.rs 손 블록)와 동일한 체인을 영상 파일에 재현한다:
//! ffmpeg rgb24 디코드 → detectFps 샘플링 → HandTask(process_gpu) →
//! HandSnapshot 변환(director와 동일 매핑) → GestureClassifier::classify.
//!
//! 사용:
//!   cargo run --release -p ai-bench --example clap_video -- <video> [fps] [out_h]
//!
//! 출력: 틱별 진단(stderr CSV: t_ms,hands,d,center_d,labels) + clap 발화
//! 타임라인/총계(stdout). 진단 d/center_d는 gesture.rs 공개 함수로 러너가
//! 재계산 — 판정 자체는 실제 GestureClassifier가 한다.

use std::io::Read;
use std::process::{Command, Stdio};

use ai_gpu::GpuContext;
use ai_tasks::features::hand::gesture::{
    min_fingertip_dist, palm_center, palm_width, Gesture, GestureClassifier, Handedness,
    HandSnapshot,
};
use ai_tasks::{GpuSession, HandTask};

fn probe_dims(path: &str) -> (u32, u32) {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0", path,
        ])
        .output()
        .expect("ffprobe 실행 실패");
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().split(',');
    let w: u32 = it.next().unwrap().parse().expect("width");
    let h: u32 = it.next().unwrap().parse().expect("height");
    (w, h)
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let video = args.next().expect("사용: clap_video <video> [fps] [out_h]");
    let fps: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(10.0);
    let out_h: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(720);

    let (src_w, src_h) = probe_dims(&video);
    // 짝수 강제 (yuv/rgb 스트라이드 안전)
    let w = ((src_w * out_h + src_h / 2) / src_h) & !1;
    let h = out_h & !1;
    eprintln!("# {video}: {src_w}x{src_h} → {w}x{h} @ {fps}fps");

    // 모델 — 워크스페이스 루트 기준 (CARGO_MANIFEST_DIR = crates/ai-bench)
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let det_bytes = std::fs::read(format!("{root}/models/mediapipe/hand/hand_detector.sw"))
        .expect("hand_detector.sw");
    let lm_bytes = std::fs::read(format!("{root}/models/mediapipe/hand/hand_landmarks.sw"))
        .expect("hand_landmarks.sw");

    let ctx = GpuContext::new_blocking().expect("GPU 초기화");
    eprintln!("# adapter: {} ({:?})", ctx.caps.info.name, ctx.caps.info.backend);
    let mut det = pollster::block_on(GpuSession::load(&ctx, &det_bytes)).expect("det 로드");
    let mut lm = pollster::block_on(GpuSession::load(&ctx, &lm_bytes)).expect("lm 로드");
    let mut task = HandTask::new(2);
    let env_f32 = |k: &str, d: f32| {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    task.set_thresholds(
        env_f32("CLAP_MIN_SCORE", 0.5),
        env_f32("CLAP_PRES_SEED", 0.5),
        env_f32("CLAP_PRES_CONT", 0.5),
    );
    let mut cls = GestureClassifier::default();

    let mut ff = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", &video,
            "-vf", &format!("fps={fps},scale={w}:{h}"),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .expect("ffmpeg 실행 실패");
    let mut pipe = ff.stdout.take().unwrap();

    let frame_len = (w * h * 3) as usize;
    let mut buf = vec![0u8; frame_len];
    let mut idx: u64 = 0;
    let mut claps: Vec<f64> = Vec::new();
    let mut others: Vec<(String, f64)> = Vec::new();

    eprintln!("t_ms,hands,d,center_d,labels");
    loop {
        if let Err(e) = pipe.read_exact(&mut buf) {
            if e.kind() != std::io::ErrorKind::UnexpectedEof {
                eprintln!("# 파이프 에러: {e}");
            }
            break;
        }
        let t_ms = idx as f64 * 1000.0 / fps;
        let hands = pollster::block_on(task.process_gpu(&ctx, &mut det, &mut lm, &buf, w, h, t_ms))
            .expect("hand_task");
        // director.rs와 동일 변환
        let snaps: Vec<HandSnapshot> = hands
            .iter()
            .map(|hr| {
                let mut landmarks = [[0f32; 2]; 21];
                for (i, p) in hr.points.iter().take(21).enumerate() {
                    landmarks[i] = [p[0], p[1]];
                }
                let handedness =
                    if hr.handedness > 0.5 { Handedness::Left } else { Handedness::Right };
                HandSnapshot { landmarks, handedness, presence: hr.presence }
            })
            .collect();

        // 진단 (판정과 무관 — 러너 재계산). 팜 중심/팜폭까지 — 오프라인 시뮬용.
        let centers: Vec<String> = snaps
            .iter()
            .zip(hands.iter())
            .map(|(s, hr)| {
                let c = palm_center(&s.landmarks);
                // 손 축 각도 — wrist(0)→middleMCP(9), 화면 위쪽=0°, 시계방향 +
                let (w, m) = (s.landmarks[0], s.landmarks[9]);
                let ang = (m[0] - w[0]).atan2(-(m[1] - w[1])).to_degrees();
                // 손 길이 (wrist↔middleMCP) — aspect = 팜폭/손길이, 팜 정면/모서리 판별용
                let hlen = ((m[0] - w[0]).powi(2) + (m[1] - w[1]).powi(2)).sqrt();
                // 팜 법선 (3D) — (indexMCP−wrist)×(pinkyMCP−wrist), 정규화
                let p3 = |i: usize| hr.points[i];
                let (w3, i3, k3) = (p3(0), p3(5), p3(17));
                let u = [i3[0] - w3[0], i3[1] - w3[1], i3[2] - w3[2]];
                let v = [k3[0] - w3[0], k3[1] - w3[1], k3[2] - w3[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
                // 랜드마크 21점 bbox — 쌍 중첩(IoU/내포) 진단용
                let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for p in &s.landmarks {
                    x0 = x0.min(p[0]);
                    y0 = y0.min(p[1]);
                    x1 = x1.max(p[0]);
                    y1 = y1.max(p[1]);
                }
                format!(
                    "{:.4}:{:.4}:{:.4}:{:.0}:{:.3}:{:.3}:{:.4}:{:.3}:{:.3}:{:.3}:{:.4}:{:.4}:{:.4}:{:.4}",
                    c[0], c[1], palm_width(&s.landmarks), ang, hr.presence, hr.handedness,
                    hlen, n[0] / nl, n[1] / nl, n[2] / nl, x0, y0, x1, y1
                )
            })
            .collect();
        let (d_s, cd_s, labels) = if snaps.len() >= 2 {
            let (a, b) = (&snaps[0], &snaps[1]);
            let scale = palm_width(&a.landmarks).max(palm_width(&b.landmarks)).max(1e-6);
            let d = min_fingertip_dist(&a.landmarks, &b.landmarks) / scale;
            let cd = dist2(palm_center(&a.landmarks), palm_center(&b.landmarks)) / scale;
            (format!("{d:.3}"), format!("{cd:.3}"), format!("{:?}/{:?}", a.handedness, b.handedness))
        } else if snaps.len() == 1 {
            (String::new(), String::new(), format!("{:?}", snaps[0].handedness))
        } else {
            (String::new(), String::new(), String::new())
        };
        eprintln!("{t_ms:.0},{},{d_s},{cd_s},{labels},{}", snaps.len(), centers.join("|"));

        for ev in cls.classify(&snaps, t_ms) {
            match ev.gesture {
                Gesture::Clap => {
                    println!("CLAP  t={:.1}s confidence={:.2}", ev.ts_ms / 1000.0, ev.confidence);
                    claps.push(ev.ts_ms);
                }
                g => others.push((format!("{g:?}"), ev.ts_ms)),
            }
        }
        idx += 1;
    }
    let _ = ff.wait();

    println!("──────────────────────────────");
    println!(
        "총 CLAP {}회  ({})",
        claps.len(),
        claps.iter().map(|t| format!("{:.1}s", t / 1000.0)).collect::<Vec<_>>().join(", ")
    );
    if !others.is_empty() {
        println!(
            "기타 제스처: {}",
            others.iter().map(|(g, t)| format!("{g}@{:.1}s", t / 1000.0)).collect::<Vec<_>>().join(", ")
        );
    }
}
