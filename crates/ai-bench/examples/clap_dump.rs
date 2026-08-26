//! clap_dump — 영상 → 틱별 HandSnapshot 덤프 (오프라인 튜닝용).
//!
//! clap_video와 동일 체인으로 손을 뽑아, 판정 없이 스냅샷만 기록한다.
//! GPU 추론을 한 번만 치르고 clap_tune이 덤프를 고속 리플레이한다.
//!
//! 사용: cargo run --release -p ai-bench --example clap_dump -- <video> <out.txt> [fps] [out_h]
//!
//! 덤프 형식 (틱당 1줄): t_ms|hand|hand
//! (hand = handedness_f32,presence,x0,y0,...,x20,y20)

use std::io::{Read, Write};
use std::process::{Command, Stdio};

use ai_gpu::GpuContext;
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

fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let video = args.next().expect("사용: clap_dump <video> <out.txt> [fps] [out_h]");
    let out_path = args.next().expect("out.txt");
    let fps: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(15.0);
    let out_h: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(720);

    let (src_w, src_h) = probe_dims(&video);
    let w = ((src_w * out_h + src_h / 2) / src_h) & !1;
    let h = out_h & !1;
    eprintln!("# {video}: {src_w}x{src_h} → {w}x{h} @ {fps}fps");

    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let det_bytes = std::fs::read(format!("{root}/models/mediapipe/hand/hand_detector.sw"))
        .expect("hand_detector.sw");
    let lm_bytes = std::fs::read(format!("{root}/models/mediapipe/hand/hand_landmarks.sw"))
        .expect("hand_landmarks.sw");

    let ctx = GpuContext::new_blocking().expect("GPU 초기화");
    let mut det = pollster::block_on(GpuSession::load(&ctx, &det_bytes)).expect("det 로드");
    let mut lm = pollster::block_on(GpuSession::load(&ctx, &lm_bytes)).expect("lm 로드");
    let mut task = HandTask::new(2);
    let env_f32 = |k: &str, d: f32| {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let (ms, ps, pc) = (
        env_f32("CLAP_MIN_SCORE", 0.5),
        env_f32("CLAP_PRES_SEED", 0.5),
        env_f32("CLAP_PRES_CONT", 0.5),
    );
    eprintln!("# thresholds: min_score={ms} pres_seed={ps} pres_cont={pc}");
    task.set_thresholds(ms, ps, pc);

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

    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("out 생성"));
    let frame_len = (w * h * 3) as usize;
    let mut buf = vec![0u8; frame_len];
    let mut idx: u64 = 0;
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
        let mut line = format!("{t_ms:.2}");
        for hr in &hands {
            let mut cell = format!("{:.4},{:.4}", hr.handedness, hr.presence);
            for p in hr.points.iter().take(21) {
                cell.push_str(&format!(",{:.5},{:.5}", p[0], p[1]));
            }
            line.push('|');
            line.push_str(&cell);
        }
        writeln!(out, "{line}").expect("write");
        idx += 1;
    }
    let _ = ff.wait();
    eprintln!("# {idx} ticks → {out_path}");
}
