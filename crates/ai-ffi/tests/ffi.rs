//! ai-ffi 네이티브 게이트 —
//!  ① C 표면 스모크: set_video_stream_info → update_effects_config →
//!     render_mask(I420 in-place, 실추론) → destroy. 합성 프레임(무인물)이라
//!     RVM 마스크 ≈ 0 → 단색 배경이 프레임을 덮는다 — Y 평면 평균으로 검증.
//!  ② 웹 diff 덤프: vb-diff.html의 결정적 픽스처(makeFrame/makeMask 1:1 포팅)를
//!     GateHarness로 돌려 RGBA를 web/models/ffi_native_*.bin에 쓴다.
//!     web/demo/ffi-diff.html이 같은 픽스처를 브라우저에서 돌려 채널 diff —
//!     "네이티브(모바일 경로) 스택 = 웹 스택" 교차 증명.
//!
//! 모델(web/models/rvm_256x144.sw)이 없으면 스킵.

use std::sync::Mutex;

use ai_ffi::VbResult;

const FW: usize = 640;
const FH: usize = 360;
const MW: usize = 256;
const MH: usize = 144;

/// C 표면은 전역 STATE 하나 — 표면 테스트끼리 직렬화
static SURFACE_LOCK: Mutex<()> = Mutex::new(());

fn model_path() -> Option<String> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
    std::path::Path::new(p).exists().then(|| p.to_string())
}

/// vb-diff.js makeFrame() 1:1 (f64 산술 — JS Math와 동일)
fn make_frame() -> Vec<u8> {
    let mut d = vec![0u8; FW * FH * 4];
    for y in 0..FH {
        for x in 0..FW {
            let i = (y * FW + x) * 4;
            let mut r = ((x as f64 / FW as f64) * 255.0).round() as u8;
            let mut g = ((y as f64 / FH as f64) * 255.0).round() as u8;
            let mut b = (((x + y) as f64 / (FW + FH) as f64) * 255.0).round() as u8;
            if x as f64 > FW as f64 * 0.55
                && (x as f64) < FW as f64 * 0.8
                && y as f64 > FH as f64 * 0.2
                && (y as f64) < FH as f64 * 0.5
            {
                (r, g, b) = (220, 60, 40);
            }
            if ((x >> 4) + (y >> 4)) % 2 == 0 && (x as f64) < FW as f64 * 0.25
                && y as f64 > FH as f64 * 0.6
            {
                (r, g, b) = (255 - r, 255 - g, 255 - b);
            }
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
            d[i + 3] = 255;
        }
    }
    d
}

/// vb-diff.js makeMask(cx) 1:1 — 1/255 격자 사전 양자화
fn make_mask(cx: f64) -> Vec<f32> {
    let q255 = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    let mut m = vec![0f32; MW * MH];
    for y in 0..MH {
        for x in 0..MW {
            let nx = (x as f64 + 0.5) / MW as f64 - cx;
            let ny = (y as f64 + 0.5) / MH as f64 - 0.58;
            let d = ((nx / 0.21).powi(2) + (ny / 0.4).powi(2)).sqrt();
            let mut v = 1.0 - (d - 1.0) / 0.25;
            if (y as f64) < MH as f64 * 0.18 {
                v = 0.0;
            }
            m[y * MW + x] = q255(v) as f32;
        }
    }
    m
}

#[test]
fn c_surface_smoke() {
    let _lock = SURFACE_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("모델 없음 — 스킵 (make convert-rvm-web)");
        return;
    };
    unsafe {
        let r = ai_ffi::set_video_stream_info(path.as_ptr(), path.len());
        assert_eq!(r, VbResult::Success, "모델 로드");

        let json = std::ffi::CString::new(r##"{"background":"#00a05a"}"##).unwrap();
        assert_eq!(ai_ffi::update_effects_config(json.as_ptr()), VbResult::Success);

        // 중간 회색 I420 (무인물 — RVM 마스크 ≈ 0 → 단색 배경이 덮는다)
        let (w, h) = (FW, FH);
        let mut y = vec![128u8; w * h];
        let mut u = vec![128u8; w * h / 4];
        let mut v = vec![128u8; w * h / 4];
        let r = ai_ffi::render_mask(
            y.as_mut_ptr(),
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            w as i32,
            h as i32,
            w as i32,
            (w / 2) as i32,
            (w / 2) as i32,
        );
        assert_eq!(r, VbResult::Success, "render_mask");
        // #00a05a → Y = 0.587·160 + 0.114·90 ≈ 104
        let mean_y: f64 = y.iter().map(|&v| v as f64).sum::<f64>() / (w * h) as f64;
        assert!(
            (95.0..115.0).contains(&mean_y),
            "배경색 Y 기대 ~104, 실측 {mean_y:.1} (마스크가 안 덮었거나 변환 상수 불일치)"
        );

        assert_eq!(ai_ffi::destroy_custom_video_stream(), VbResult::Success);
    }
}

/// 신규 표면 스모크 — 단일 JSON 설정으로 집중도(실추론) + 결과 폴링 + 오디오 +
/// destroy=리셋(웜 재가동)까지. 모델 없으면 스킵.
#[test]
fn extended_surface_smoke() {
    let _lock = SURFACE_LOCK.lock().unwrap();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let seg = format!("{root}/web/models/rvm_256x144.sw");
    let det = format!("{root}/models/mediapipe/face/face_detector.sw");
    let lm = format!("{root}/models/mediapipe/face/face_landmarks.sw");
    let gaze = format!("{root}/models/gaze.sw");
    let bs = format!("{root}/models/mediapipe/face/face_blendshapes.sw");
    let frame_p = format!("{root}/tests/data/frame_256x144.rgb");
    for p in [&seg, &det, &lm, &gaze, &frame_p] {
        if !std::path::Path::new(p).exists() {
            eprintln!("스킵: {p} 없음");
            return;
        }
    }
    let rgb = std::fs::read(&frame_p).unwrap();
    let (w, h) = (256usize, 144usize);
    // 실얼굴 픽스처 → I420 (엔진 yuv 모듈 재사용 — BT.601 full)
    let mut rgba = vec![255u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
    }
    let (mut y, mut u, mut v) =
        (vec![0u8; w * h], vec![0u8; w * h / 4], vec![0u8; w * h / 4]);
    ai_ffi::yuv::rgba_to_i420(&rgba, w, h, &mut y, &mut u, &mut v, w, w / 2, w / 2);
    let y0 = y.clone();

    unsafe {
        let c = |s: &str| std::ffi::CString::new(s).unwrap();
        assert_eq!(ai_ffi::set_video_stream_info(seg.as_ptr(), seg.len()), VbResult::Success);
        assert_eq!(
            ai_ffi::set_face_model_info(c(&det).as_ptr(), c(&lm).as_ptr()),
            VbResult::Success
        );
        assert_eq!(
            ai_ffi::set_gaze_model_info(c(&gaze).as_ptr(), c(&bs).as_ptr()),
            VbResult::Success
        );
        let cfg = c(r##"{"background":"#00a05a","focusDetection":{"enabled":true,"detectFps":30}}"##);
        assert_eq!(ai_ffi::update_effects_config(cfg.as_ptr()), VbResult::Success);

        // 음수 치수 계약: |값| 사용 (모바일 renderMask 규약)
        let render = |y: &mut [u8], u: &mut [u8], v: &mut [u8], neg: bool| {
            let s = if neg { -1i32 } else { 1 };
            ai_ffi::render_mask(
                y.as_mut_ptr(),
                u.as_mut_ptr(),
                v.as_mut_ptr(),
                w as i32 * s,
                h as i32 * s,
                w as i32,
                (w / 2) as i32,
                (w / 2) as i32,
            )
        };
        // FOCUSED까지: 온타깃 250ms + baseline — 벽시계 페이싱이라 실시간 대기
        for i in 0..8 {
            assert_eq!(render(&mut y, &mut u, &mut v, i == 0), VbResult::Success);
            std::thread::sleep(std::time::Duration::from_millis(60));
        }
        let fs = ai_ffi::get_focus_state();
        assert!(!fs.is_null());
        let s = std::ffi::CStr::from_ptr(fs).to_string_lossy().into_owned();
        ai_ffi::vcx_string_free(fs);
        assert!(s.contains("\"status\":\"FOCUSED\""), "focus: {s}");
        assert!(ai_ffi::poll_hand_gesture().is_null(), "hand 미기동 — 이벤트 없음");
        assert_ne!(y, y0, "배경 합성이 프레임을 바꿔야");

        // destroy = 리셋 (모델 유지) → 바로 재가동
        assert_eq!(ai_ffi::destroy_custom_video_stream(), VbResult::Success);
        assert_eq!(render(&mut y, &mut u, &mut v, false), VbResult::Success, "웜 재가동");

        // 오디오 — 48k 실모델 + 미지원 레이트 passthrough
        let adir = c(&format!("{root}/models/fastenhancer"));
        let fe = ai_ffi::fe_create_c(48000, adir.as_ptr());
        if fe.is_null() {
            eprintln!("오디오 모델 없음 — fe 스킵 (make convert-fastenhancer)");
        } else {
            let n = ai_ffi::fe_get_in_frame_len(fe);
            assert!(n > 0 && ai_ffi::fe_get_sample_rate(fe) == 48000);
            let inp = vec![0f32; n];
            let mut out = vec![1f32; n];
            ai_ffi::fe_process_frame(fe, inp.as_ptr(), out.as_mut_ptr());
            ai_ffi::fe_free_c(fe);
        }
        let pt = ai_ffi::fe_create_c(44100, adir.as_ptr());
        assert!(!pt.is_null());
        assert_eq!(ai_ffi::fe_get_in_frame_len(pt), 480);
        let inp: Vec<f32> = (0..480).map(|i| i as f32 / 480.0).collect();
        let mut out = vec![0f32; 480];
        ai_ffi::fe_process_frame(pt, inp.as_ptr(), out.as_mut_ptr());
        assert_eq!(out, inp, "passthrough는 무가공");
        ai_ffi::fe_free_c(pt);

        assert_eq!(ai_ffi::destroy_custom_video_stream(), VbResult::Success);
        println!("extended surface OK — focus={s}");
    }
}

/// 웹 diff 픽스처 덤프 — ffi-diff.html이 소비.
/// 재현: cargo test -p ai-ffi --release → node tools/run_web.mjs demo/ffi-diff.html
#[test]
fn dump_native_stack_for_web_diff() {
    let Some(path) = model_path() else {
        eprintln!("모델 없음 — 스킵");
        return;
    };
    use ai_gpu::GpuContext;
    use ai_tasks::features::vb::GateHarness;
    use ai_tasks::GpuSession;

    let ctx = GpuContext::new_blocking().expect("GPU");
    let bytes = std::fs::read(&path).unwrap();
    let seg = pollster::block_on(GpuSession::load(&ctx, &bytes)).expect("RVM 로드");
    let frame = make_frame();
    let mask = make_mask(0.5);

    // vb-diff.js MODES의 color/blur — S상과 동일 (ch=1, ema=false)
    let modes = [
        ("color", r##"{"background":"#00a05a","blur":0,"brightness":1,"grayscale":0,"studioLight":null}"##),
        ("blur", r##"{"background":null,"blur":0.6,"brightness":1,"grayscale":0,"studioLight":null}"##),
    ];
    let out_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models");
    for (name, cfg) in modes {
        let mut g = GateHarness::new(&ctx);
        g.pipeline.apply_json(cfg).unwrap();
        let rgba = pollster::block_on(g.frame(
            &ctx,
            &seg,
            &frame,
            FW as u32,
            FH as u32,
            &mask,
            1,
            MW as u32,
            MH as u32,
            false,
        ))
        .expect("gate frame");
        assert_eq!(rgba.len(), FW * FH * 4);
        let p = format!("{out_dir}/ffi_native_{name}.bin");
        std::fs::write(&p, &rgba).unwrap();
        eprintln!("dumped {p}");
    }
}
