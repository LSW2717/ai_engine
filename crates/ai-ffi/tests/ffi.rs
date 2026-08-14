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

use ai_ffi::VbResult;

const FW: usize = 640;
const FH: usize = 360;
const MW: usize = 256;
const MH: usize = 144;

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
