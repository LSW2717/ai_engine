//! C 티어 소프트 합성 게이트 — GPU 관여 0으로 실프레임·실모델 합성 검증
//! (vb_pipeline.rs 의 소프트판: 단색 배경 합성 후 전경 비율 실측).
//!
//! 모델(.sw)이 없으면 스킵 — `make convert-rvm-web`.

use ai_tasks::features::vb::SoftPipeline;

const SW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn soft_mask_appears() {
    let (Ok(sw), Ok(rgb)) = (std::fs::read(SW), std::fs::read(FRAME)) else {
        eprintln!("skip: rvm_256x144.sw 또는 프레임 없음");
        return;
    };
    let (w, h) = (256usize, 144usize);
    let mut base = vec![255u8; w * h * 4];
    for i in 0..w * h {
        base[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
    }

    let mut pipe = SoftPipeline::default();
    pipe.set_model(sw);
    pipe.apply_json(r##"{"background":"#00ff00"}"##).unwrap();

    // EMA 수렴 수 프레임 — 입력은 매번 원본에서 (합성 결과 재입력 금지)
    let mut out = base.clone();
    let t0 = std::time::Instant::now();
    let reps = 5;
    for _ in 0..reps {
        out.copy_from_slice(&base);
        assert!(pipe.process(&mut out, w, h).unwrap(), "any_active인데 무수정");
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    let mut fg = 0usize;
    for px in out.chunks_exact(4) {
        if !(px[0] < 60 && px[1] > 180 && px[2] < 60) {
            fg += 1;
        }
    }
    let frac = fg as f32 / (w * h) as f32;
    println!("C티어 전경 비율 {frac:.3} (기대 0.10~0.35) | {ms:.1}ms/frame(256x144)");
    assert!(frac > 0.05 && frac < 0.6, "소프트 합성 전경 비율 비정상 {frac}");

    // passthrough 계약: 효과 전부 해제 → 무수정
    pipe.apply_json(r#"{"background":null}"#).unwrap();
    let mut untouched = base.clone();
    assert!(!pipe.process(&mut untouched, w, h).unwrap());
    assert_eq!(untouched, base);
}

/// C 티어 실배선 모델 게이트 — 모바일 C 티어는 RVM 이 아니라 경량 r11(2ch 로짓)을
/// 싣는다 (RVM CPU 는 프레임타임 미달). 로짓 EMA 경로 + 속도를 함께 잰다.
#[test]
fn soft_r11_speed() {
    let sw_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/segm_r11_160x288.sw");
    let (Ok(sw), Ok(rgb)) = (std::fs::read(sw_path), std::fs::read(FRAME)) else {
        eprintln!("skip: segm_r11_160x288.sw 또는 프레임 없음");
        return;
    };
    let (w, h) = (256usize, 144usize);
    let mut base = vec![255u8; w * h * 4];
    for i in 0..w * h {
        base[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
    }
    let mut pipe = SoftPipeline::default();
    pipe.set_model(sw);
    pipe.apply_json(r##"{"background":"#00ff00"}"##).unwrap();
    let mut out = base.clone();
    let t0 = std::time::Instant::now();
    let reps = 10;
    for _ in 0..reps {
        out.copy_from_slice(&base);
        pipe.process(&mut out, w, h).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let fg = out.chunks_exact(4).filter(|px| !(px[0] < 60 && px[1] > 180 && px[2] < 60)).count();
    let frac = fg as f32 / (w * h) as f32;
    println!("C티어(r11) 전경 비율 {frac:.3} | {ms:.1}ms/frame(256x144)");
    assert!(frac > 0.03 && frac < 0.7, "r11 소프트 합성 전경 비율 비정상 {frac}");
}
