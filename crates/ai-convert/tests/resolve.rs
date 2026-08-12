//! P2-3 게이트: RVM 512×288 정적 해석 — 전 텐서 정적, 상태 shape 확정, Shape/Expand 전멸.

use ai_convert::onnx::import::import;
use ai_convert::passes::{dce, resolve_static, Ctx};

fn rvm_ctx() -> Ctx {
    Ctx {
        size: Some((512, 288)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: vec![
            ("r1i".into(), "r1o".into()),
            ("r2i".into(), "r2o".into()),
            ("r3i".into(), "r3o".into()),
            ("r4i".into(), "r4o".into()),
        ],
        ..Default::default()
    }
}

#[test]
fn rvm_resolves_fully_static_at_512x288() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    if !path.exists() {
        eprintln!("skip: rvm_fp32.onnx 없음");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    let ctx = rvm_ctx();
    resolve_static::run(&mut g, &ctx).unwrap();
    dce::run(&mut g, &ctx).unwrap();

    // 상태 shape 확정 (512×288: 다운샘플 1/2씩)
    let shape = |n: &str| g.info(n).unwrap().static_shape().unwrap().to_vec();
    assert_eq!(shape("r1i"), vec![1, 16, 144, 256]);
    assert_eq!(shape("r2i"), vec![1, 20, 72, 128]);
    assert_eq!(shape("r3i"), vec![1, 40, 36, 64]);
    assert_eq!(shape("r4i"), vec![1, 64, 18, 32]);

    // Shape/Expand/배관 전멸
    let h = g.op_histogram();
    assert_eq!(h.get("Shape").copied().unwrap_or(0), 0, "{h:?}");
    assert_eq!(h.get("Expand").copied().unwrap_or(0), 0);
    assert_eq!(h.get("Cast").copied().unwrap_or(0), 0);
    assert_eq!(h.get("Div").copied().unwrap_or(0), 1, "실데이터 Div(std)만 남아야");

    println!("정적 해석 후 히스토그램: {h:?}");
}

#[test]
fn segm_resolves_without_size_flag() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    if !path.exists() {
        eprintln!("skip: segm onnx 없음");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    let ctx = Ctx::default(); // 정적 모델 — size 불필요
    resolve_static::run(&mut g, &ctx).unwrap();
    dce::run(&mut g, &ctx).unwrap();
    println!("segm 정적 해석 OK: {:?}", g.op_histogram());
}
