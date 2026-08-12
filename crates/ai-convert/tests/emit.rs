//! P2-6 게이트: RVM/segm 전체 변환 → .sw 왕복·정렬·크기 검증.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::format::{SwModel, BLOB_ALIGN};

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
fn rvm_emits_and_roundtrips() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    let ctx = rvm_ctx();
    run_full(&mut g, &ctx).unwrap();
    let (model, blob) = lower(&g, &ctx, "rvm").unwrap();

    // 크기: fp32 가중치 ≈ 15MB (원본 14.9MB + 정렬 패딩, dedup 감안)
    let mb = blob.len() as f64 / 1e6;
    assert!((10.0..20.0).contains(&mb), "블롭 크기 이상: {mb:.2}MB");

    // 모든 가중치 오프셋 256 정렬
    for op in &model.ops {
        if let ai_core::format::SwOp::Conv { w, b, .. } = op {
            assert_eq!(w.off % BLOB_ALIGN as u64, 0);
            assert_eq!(b.off % BLOB_ALIGN as u64, 0);
            assert!(w.off + w.len <= blob.len() as u64);
        }
    }

    // 입출력·상태
    assert_eq!(model.inputs.len(), 5); // src + r1i..r4i
    assert_eq!(model.outputs.len(), 6); // fgr, pha, r1o..r4o
    assert_eq!(model.states.len(), 4);
    assert_eq!(model.size.h, 288);
    assert_eq!(model.size.w, 512);

    // 컨테이너 왕복
    let bytes = model.write_container(&blob).unwrap();
    let (m2, blob2) = SwModel::parse_container(&bytes).unwrap();
    assert_eq!(model, m2);
    assert_eq!(blob2.len(), blob.len());

    println!(
        "RVM .sw OK: 텐서 {} | op {} | 블롭 {mb:.2}MB | 컨테이너 {:.2}MB",
        model.tensors.len(),
        model.ops.len(),
        bytes.len() as f64 / 1e6
    );
}

#[test]
fn segm_emits() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (model, blob) = lower(&g, &ctx, "segm").unwrap();
    let bytes = model.write_container(&blob).unwrap();
    let (m2, _) = SwModel::parse_container(&bytes).unwrap();
    assert_eq!(model, m2);
    println!(
        "segm .sw OK: 텐서 {} | op {} | 블롭 {:.2}MB",
        model.tensors.len(),
        model.ops.len(),
        blob.len() as f64 / 1e6
    );
}
