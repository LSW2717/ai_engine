//! P2-4 게이트: canonicalize 후 잔존 op이 엔진 op 화이트리스트에만 속해야 한다.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_to_canon, Ctx};

const WHITELIST: &[&str] = &[
    "Conv", "Mul", "Add", "Sub", "Relu", "Sigmoid", "Tanh", "HardSigmoid", "hswish", "Concat",
    "chview", "chcopy", "resize", "avgpool", "gpool", "act",
];

#[test]
fn rvm_canon_whitelist() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    let ctx = Ctx {
        size: Some((512, 288)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: vec![
            ("r1i".into(), "r1o".into()),
            ("r2i".into(), "r2o".into()),
            ("r3i".into(), "r3o".into()),
            ("r4i".into(), "r4o".into()),
        ],
        ..Default::default()
    };
    run_to_canon(&mut g, &ctx).unwrap();
    let h = g.op_histogram();
    println!("canon 후: {h:?}");
    for op in h.keys() {
        assert!(WHITELIST.contains(&op.as_str()), "화이트리스트 밖 op: {op}\n{h:?}");
    }
    // 분해 HardSwish 20개가 재합성됐는지 (SE 게이트 8개는 HardSigmoid로 잔존)
    assert_eq!(h.get("hswish").copied().unwrap_or(0), 20);
    assert_eq!(h.get("HardSigmoid").copied().unwrap_or(0), 8);
    assert_eq!(h.get("Clip").copied().unwrap_or(0), 0);
    assert_eq!(h.get("Div").copied().unwrap_or(0), 0);
    assert_eq!(h.get("gpool").copied().unwrap_or(0), 9);
}

#[test]
fn segm_canon_whitelist() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    run_to_canon(&mut g, &Ctx::default()).unwrap();
    let h = g.op_histogram();
    println!("segm canon 후: {h:?}");
    for op in h.keys() {
        assert!(WHITELIST.contains(&op.as_str()), "화이트리스트 밖 op: {op}");
    }
    // NHWC 경계 마킹 확인
    assert_eq!(g.nhwc_inputs.len(), 1);
    assert_eq!(g.nhwc_outputs.len(), 1);
}
