//! P2-5 게이트: 전체 파이프라인 히스토그램 회귀 잠금 + 융합 통계.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};

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
fn rvm_full_pipeline_histogram() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    run_full(&mut g, &rvm_ctx()).unwrap();
    let h = g.op_histogram();
    println!("융합 후: {h:?}");

    // 융합 품질 지표
    let conv_with_act = g
        .live_nodes()
        .filter(|(_, n)| n.op == "Conv" && n.attr_s("act").is_some())
        .count();
    let conv_with_res = g
        .live_nodes()
        .filter(|(_, n)| n.op == "Conv" && n.attrs.contains_key("res"))
        .count();
    let scalar_binaries = g
        .live_nodes()
        .filter(|(_, n)| n.attrs.contains_key("scalar"))
        .count();
    let cvec_binaries = g
        .live_nodes()
        .filter(|(_, n)| n.attrs.contains_key("cvec"))
        .count();
    println!(
        "conv+act {conv_with_act} | conv+res {conv_with_res} | scalar {scalar_binaries} | cvec {cvec_binaries}"
    );

    // 골든 plan 대비 지표 (같은 백본 78 conv 중 act 융합 62개 상당) — 우리 export엔
    // refiner(+9 conv)가 있어 총량이 다르다. 회귀 잠금:
    assert_eq!(h.get("Conv").copied().unwrap_or(0), 87);
    assert!(conv_with_act >= 55, "conv act 융합이 너무 적음: {conv_with_act}");
    assert!(conv_with_res >= 6, "residual 융합이 너무 적음: {conv_with_res}");
    // GRU 갱신 4체인(sub/mul/mul/add)은 mix로 융합 — Sub(1,z)가 남으면 퇴행
    assert_eq!(h.get("mix").copied().unwrap_or(0), 4, "GRU mix 융합 실패: {h:?}");
    let _ = scalar_binaries; // mix 융합 후 scalar Sub는 사라진다 (지표만 출력)
    // concat-into-conv: UNet 스킵 concat 12개가 conv에 흡수 (잔여 8 = resize/pw
    // 소비자 5 + 비정렬 fgr+pha 3 — 확장 시 이 잠금을 내릴 것)
    assert!(
        h.get("Concat").copied().unwrap_or(0) <= 8,
        "concat-into-conv 융합 퇴행: {h:?}"
    );
    // 정규화 mean/std cvec 2개
    assert!(cvec_binaries >= 2);
    // 단독 활성화가 남아도 소수여야 (elementwise Unary로 lowering)
    let standalone_acts: usize = ["Relu", "Sigmoid", "Tanh", "HardSigmoid", "hswish", "act"]
        .iter()
        .map(|k| h.get(*k).copied().unwrap_or(0))
        .sum();
    assert!(standalone_acts <= 14, "단독 활성화 과다: {standalone_acts} ({h:?})");
}

#[test]
fn segm_full_pipeline() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    if !path.exists() {
        eprintln!("skip");
        return;
    }
    let mut g = import(&std::fs::read(path).unwrap()).unwrap().graph;
    run_full(&mut g, &Ctx::default()).unwrap();
    let h = g.op_histogram();
    let conv_with_act = g
        .live_nodes()
        .filter(|(_, n)| n.op == "Conv" && n.attr_s("act").is_some())
        .count();
    let conv_with_res = g
        .live_nodes()
        .filter(|(_, n)| n.op == "Conv" && n.attrs.contains_key("res"))
        .count();
    println!("segm 융합 후: {h:?} | conv+act {conv_with_act} | conv+res {conv_with_res}");
    assert_eq!(h.get("Conv").copied().unwrap_or(0), 36);
    assert!(conv_with_act >= 20);
    assert!(conv_with_res >= 5, "MNv4 residual 융합 실패: {conv_with_res}");
}
