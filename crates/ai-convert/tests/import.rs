//! P2-2 게이트: RVM/segm ONNX 임포트 — raw 히스토그램이 사전 분석값과 일치해야 한다.
//! 모델 파일이 없으면 skip (env AI_TEST_RVM_ONNX 또는 workspace models/ 기본 경로).

use ai_convert::onnx::import::import;

fn model_path(env: &str, default_rel: &str) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(p.into());
    }
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(default_rel);
    p.exists().then_some(p)
}

#[test]
fn rvm_imports_with_expected_histogram() {
    let Some(path) = model_path("AI_TEST_RVM_ONNX", "../../models/rvm_fp32.onnx") else {
        eprintln!("skip: rvm_fp32.onnx 없음");
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    let imported = import(&bytes).unwrap();
    assert_eq!(imported.opset, 12);

    let h = imported.graph.op_histogram();
    let get = |k: &str| h.get(k).copied().unwrap_or(0);
    // 사전 분석(에이전트가 onnx 파이썬으로 검증한 값). Constant 48개는 임포트에서 상수화됨.
    assert_eq!(get("Conv"), 85);
    assert_eq!(get("Constant"), 0);
    assert_eq!(get("Mul"), 47);
    assert_eq!(get("HardSigmoid"), 28);
    assert_eq!(get("Relu"), 27);
    assert_eq!(get("Concat"), 23);
    assert_eq!(get("Add"), 16);
    assert_eq!(get("Shape"), 12);
    assert_eq!(get("Slice"), 12);
    assert_eq!(get("Split"), 10);
    assert_eq!(get("GlobalAveragePool"), 9);
    assert_eq!(get("Sub"), 8);
    assert_eq!(get("Resize"), 7);
    assert_eq!(get("Sigmoid"), 5);
    assert_eq!(get("Expand"), 4);
    assert_eq!(get("Tanh"), 4);
    assert_eq!(get("AveragePool"), 3);
    assert_eq!(get("ReduceMean"), 2);
    assert_eq!(get("Clip"), 2);
    assert_eq!(get("Div"), 1);

    // 입출력·상태 이름
    assert!(imported.graph.inputs.iter().any(|i| i == "src"));
    assert!(imported.graph.inputs.iter().any(|i| i == "downsample_ratio"));
    for r in ["r1i", "r2i", "r3i", "r4i"] {
        assert!(imported.graph.inputs.iter().any(|i| i == r), "{r} 없음");
    }
    for o in ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"] {
        assert!(imported.graph.outputs.iter().any(|x| x == o), "{o} 없음");
    }
    println!("RVM 임포트 OK: 노드 {}개", imported.graph.nodes.len());
}

#[test]
fn segm_nhwc_imports() {
    let Some(path) =
        model_path("AI_TEST_SEGM_ONNX", "../../models/segm_mnv4s050_s2_160x288_nhwc.onnx")
    else {
        eprintln!("skip: segm onnx 없음");
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    let imported = import(&bytes).unwrap();
    assert_eq!(imported.opset, 18);
    let h = imported.graph.op_histogram();
    // NHWC export의 지문: 경계 Transpose 2개
    assert_eq!(h.get("Transpose").copied().unwrap_or(0), 2);
    assert!(h.get("Conv").copied().unwrap_or(0) > 10);
    println!("segm 임포트 OK: {h:?}");
}
