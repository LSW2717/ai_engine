//! **정확도 게이트 (범용)** — onnxruntime 기준값과 Model 전 출력 대조.
//!
//! oracle_real은 엔진 자체 레퍼런스(CpuExec)와 비교라 새 op의 lowering이
//! 양쪽에서 같이 틀리면 통과한다 — MediaPipe 계열(op 3종+canon)을 들일 때는
//! 반드시 이걸 돌린다. 기준값 생성: tools/ort_dump.py.
//!
//! 사용:
//!   /usr/bin/python3 tools/ort_dump.py <model.onnx> target/oracle_<이름>
//!   AI_ONNX=<model.onnx> AI_ORACLE=target/oracle_<이름> \
//!     cargo test --release -p ai-cpu --test gate_ort_models -- --ignored --nocapture

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;

fn read_f32s(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p:?}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore]
fn cpu_outputs_match_ort() {
    let onnx = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx> 필요");
    let oracle = std::path::PathBuf::from(std::env::var("AI_ORACLE").expect("AI_ORACLE=<dir>"));

    let mut g = import(&std::fs::read(&onnx).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "gate").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let mut m = ai_cpu::Model::load(&container).unwrap();

    let in_name = sw.tensors[sw.inputs[0] as usize].name.clone();
    let input = read_f32s(&oracle.join("input_nhwc.f32"));
    m.set_input(&in_name, &input).unwrap();
    m.infer().unwrap();

    let mut checked = 0;
    let mut worst = (0f32, String::new());
    for entry in std::fs::read_dir(&oracle).unwrap() {
        let path = entry.unwrap().path();
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(name) = fname.strip_prefix("out__").and_then(|s| s.strip_suffix(".f32"))
        else {
            continue;
        };
        let want = read_f32s(&path);
        let got = m
            .read_output(name)
            .unwrap_or_else(|e| panic!("출력 {name} 읽기 실패: {e:?}"));
        assert_eq!(want.len(), got.len(), "{name} 길이");
        let mut max_err = 0f32;
        for (a, g) in want.iter().zip(&got) {
            max_err = max_err.max((a - g).abs() / a.abs().max(1.0));
        }
        println!("{name}: max_err {max_err:.3e} ({}elem)", want.len());
        if max_err > worst.0 {
            worst = (max_err, name.to_string());
        }
        let tol: f32 = std::env::var("AI_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-3);
        assert!(max_err <= tol, "{name} max_err {max_err} 초과");
        checked += 1;
    }
    assert!(checked > 0, "오라클 출력 파일 없음: {oracle:?}");
    println!("worst: {} {:.3e} ({checked}출력)", worst.1, worst.0);
}
