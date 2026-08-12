//! P2-7 게이트: RVM 256×144 변환 → CPU 레퍼런스 실행 → onnxruntime 오라클 대조.
//! 오라클 덤프가 없으면 /usr/bin/python3로 생성(불가 시 skip).

use std::path::{Path, PathBuf};

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::{compare, npy, CpuExec};

fn ensure_oracle(root: &Path, model: &Path, dir: &Path, state_shapes: &[String]) -> bool {
    if dir.join("manifest.json").exists() {
        return true;
    }
    let mut cmd = std::process::Command::new("/usr/bin/python3");
    cmd.arg(root.join("tools/onnx_oracle.py"))
        .arg(model)
        .args(["--size", "256x144", "--seed", "7"])
        .args(["--set-input", "downsample_ratio=1.0"])
        .arg("--intermediates")
        .arg("--out")
        .arg(dir);
    for s in state_shapes {
        cmd.args(["--input-shape", s]);
    }
    matches!(cmd.status(), Ok(s) if s.success())
}

#[test]
fn rvm_cpu_matches_onnxruntime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_path = root.join("models/rvm_fp32.onnx");
    if !model_path.exists() {
        eprintln!("skip: rvm 없음");
        return;
    }
    // 변환 먼저 (256×144 — CPU 레퍼런스가 감당 가능한 크기) — 상태 shape 확보
    let mut g = import(&std::fs::read(&model_path).unwrap()).unwrap().graph;
    let ctx = Ctx {
        size: Some((256, 144)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: vec![
            ("r1i".into(), "r1o".into()),
            ("r2i".into(), "r2o".into()),
            ("r3i".into(), "r3o".into()),
            ("r4i".into(), "r4o".into()),
        ],
        ..Default::default()
    };
    run_full(&mut g, &ctx).unwrap();
    let (model, blob) = lower(&g, &ctx, "rvm").unwrap();

    // 변환기가 해석한 상태 shape을 오라클에 전달
    let state_shapes: Vec<String> = model
        .states
        .iter()
        .map(|s| {
            let t = &model.tensors[s.input as usize];
            format!("{}=1x{}x{}x{}", t.name, t.c, t.h, t.w)
        })
        .collect();
    let oracle_dir: PathBuf = root.join("target/oracle_rvm_256x144");
    if !ensure_oracle(&root, &model_path, &oracle_dir, &state_shapes) {
        eprintln!("skip: 오라클 생성 불가 (python3/onnxruntime)");
        return;
    }

    // 오라클 입력 재생
    let manifest = compare::load_manifest(&oracle_dir).unwrap();
    let mut exec = CpuExec::new(&model, &blob);
    let src_file = &manifest.inputs["src"];
    let (shape, data) = npy::read_npy_f32(&std::fs::read(oracle_dir.join(src_file)).unwrap())
        .unwrap();
    assert_eq!(shape, vec![1, 3, 144, 256]);
    let src_tid = model.inputs[0];
    exec.set_input(src_tid, npy::nchw_to_nhwc(&data, 3, 144, 256));

    exec.run().unwrap();

    // 전 텐서 대조 (출력 + 중간, 융합으로 의미 바뀐 이름 제외)
    let skip: std::collections::HashSet<String> = g.semantic_changed.iter().cloned().collect();
    let reports = compare::compare_all(&exec, &oracle_dir, &manifest, &skip, 2e-3, 2e-3).unwrap();
    assert!(!reports.is_empty(), "매칭된 텐서 없음");
    let failed: Vec<_> = reports.iter().filter(|r| !r.passed).collect();
    for r in failed.iter().take(10) {
        eprintln!("FAIL {} max {:.3e} mean {:.3e}", r.name, r.max_abs, r.mean_abs);
    }
    // 최종 출력은 반드시 포함·통과
    for out_name in ["pha", "fgr", "r1o", "r2o", "r3o", "r4o"] {
        let r = reports.iter().find(|r| r.name == out_name);
        assert!(r.is_some(), "{out_name} 미검증");
        assert!(r.unwrap().passed, "{out_name} max {:.3e}", r.unwrap().max_abs);
    }
    println!(
        "오라클 대조: {}/{} 텐서 통과 (출력 6종 전부 통과)",
        reports.len() - failed.len(),
        reports.len()
    );
    assert!(failed.len() * 20 <= reports.len(), "발산 텐서 과다: {}/{}", failed.len(), reports.len());
}
