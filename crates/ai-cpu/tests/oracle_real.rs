//! 실모델 오라클 대조 — 변환기 산출물을 Model과 CpuExec 양쪽에 태워
//! 전 출력 diff. 상태(state)는 양쪽 다 0 초기화라 1프레임 비교가 유효하다.
//!
//! 사용: `AI_ONNX=<경로> cargo test --release -p ai-cpu --test oracle_real -- --ignored --nocapture`

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;

#[test]
#[ignore]
fn real_model_matches_cpuexec() {
    let path = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx 경로> 필요");
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "oracle").unwrap();

    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let input = XorShift32::new(7).vec_f32((t.h * t.w * t.c) as usize);

    let mut oracle = CpuExec::new(&sw, &blob);
    oracle.set_input(in_tid, input.clone());
    oracle.run().unwrap();

    let container = sw.write_container(&blob).unwrap();
    let mut m = ai_cpu::Model::load(&container).unwrap();
    m.set_input(&sw.tensors[in_tid as usize].name.clone(), &input).unwrap();
    m.infer().unwrap();

    let mut worst = (0f32, String::new());
    for &o in &sw.outputs {
        let name = &sw.tensors[o as usize].name;
        let want = oracle.read(o).unwrap();
        let got = m.read_output(name).unwrap();
        assert_eq!(want.len(), got.len(), "{name} 길이");
        let mut max_err = 0f32;
        for (a, g) in want.iter().zip(&got) {
            max_err = max_err.max((a - g).abs() / a.abs().max(1.0));
        }
        println!("{name}: max_err {max_err:.3e} ({}elem)", want.len());
        if max_err > worst.0 {
            worst = (max_err, name.clone());
        }
        assert!(max_err <= 5e-4, "{name} max_err {max_err} 초과");
    }
    println!("worst: {} {:.3e}", worst.1, worst.0);
}
