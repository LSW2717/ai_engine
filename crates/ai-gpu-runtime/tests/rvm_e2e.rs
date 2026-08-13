//! P2-10/11 게이트: RVM 256×144 GPU E2E — 2프레임 상태 ping-pong 검증 + 프레임타임.
//! CPU 레퍼런스는 P2-7에서 onnxruntime과 112/112 일치가 증명됐으므로 전이적 오라클이다.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn rvm_two_frames_with_state() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    if !path.exists() {
        eprintln!("skip: rvm 없음");
        return;
    }
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };

    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
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
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();

    let input = XorShift32::new(7).vec_f32((144 * 256 * 3) as usize);

    // ---- CPU 프레임 1 (상태 0) / 프레임 2 (상태 = 프레임 1 출력) ----
    let tid_of = |name: &str| {
        sw.tensors.iter().position(|t| t.name == name).map(|i| i as u32).unwrap()
    };
    let mut cpu1 = CpuExec::new(&sw, &blob);
    cpu1.set_input(tid_of("src"), input.clone());
    cpu1.run().unwrap();
    let cpu1_pha = cpu1.read(tid_of("pha")).unwrap();

    let mut cpu2 = CpuExec::new(&sw, &blob);
    cpu2.set_input(tid_of("src"), input.clone());
    for s in &sw.states {
        let out_val = cpu1.read(s.output).unwrap();
        cpu2.set_input(s.input, out_val);
    }
    cpu2.run().unwrap();
    let cpu2_pha = cpu2.read(tid_of("pha")).unwrap();

    // ---- GPU 2프레임 ----
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    println!(
        "RVM load: ops {} | 고유 파이프라인 {} | 슬롯 {} | arena {:.1}MB | weights {:.1}MB | {:.0}ms",
        model.report.ops,
        model.report.unique_pipelines,
        model.report.slots,
        model.report.arena_bytes as f64 / 1e6,
        model.report.weights_bytes as f64 / 1e6,
        model.report.load_ms
    );

    model.upload_input(&ctx, "src", &input).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();
    let gpu1_pha = pollster::block_on(model.read_output(&ctx, "pha")).unwrap();
    let r4o_1 = pollster::block_on(model.read_output(&ctx, "r4o")).unwrap();
    let cpu1_r4o = cpu1.read(tid_of("r4o")).unwrap();
    assert!(max_err(&r4o_1, &cpu1_r4o) < 5e-3, "프레임1 상태 출력 발산");
    let e1 = max_err(&gpu1_pha, &cpu1_pha);
    println!("프레임1 pha max_err: {e1:.3e}");
    assert!(e1 < 5e-3, "프레임1 발산: {e1}");

    pollster::block_on(model.infer(&ctx)).unwrap();
    let gpu2_pha = pollster::block_on(model.read_output(&ctx, "pha")).unwrap();
    let r1o_2 = pollster::block_on(model.read_output(&ctx, "r1o")).unwrap();
    let cpu2_r1o = cpu2.read(tid_of("r1o")).unwrap();
    assert!(max_err(&r1o_2, &cpu2_r1o) < 5e-3, "프레임2 상태 출력 발산");
    let e2 = max_err(&gpu2_pha, &cpu2_pha);
    let diff_frames = max_err(&gpu1_pha, &gpu2_pha);
    println!("프레임2 pha max_err: {e2:.3e} (프레임간 변화 {diff_frames:.3e})");
    assert!(e2 < 5e-3, "프레임2 발산 (상태 전달 오류 의심): {e2}");
    assert!(diff_frames > 1e-6, "상태가 전달되지 않음 (두 프레임 동일)");

    // ---- 프레임타임 (30프레임 wall) ----
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let t0 = std::time::Instant::now();
    const N: u32 = 30;
    for _ in 0..N {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / N as f64;
    println!("RVM 256×144 프레임타임: {ms:.3}ms/frame ({N}프레임 평균, 리드백 제외)");
}
