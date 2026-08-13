//! CPU 인코딩 vs GPU 실행 분해 (진단용, #[ignore])
//!
//! infer()는 record+submit만 하고 대기하지 않는다. 따라서
//!   - N회 루프 시간(대기 없음) ≈ CPU 인코딩 시간 (큐가 안 차는 한)
//!   - 루프 + wait_idle ≈ max(CPU, GPU)
//! 둘의 차이가 GPU가 CPU를 얼마나 앞지르는지(또는 CPU가 병목인지)를 말해준다.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

#[test]
#[ignore]
fn diag_cpu_gpu() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fixed_144x256.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
        size: Some((256, 144)),
        states: (1..=4).map(|i| (format!("r{i}"), format!("r{i}o"))).collect(),
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let input = XorShift32::new(7).vec_f32((144 * 256 * 3) as usize);
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, "input_1", &input).unwrap();
    for _ in 0..5 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();

    const N: u32 = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..N {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    let record_ms = t0.elapsed().as_secs_f64() * 1e3 / N as f64;
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let total_ms = t0.elapsed().as_secs_f64() * 1e3 / N as f64;

    println!(
        "op {} | CPU record+submit {:.3}ms/frame | 전체(대기 포함) {:.3}ms/frame | CPU 비중 {:.0}%",
        model.report.ops,
        record_ms,
        total_ms,
        record_ms / total_ms * 100.0
    );
    println!("=> 디스패치당 CPU {:.2}us", record_ms * 1e3 / model.report.ops as f64);
}
