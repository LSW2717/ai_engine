//! ai-bench — 네이티브 마이크로벤치마크 러너.
//! wasm 데모와 동일한 루틴·시드 입력을 사용하므로 네이티브/브라우저 수치를 직접 비교할 수 있다.

use ai_gpu::GpuContext;

fn main() {
    env_logger::init();
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU 초기화 실패: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "adapter: {} ({:?}) | f16={} timestamps={}\n",
        ctx.caps.info.name, ctx.caps.info.backend, ctx.caps.f16, ctx.caps.timestamps
    );

    let results = pollster::block_on(ai_gpu::bench::run_benchmarks(&ctx)).unwrap_or_else(|e| {
        eprintln!("벤치 실패: {e}");
        std::process::exit(1);
    });

    println!(
        "{:<64} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "kernel", "gpu_min", "gpu_med", "wall", "GFLOP/s", "pipe_ms"
    );
    let fmt_us = |ms: f64| format!("{:.1}us", ms * 1e3);
    let mut total_min = 0.0;
    for r in &results {
        total_min += r.gpu_min_ms.unwrap_or(r.wall_ms);
        println!(
            "{:<64} {:>10} {:>10} {:>10} {:>9.1} {:>9.2}",
            r.name,
            r.gpu_min_ms.map(fmt_us).unwrap_or_else(|| "-".into()),
            r.gpu_median_ms.map(fmt_us).unwrap_or_else(|| "-".into()),
            fmt_us(r.wall_ms),
            r.gflops,
            r.pipeline_ms
        );
    }
    println!("\n표 전체 합(gpu_min 기준): {:.3} ms", total_min);
}
