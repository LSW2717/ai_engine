//! ai-cpu 프레임타임 측정 — ai-convert bench_cpu(순진 스칼라 하한)와 같은
//! 모델·입력·MAC 집계로 직접 비교 가능하게 맞춘다.
//!
//! 사용: `AI_ONNX=<경로> [AI_REPS=50] cargo test --release -p ai-cpu --test bench_cpu -- --ignored --nocapture`

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::rng::XorShift32;

#[test]
#[ignore]
fn bench_cpu_model() {
    let path = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx 경로> 필요");
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "bench").unwrap();

    // MAC 집계 — ai-convert/tests/bench_cpu.rs와 동일 규칙 (dw/std 구분)
    let mut mac: u64 = 0;
    for op in &sw.ops {
        if let ai_core::format::SwOp::Conv { input, cin, cout, kh, kw, sh, groups, .. } = op {
            let it = &sw.tensors[*input as usize];
            let (oh, ow) = ((it.h / *sh).max(1), (it.w / *sh).max(1));
            let per = if *groups > 1 {
                (*cin as u64) * (*kh as u64) * (*kw as u64)
            } else {
                (*cin as u64) * (*cout as u64) * (*kh as u64) * (*kw as u64)
            };
            mac += (oh as u64) * (ow as u64) * per;
        }
    }

    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let (h, w, c) = (t.h, t.w, t.c);
    let name = t.name.clone();
    let input = XorShift32::new(7).vec_f32((h * w * c) as usize);

    let container = sw.write_container(&blob).unwrap();
    let t_load = std::time::Instant::now();
    let mut m = ai_cpu::Model::load(&container).unwrap();
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let threads: usize =
        std::env::var("AI_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    m.set_threads(threads).unwrap();

    m.set_input(&name, &input).unwrap();
    // 워밍업
    for _ in 0..3 {
        m.infer().unwrap();
    }

    let reps: usize = std::env::var("AI_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        m.infer().unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!(
        "\n== ai-cpu (SIMD f32, {threads}스레드) ==\n\
         모델 {path}\n\
         입력 {w}x{h}x{c} | op {} | 로드 {load_ms:.1}ms | {:.3} GMAC\n\
         프레임타임 **{ms:.2}ms** ({:.1} GMAC/s)",
        sw.ops.len(),
        mac as f64 / 1e9,
        mac as f64 / ms * 1e-6,
    );
}
