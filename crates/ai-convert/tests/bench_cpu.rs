//! CPU 실행기 성능 하한 측정 (진단용, #[ignore]).
//!
//! `CpuExec`는 정확도 레퍼런스로 만든 **순진 스칼라 f32** 구현이다 — SIMD도
//! 멀티스레드도 없다. 여기서 나오는 숫자가 `ai-cpu` 백엔드의 출발점(하한)이고,
//! 목표까지의 배수가 곧 필요한 최적화 규모다.
//!
//! 비교 기준선 (ncnn, 손튜닝 SIMD+멀티스레드, M2 Pro, RVM 144×256):
//! CPU 4스레드 fp16 12.3ms / fp32 19.6ms.
//!
//! 사용: `AI_ONNX=<경로> cargo test --release -p ai-convert --test bench_cpu -- --ignored --nocapture`

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;

#[test]
#[ignore]
fn bench_cpu_exec() {
    let path = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx 경로> 필요");
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "bench").unwrap();

    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let (h, w, c) = (t.h, t.w, t.c);
    let input = XorShift32::new(7).vec_f32((h * w * c) as usize);

    // MAC 집계 — dw(groups>1)와 std를 구분해야 숫자가 안 부풀려진다
    let mut mac: u64 = 0;
    for op in &sw.ops {
        if let ai_core::format::SwOp::Conv {
            input, cin, cout, kh, kw, sh, groups, ..
        } = op
        {
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

    // 워밍업 1 + 측정 (순진 구현이라 프레임당 수백 ms일 수 있어 반복을 적게)
    let reps: usize = std::env::var("AI_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let mut exec = CpuExec::new(&sw, &blob);
    exec.set_input(in_tid, input.clone());
    exec.run().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let mut e = CpuExec::new(&sw, &blob);
        e.set_input(in_tid, input.clone());
        e.run().unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!(
        "\n== CPU 실행기 (순진 스칼라 f32, 1스레드) ==\n\
         모델 {path}\n\
         입력 {w}x{h}x{c} | op {} | 블롭 {:.2}MB | {:.3} GMAC\n\
         프레임타임 **{ms:.1}ms** ({:.1} GMAC/s)",
        sw.ops.len(),
        blob.len() as f64 / 1e6,
        mac as f64 / 1e9,
        mac as f64 / ms * 1e-6,
    );
}
