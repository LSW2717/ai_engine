//! per-op 예산표 — 실측(rep별 min) vs 이론하한 max(bytes/BW, flops/PEAK).
//! GPU 쪽 예산표 방법론(profile_rvm BUDGET)의 CPU판.
//!
//! 사용: `AI_ONNX=<경로> [AI_REPS=30] cargo test --release -p ai-cpu --test prof_cpu -- --ignored --nocapture`
//!
//! 피크 가정(1스레드, 환경변수로 조정): AI_PEAK_GFLOPS(기본 50 — M1 P코어
//! 4×128b FMA), AI_PEAK_GBS(기본 60 — 단일코어 L2/SLC 대역). 하한은 순위용
//! 근사지 절대 진리가 아니다 — "실측/하한" 배율이 큰 op부터 판다.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::rng::XorShift32;

#[test]
#[ignore]
fn budget_table() {
    let path = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx 경로> 필요");
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "prof").unwrap();

    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let name = t.name.clone();
    let input = XorShift32::new(7).vec_f32((t.h * t.w * t.c) as usize);

    let container = sw.write_container(&blob).unwrap();
    let mut m = ai_cpu::Model::load(&container).unwrap();
    m.set_input(&name, &input).unwrap();
    for _ in 0..3 {
        m.infer().unwrap();
    }

    let reps: usize = std::env::var("AI_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let mut rows = m.infer_profiled();
    for _ in 1..reps {
        for (acc, r) in rows.iter_mut().zip(m.infer_profiled()) {
            if r.ms < acc.ms {
                acc.ms = r.ms;
            }
        }
    }

    let peak_gflops: f64 =
        std::env::var("AI_PEAK_GFLOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(50.0);
    let peak_gbs: f64 =
        std::env::var("AI_PEAK_GBS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0);

    // 같은 라벨의 op는 합쳐서 본다 (동형 conv가 많다)
    use std::collections::HashMap;
    let mut agg: HashMap<String, (f64, f64, f64, usize)> = HashMap::new();
    for r in &rows {
        let e = agg.entry(r.label.clone()).or_insert((0.0, 0.0, 0.0, 0));
        e.0 += r.ms;
        e.1 += r.mflop;
        e.2 += r.mb;
        e.3 += 1;
    }
    let mut list: Vec<_> = agg.into_iter().collect();
    list.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));

    let total_ms: f64 = rows.iter().map(|r| r.ms).sum();
    // MFLOP / (GFLOP/s) = ms, MB / (GB/s) = ms — 단위가 딱 맞는다
    let total_bound: f64 =
        rows.iter().map(|r| (r.mflop / peak_gflops).max(r.mb / peak_gbs)).sum();

    println!("\n== ai-cpu 예산표 (1스레드, rep별 min, 하한: {peak_gflops:.0}GF/s·{peak_gbs:.0}GB/s) ==");
    println!("{:>8} {:>7} {:>8} {:>6} {:>5}  라벨", "ms", "하한ms", "GF/s", "배율", "×n");
    let mut cum = 0.0;
    for (label, (ms, mflop, mb, n)) in &list {
        let bound = (mflop / peak_gflops).max(mb / peak_gbs);
        cum += ms;
        println!(
            "{:>8.3} {:>7.3} {:>8.1} {:>6.1} {:>5}  {label}  (누적 {:.0}%)",
            ms,
            bound,
            mflop / ms,
            ms / bound,
            n,
            cum / total_ms * 100.0
        );
    }
    println!(
        "\n합계 {total_ms:.2}ms | 하한 합 {total_bound:.2}ms | 전체 배율 {:.1}x | op {}개",
        total_ms / total_bound,
        rows.len()
    );
}
