//! op 격리 반복 측정 프로파일 (진단용, #[ignore]).
//!
//! Apple GPU는 컴퓨트 패스 타임스탬프를 못 믿는다. 대신 op 하나만 수백 번 반복
//! 제출해 wall로 나눈다. 합계를 실제 프레임타임과 대조해 신뢰도를 확인한다.
//!
//! `AI_PROFILE_FP32=1`이면 fp32, 기본은 가중치 fp16 (추격 구성).

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

/// 기본은 **배포되는 그래프**(고정 export, 116 op)다. 공식 export는 변환기가
/// 25개 op을 더 남기므로(canon 갭) 그걸 프로파일하면 배포판에 없는 op을 튜닝하게 된다.
/// `AI_ONNX=models/rvm_fp32.onnx`로 공식 그래프도 볼 수 있다.
fn build(ctx: &GpuContext, fp16_weights: bool) -> (Model, &'static str) {
    let rel = std::env::var("AI_ONNX")
        .unwrap_or_else(|_| "models/rvm_fixed_144x256.onnx".into());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(&rel);
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    // 공식 export는 src/r1i~r4i + downsample_ratio, 고정 export는 input_1/r1~r4
    let official = g.inputs.iter().any(|n| n == "src");
    let (in_name, suffix) = if official { ("src", "i") } else { ("input_1", "") };
    let cctx = Ctx {
        size: Some((256, 144)),
        set_inputs: if official {
            vec![("downsample_ratio".into(), 1.0)]
        } else {
            vec![]
        },
        states: (1..=4).map(|i| (format!("r{i}{suffix}"), format!("r{i}o"))).collect(),
        fp16_weights,
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let mut model = pollster::block_on(Model::load(ctx, &container)).unwrap();
    let input = XorShift32::new(7).vec_f32(144 * 256 * 3);
    model.upload_input(ctx, in_name, &input).unwrap();
    (model, in_name)
}

#[test]
#[ignore]
fn prof_isolated() {
    let ctx = GpuContext::new_blocking().unwrap();
    let fp16_w = std::env::var("AI_PROFILE_FP32").is_err() && ctx.caps.f16;
    let (mut model, _) = build(&ctx, fp16_w);
    let n = model.op_labels().len();

    // 기준 프레임타임 (큐드 스루풋) — 3라운드 최소값.
    // 단발 측정은 이 기기에서 ±10% 흔들린다. 최소값이 "방해 없는 실행"에 가장 가깝다.
    let mut frame_ms = f64::MAX;
    for _ in 0..3 {
        for _ in 0..5 {
            pollster::block_on(model.infer(&ctx)).unwrap();
        }
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            pollster::block_on(model.infer(&ctx)).unwrap();
        }
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
        frame_ms = frame_ms.min(t0.elapsed().as_secs_f64() * 1e3 / 50.0);
    }

    let labels: Vec<String> = model.op_labels().to_vec();
    let mut costs = Vec::with_capacity(n);
    for i in 0..n {
        // 큰 op은 반복을 줄여 총 측정시간을 억제
        let probe = pollster::block_on(model.bench_op(&ctx, i, 64)).unwrap();
        let reps = if probe > 0.05 { 64 } else { 400 };
        let mut best = probe;
        for _ in 0..3 {
            best = best.min(pollster::block_on(model.bench_op(&ctx, i, reps)).unwrap());
        }
        costs.push(best);
    }
    let sum: f64 = costs.iter().sum();
    println!(
        "\n== 격리 프로파일 (fp16_w={fp16_w}) — 프레임 {frame_ms:.3}ms / 격리합 {sum:.3}ms (비 {:.2}) ==",
        sum / frame_ms
    );

    let mut by_kind: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
    for (l, c) in labels.iter().zip(&costs) {
        let e = by_kind.entry(l.split_whitespace().next().unwrap_or("?").into()).or_insert((0.0, 0));
        e.0 += c;
        e.1 += 1;
    }
    for (k, (ms, cnt)) in &by_kind {
        println!("{k:>14}: {:7.1}µs ({cnt}개, {:5.1}%)", ms * 1e3, ms / sum * 100.0);
    }

    println!("\n-- 비싼 순 25개 --");
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| costs[*b].partial_cmp(&costs[*a]).unwrap());
    for &i in order.iter().take(25) {
        println!("{:7.1}µs  [{i:3}] {}", costs[i] * 1e3, labels[i]);
    }
    println!("\n-- 전체 (실행 순서) --");
    for i in 0..n {
        println!("ISO\t{i}\t{:.2}\t{}", costs[i] * 1e3, labels[i]);
    }
}
