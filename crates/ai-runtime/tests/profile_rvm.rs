//! RVM per-op 프로파일 — 프레임타임 예산 분석 (진단용, #[ignore])

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_runtime::Model;

#[test]
#[ignore]
fn profile_rvm() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
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

    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, "src", &input).unwrap();
    // 워밍업 몇 프레임
    for _ in 0..3 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();

    let prof = pollster::block_on(model.profile_infer(&ctx)).unwrap();
    let total: f64 = prof.iter().map(|(_, ms)| ms).sum();
    println!("== per-op 프로파일 (총 {total:.3}ms, {}개 op) ==", prof.len());

    // 라벨 접두어별 집계
    let mut by_kind: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
    for (label, ms) in &prof {
        let kind = label.split_whitespace().next().unwrap_or("?").to_string();
        let e = by_kind.entry(kind).or_insert((0.0, 0));
        e.0 += ms;
        e.1 += 1;
    }
    for (kind, (ms, n)) in &by_kind {
        println!("{kind:>10}: {ms:8.3}ms ({n}개, 평균 {:.1}µs)", ms / *n as f64 * 1e3);
    }

    println!("-- 상위 15개 op --");
    let mut sorted: Vec<_> = prof.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (label, ms) in sorted.iter().take(15) {
        println!("{:8.1}µs  {label}", ms * 1e3);
    }
}

/// 실전 타겟 512×288 프레임타임 (진단용)
#[test]
#[ignore]
fn rvm_512x288_frametime() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
        size: Some((512, 288)),
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
    let (sw, blob) = lower(&g, &cctx, "rvm512").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let input = XorShift32::new(7).vec_f32((288 * 512 * 3) as usize);
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    println!(
        "512×288 load: {:.0}ms | 파이프라인 {} | arena {:.1}MB",
        model.report.load_ms,
        model.report.unique_pipelines,
        model.report.arena_bytes as f64 / 1e6
    );
    model.upload_input(&ctx, "src", &input).unwrap();
    for _ in 0..3 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..30 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    println!("512×288 프레임타임: {:.3}ms/frame", t0.elapsed().as_secs_f64() * 1e3 / 30.0);
}

/// segm (ORT-web WebGPU 2ms 비교 대상) 프레임타임
#[test]
#[ignore]
fn segm_frametime() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    run_one(&path, "s050");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4_w100m64_160x288_nhwc.onnx");
    run_one(&path, "w100m64");
}

fn run_one(path: &std::path::Path, tag: &str) {
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx::default();
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "segm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let input = XorShift32::new(7).vec_f32((t.h * t.w * t.c) as usize);
    let in_name = t.name.clone();
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, &in_name, &input).unwrap();
    for _ in 0..5 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..50 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    println!(
        "segm[{tag}] 160×288 프레임타임: {:.3}ms/frame (load {:.0}ms, op {}개)",
        t0.elapsed().as_secs_f64() * 1e3 / 50.0,
        model.report.load_ms,
        model.report.ops
    );
}
