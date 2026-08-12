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
    // 추격 구성 = fp16 (AI_PROFILE_FP32=1로 fp32 프로파일)
    let fp16 = std::env::var("AI_PROFILE_FP32").is_err() && ctx.caps.f16;
    let cctx = Ctx {
        size: Some((256, 144)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: vec![
            ("r1i".into(), "r1o".into()),
            ("r2i".into(), "r2o".into()),
            ("r3i".into(), "r3o".into()),
            ("r4i".into(), "r4o".into()),
        ],
        fp16,
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
    // 예산표용 전체 덤프 (실행 순서 = plan op 순서)
    println!("-- 전체 op 덤프 (BUDGET) --");
    for (i, (label, ms)) in prof.iter().enumerate() {
        println!("BUDGET\t{i}\t{:.1}\t{label}", ms * 1e3);
    }
}

/// 추격 기준 256×144 프레임타임 (webgl2-engine-span 1.65ms가 목표)
#[test]
#[ignore]
fn rvm_256x144_frametime() {
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
        "256×144 프레임타임: {:.3}ms/frame (op {}개)",
        t0.elapsed().as_secs_f64() * 1e3 / 50.0,
        model.report.ops
    );
}

/// f16 전그래프 A/B: fp32 대비 출력 오차 + 프레임타임 (2ms 추격의 구조 카드)
#[test]
#[ignore]
fn rvm_256x144_fp16() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    if !ctx.caps.f16 {
        eprintln!("skip: f16 미지원 기기");
        return;
    }
    let input = XorShift32::new(7).vec_f32((144 * 256 * 3) as usize);
    let mut outs: Vec<(bool, Vec<f32>, f64)> = Vec::new();
    for fp16 in [false, true] {
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
            fp16,
            ..Default::default()
        };
        run_full(&mut g, &cctx).unwrap();
        let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
        let container = sw.write_container(&blob).unwrap();
        let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
        model.upload_input(&ctx, "src", &input).unwrap();
        for _ in 0..5 {
            pollster::block_on(model.infer(&ctx)).unwrap();
        }
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            pollster::block_on(model.infer(&ctx)).unwrap();
        }
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 50.0;
        let pha = pollster::block_on(model.read_output(&ctx, "pha")).unwrap();
        outs.push((fp16, pha, ms));
    }
    let (_, pha32, ms32) = &outs[0];
    let (_, pha16, ms16) = &outs[1];
    let max_err =
        pha32.iter().zip(pha16).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!(
        "fp32 {ms32:.3}ms | fp16 {ms16:.3}ms ({:.2}x) | pha max_err {max_err:.4}",
        ms32 / ms16
    );
    assert!(max_err < 0.05, "f16 정확도 파괴: max_err {max_err}");
}

/// 로드 시간 분해 (진단용): 같은 프로세스에서 2회 로드 — 1회차와 2회차 차이가
/// 파이프라인 컴파일 비용, 2회차 잔여가 버퍼/바인드그룹/업로드 비용이다.
#[test]
#[ignore]
fn rvm_load_times() {
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
        fp16: ctx.caps.f16,
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    for i in 0..3 {
        let model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
        println!(
            "로드 {}회차: {:.0}ms (파이프라인 {}개, arena {:.1}MB)",
            i + 1,
            model.report.load_ms,
            model.report.unique_pipelines,
            model.report.arena_bytes as f64 / 1e6
        );
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
