//! P2-9 게이트: segm 모델(상태 없음, 45 ops)을 GPU executor로 실행,
//! ai-convert의 CPU 레퍼런스 실행기와 대조.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

#[test]
fn segm_gpu_matches_cpu_reference() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    if !path.exists() {
        eprintln!("skip: segm onnx 없음");
        return;
    }
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };

    // 변환
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx::default();
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "segm").unwrap();
    let container = sw.write_container(&blob).unwrap();

    // 시드 입력 (논리 NHWC)
    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let input = XorShift32::new(77).vec_f32((t.h * t.w * t.c) as usize);
    let in_name = t.name.clone();

    // CPU 레퍼런스
    let mut cpu = CpuExec::new(&sw, &blob);
    cpu.set_input(in_tid, input.clone());
    cpu.run().unwrap();

    // GPU
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    println!(
        "load: ops {} | 고유 파이프라인 {} | 슬롯 {} | arena {:.1}MB | {:.0}ms",
        model.report.ops,
        model.report.unique_pipelines,
        model.report.slots,
        model.report.arena_bytes as f64 / 1e6,
        model.report.load_ms
    );
    model.upload_input(&ctx, &in_name, &input).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();

    // 출력 대조
    for &out_tid in &sw.outputs {
        let name = sw.tensors[out_tid as usize].name.clone();
        let got = pollster::block_on(model.read_output(&ctx, &name)).unwrap();
        let want = cpu.read(out_tid).unwrap();
        assert_eq!(got.len(), want.len());
        let mut max_err = 0f32;
        for (g, w) in got.iter().zip(&want) {
            max_err = max_err.max((g - w).abs());
        }
        println!("{name}: max_err {max_err:.3e}");
        assert!(max_err < 2e-3, "{name} 발산: {max_err}");
    }
}
