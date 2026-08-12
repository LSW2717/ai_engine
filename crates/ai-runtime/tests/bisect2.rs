//! 프레임2 이분탐색 — AI_RT_NO_REUSE=1로 2프레임 실행 후 전 텐서를 cpu2와 대조.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_runtime::Model;

#[test]
#[ignore]
fn bisect_rvm_frame2() {
    unsafe { std::env::set_var("AI_RT_NO_REUSE", "1") };
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
    let tid_of = |name: &str| {
        sw.tensors.iter().position(|t| t.name == name).map(|i| i as u32).unwrap()
    };

    let mut cpu1 = CpuExec::new(&sw, &blob);
    cpu1.set_input(tid_of("src"), input.clone());
    cpu1.run().unwrap();
    let mut cpu2 = CpuExec::new(&sw, &blob);
    cpu2.set_input(tid_of("src"), input.clone());
    for s in &sw.states {
        cpu2.set_input(s.input, cpu1.read(s.output).unwrap());
    }
    cpu2.run().unwrap();

    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, "src", &input).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();

    for (i, op) in sw.ops.iter().enumerate() {
        let out_tid = match op {
            ai_core::format::SwOp::Conv { out, .. }
            | ai_core::format::SwOp::Binary { out, .. }
            | ai_core::format::SwOp::Gpool { out, .. }
            | ai_core::format::SwOp::Avgpool { out, .. }
            | ai_core::format::SwOp::Resize { out, .. }
            | ai_core::format::SwOp::Concat { out, .. }
            | ai_core::format::SwOp::Chcopy { out, .. }
            | ai_core::format::SwOp::Act { out, .. }
            | ai_core::format::SwOp::Mix { out, .. } => *out,
        };
        // 상태 텐서의 debug_read는 parity 무시라 짝수 프레임 후 stale — 건너뜀
        if sw.states.iter().any(|s| s.output == out_tid || s.input == out_tid) {
            continue;
        }
        let got = pollster::block_on(model.debug_read_tensor(&ctx, out_tid)).unwrap();
        let want = cpu2.read(out_tid).unwrap();
        let max_err =
            got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        if max_err > 5e-3 {
            let name = &sw.tensors[out_tid as usize].name;
            panic!("프레임2 최초 발산 op[{i}] out={name} max_err={max_err:.3e}\nop: {op:?}");
        }
    }
    println!("프레임2 발산 없음?!");
}
