//! 디버그 이분탐색 — AI_RT_NO_REUSE=1로 전 텐서를 CPU 레퍼런스와 대조해
//! 최초 발산 텐서를 찾는다. (평시 #[ignore])

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_runtime::Model;

#[test]
#[ignore]
fn bisect_segm() {
    unsafe { std::env::set_var("AI_RT_NO_REUSE", "1") };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/segm_mnv4s050_s2_160x288_nhwc.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx::default();
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "segm").unwrap();
    let container = sw.write_container(&blob).unwrap();

    let in_tid = sw.inputs[0];
    let t = &sw.tensors[in_tid as usize];
    let input = XorShift32::new(77).vec_f32((t.h * t.w * t.c) as usize);
    let in_name = t.name.clone();

    let mut cpu = CpuExec::new(&sw, &blob);
    cpu.set_input(in_tid, input.clone());
    cpu.run().unwrap();

    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, &in_name, &input).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();

    // op 순서대로 출력 텐서 대조 → 최초 발산 지점
    for (i, op) in sw.ops.iter().enumerate() {
        let out_tid = match op {
            ai_core::format::SwOp::Conv { out, .. }
            | ai_core::format::SwOp::Binary { out, .. }
            | ai_core::format::SwOp::Gpool { out, .. }
            | ai_core::format::SwOp::Avgpool { out, .. }
            | ai_core::format::SwOp::Maxpool { out, .. }
            | ai_core::format::SwOp::Resize { out, .. }
            | ai_core::format::SwOp::Concat { out, .. }
            | ai_core::format::SwOp::Chcopy { out, .. }
            | ai_core::format::SwOp::SeGate { out, .. }
            | ai_core::format::SwOp::Act { out, .. }
            | ai_core::format::SwOp::Mix { out, .. } => *out,
        };
        let got = pollster::block_on(model.debug_read_tensor(&ctx, out_tid)).unwrap();
        let want = cpu.read(out_tid).unwrap();
        let max_err =
            got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        let name = &sw.tensors[out_tid as usize].name;
        if max_err > 2e-3 {
            panic!("최초 발산 op[{i}] out={name} max_err={max_err:.3e}\nop: {op:?}");
        }
    }
    println!("발산 없음?!");
}
