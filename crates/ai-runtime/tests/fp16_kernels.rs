//! fp16 컨테이너의 전 커널을 에러 스코프로 개별 컴파일 — 실패 커널 색출 (진단용)

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::format::SwModel;
use ai_gpu::GpuContext;

#[test]
#[ignore]
fn fp16_compile_all() {
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
        fp16: true,
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let (sw2, _) = SwModel::parse_container(&container).unwrap();

    let mut seen = std::collections::HashSet::new();
    let mut fails = 0;
    for op in &sw2.ops {
        let lo = match ai_runtime::lowering::lower_op(&sw2, op, &|t| t) {
            Ok(lo) => lo,
            Err(e) => {
                println!("LOWER-FAIL: {e}");
                fails += 1;
                continue;
            }
        };
        let key = lo.spec.cache_key(&ctx.caps);
        if !seen.insert(key.clone()) {
            continue;
        }
        if let Err(e) = pollster::block_on(ai_gpu::kernel::compile(&ctx, lo.spec.as_ref())) {
            println!("COMPILE-FAIL: {}", e.lines().take(6).collect::<Vec<_>>().join(" | "));
            fails += 1;
        }
    }
    println!("고유 커널 {}개 중 실패 {}개", seen.len(), fails);
    assert_eq!(fails, 0);
}
