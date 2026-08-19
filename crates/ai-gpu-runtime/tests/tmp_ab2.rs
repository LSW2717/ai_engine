//! 임시 진단 (삭제 예정) — no-fgr 모델 min-of-8 프레임타임
use ai_core::rng::XorShift32;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

#[test]
#[ignore]
fn tmp_ab2() {
    let ctx = GpuContext::new_blocking().unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bytes = std::fs::read(root.join("web/models/rvm_256x144.sw")).unwrap();
    let mut model = pollster::block_on(Model::load(&ctx, &bytes)).unwrap();
    let inp = XorShift32::new(7).vec_f32(144 * 256 * 3);
    model.upload_input(&ctx, "src", &inp).unwrap();
    for _ in 0..5 {
        pollster::block_on(model.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let mut best = f64::MAX;
    for _ in 0..8 {
        let t = std::time::Instant::now();
        for _ in 0..10 {
            pollster::block_on(model.infer(&ctx)).unwrap();
        }
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 100.0);
    }
    println!("no-fgr min frame {best:.3}ms");
}
