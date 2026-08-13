//! 타임스탬프 쿼리 진단 — 원시 tick과 period를 그대로 찍는다.
//! (이 머신에서 gpu_min이 0으로 나오는 원인 규명용, #[ignore])

use ai_gpu::GpuContext;

#[test]
#[ignore]
fn diag_timestamp() {
    let ctx = GpuContext::new_blocking().unwrap();
    println!("adapter: {}", ctx.caps.info.name);
    println!("caps.timestamps: {}", ctx.caps.timestamps);
    println!("queue.get_timestamp_period(): {}", ctx.queue.get_timestamp_period());
    println!(
        "features TIMESTAMP_QUERY: {}",
        ctx.device.features().contains(ai_gpu::wgpu::Features::TIMESTAMP_QUERY)
    );
}
