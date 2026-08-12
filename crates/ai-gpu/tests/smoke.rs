//! 디바이스 초기화 + 버퍼 왕복 스모크 테스트.
//! adapter가 없는 환경(CI 등)에서는 실패가 아니라 skip.

use ai_gpu::{readback, GpuContext};

#[test]
fn context_init_and_buffer_roundtrip() {
    let ctx = match GpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    println!(
        "adapter: {} ({:?}), f16={}, timestamps={}, storage_align={}",
        ctx.caps.info.name, ctx.caps.info.backend, ctx.caps.f16, ctx.caps.timestamps,
        ctx.caps.storage_align
    );

    let data: Vec<u8> = (0..=255).collect();
    let storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke-storage"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke-staging"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    ctx.queue.write_buffer(&storage, 0, &data);
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("smoke") });
    enc.copy_buffer_to_buffer(&storage, 0, &staging, 0, 256);
    ctx.queue.submit([enc.finish()]);

    let out = pollster::block_on(readback::read_buffers(&ctx, &[&staging])).unwrap();
    assert_eq!(out[0], data);
}
