//! 유니폼 params 테이블 — op당 256B 슬롯 하나, 로드 시 1회 기록,
//! 디스패치는 dynamic offset 하나만 바꾼다. 프레임 루프의 유니폼 버퍼 생성은 0.

use crate::context::GpuContext;

/// 슬롯 크기 = WebGPU min_uniform_buffer_offset_alignment 기본값
pub const SLOT_BYTES: u32 = 256;

pub struct ParamsTable {
    pub buffer: wgpu::Buffer,
    slots: u32,
}

impl ParamsTable {
    pub fn create(ctx: &GpuContext, slots: u32) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params-table"),
            size: slots.max(1) as u64 * SLOT_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, slots: slots.max(1) }
    }

    /// 슬롯에 params 기록 (로드 시 1회)
    pub fn write(&self, ctx: &GpuContext, slot: u32, data: &[u8]) {
        assert!(slot < self.slots, "params 슬롯 범위 초과");
        assert!(data.len() as u32 <= SLOT_BYTES);
        ctx.queue.write_buffer(&self.buffer, slot as u64 * SLOT_BYTES as u64, data);
    }

    /// OpDispatch.param_offset에 넣을 dynamic offset
    pub fn offset(slot: u32) -> u32 {
        slot * SLOT_BYTES
    }

    /// binding 0에 넣을 바인딩 (dynamic offset 기준점, 크기는 슬롯 하나)
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: Some(std::num::NonZeroU64::new(SLOT_BYTES as u64).unwrap()),
        })
    }
}
