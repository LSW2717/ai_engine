//! GPU 타임스탬프 프로파일러 — 패스 경계 timestamp_writes만 사용한다.
//! (encoder 내 write_timestamp는 Metal/WebGPU에서 이식성이 없다)
//!
//! Chrome은 기본 100µs 양자화라, 벤치는 한 패스에 N회 디스패치를 몰아넣고
//! 나눠서 해상도를 확보한다. TIMESTAMP_QUERY 미지원 기기는 None → wall-clock 폴백.

use crate::context::GpuContext;
use crate::readback;

pub struct GpuProfiler {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    capacity_scopes: u32,
}

impl GpuProfiler {
    /// capacity_scopes = 측정 구간(패스) 수. 미지원 기기는 None.
    pub fn new(ctx: &GpuContext, capacity_scopes: u32) -> Option<Self> {
        if !ctx.caps.timestamps {
            return None;
        }
        let queries = capacity_scopes * 2;
        let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let size = queries as u64 * 8;
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler-resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self { qs, resolve, read, capacity_scopes })
    }

    /// scope번 패스의 timestamp_writes 기술자
    pub fn timestamp_writes(&self, scope: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        assert!(scope < self.capacity_scopes);
        wgpu::ComputePassTimestampWrites {
            query_set: &self.qs,
            beginning_of_pass_write_index: Some(scope * 2),
            end_of_pass_write_index: Some(scope * 2 + 1),
        }
    }

    /// 사용한 scope들의 쿼리를 resolve + 리드백 버퍼로 복사 (submit 전에 호출)
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder, scopes_used: u32) {
        let queries = scopes_used * 2;
        enc.resolve_query_set(&self.qs, 0..queries, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, queries as u64 * 8);
    }

    /// scope별 (end - begin) 밀리초
    pub async fn read_ms(
        &self,
        ctx: &GpuContext,
        scopes_used: u32,
    ) -> Result<Vec<f64>, String> {
        let bytes = readback::read_buffers(ctx, &[&self.read]).await?.remove(0);
        let ticks: &[u64] = bytemuck::cast_slice(&bytes);
        let period_ns = ctx.queue.get_timestamp_period() as f64;
        let mut out = Vec::with_capacity(scopes_used as usize);
        for s in 0..scopes_used as usize {
            let dt = ticks[s * 2 + 1].saturating_sub(ticks[s * 2]);
            out.push(dt as f64 * period_ns / 1.0e6);
        }
        Ok(out)
    }
}
