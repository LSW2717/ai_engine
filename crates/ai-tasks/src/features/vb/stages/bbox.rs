//! 인물 bbox 리덕션 — 컴퓨트 + 20B 비동기 리드백 링 (프레이밍 입력).
//!
//! v-ai 엔진 티어의 PBO+fence 링(37KB 마스크 리드백 → CPU 스캔) 대응:
//! WebGPU에선 리덕션을 GPU에서 하고 **20B만** 내린다. 링 2슬롯 — 쓴 버퍼는
//! 회수 후에만 재기록(v-ai와 같은 규율). 동기 대기 0 — 결과는 수 프레임
//! 지연으로 도착하고 프레이밍 스무딩이 그 지연을 흡수한다.

use std::sync::{Arc, Mutex};

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::framing::BBox;

const SRC: &str = include_str!("../shaders/bbox.wgsl");
/// min=∞, max=0, count=0 — 디스패치 전 초기화 규약 (bbox.wgsl 헤더)
const INIT: [u32; 5] = [0xffff_ffff, 0, 0xffff_ffff, 0, 0];
const BYTES: u64 = 20;

#[derive(Clone, Copy, PartialEq)]
enum SlotState {
    Idle,
    /// map_async 발행됨 — 콜백 대기
    Pending,
    /// 콜백 완료 — 프레임 루프가 회수(get_mapped_range + unmap)
    Mapped,
}

struct RingSlot {
    buf: wgpu::Buffer,
    state: Arc<Mutex<SlotState>>,
}

pub(crate) struct BboxStage {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
    storage: wgpu::Buffer,
    ring: [RingSlot; 2],
}

impl BboxStage {
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vb-bbox"),
            source: wgpu::ShaderSource::Wgsl(SRC.into()),
        });
        // 명시적 레이아웃 — auto-layout은 파이프라인마다 별개 정체성 (마스크 전멸
        // 사고의 교훈, NEXT.md)
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vb-bbox"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vb-bbox"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline =
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vb-bbox"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            });
        let storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vb-bbox-out"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = [0, 1].map(|i| RingSlot {
            buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("vb-bbox-read{i}")),
                size: BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: Arc::new(Mutex::new(SlotState::Idle)),
        });
        BboxStage { pipeline, bgl, storage, ring }
    }

    pub fn storage(&self) -> &wgpu::Buffer {
        &self.storage
    }

    /// 프레임 시작: 완료 슬롯 회수 → 최신 스캔 결과.
    /// 반환 None = 이번 프레임 도착분 없음(직전 값 유지), Some(None) = 인물 없음.
    pub fn pump(&mut self, mw: u32, mh: u32) -> Option<Option<BBox>> {
        let mut latest = None;
        for slot in &self.ring {
            let mut st = slot.state.lock().unwrap();
            if *st != SlotState::Mapped {
                continue;
            }
            let v: [u32; 5] = {
                let range = slot.buf.slice(..).get_mapped_range().unwrap();
                let mut v = [0u32; 5];
                for (i, c) in range.chunks_exact(4).take(5).enumerate() {
                    v[i] = u32::from_le_bytes(c.try_into().unwrap());
                }
                v
            };
            slot.buf.unmap();
            *st = SlotState::Idle;
            // v-ai: 인물 픽셀 1% 미만은 노이즈 — 없음 처리
            latest = Some(if v[4] < mw * mh / 100 {
                None
            } else {
                Some([
                    v[0] as f32 / mw as f32,
                    (v[1] + 1) as f32 / mw as f32,
                    v[2] as f32 / mh as f32,
                    (v[3] + 1) as f32 / mh as f32,
                ])
            });
        }
        latest
    }

    /// 발행 준비: 빈 슬롯 확보 + 스토리지 초기화. None이면 이번 프레임 스킵
    /// (두 슬롯 다 비행 중 — v-ai와 같은 규율)
    pub fn prepare(&mut self, ctx: &GpuContext) -> Option<usize> {
        let idx = self
            .ring
            .iter()
            .position(|s| *s.state.lock().unwrap() == SlotState::Idle)?;
        ctx.queue.write_buffer(&self.storage, 0, bytemuck::cast_slice(&INIT));
        Some(idx)
    }

    /// 리덕션 디스패치 + 스토리지 → 링 슬롯 복사 (ingest 패스 뒤에 인코드)
    pub fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        mw: u32,
        mh: u32,
        slot: usize,
    ) {
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vb-bbox"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(mw.div_ceil(16), mh.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&self.storage, 0, &self.ring[slot].buf, 0, BYTES);
    }

    /// 제출 후 호출 — 논블로킹 map_async 발행 (완료는 pump가 회수)
    pub fn map(&self, slot: usize) {
        let s = &self.ring[slot];
        *s.state.lock().unwrap() = SlotState::Pending;
        let state = s.state.clone();
        s.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *state.lock().unwrap() =
                if r.is_ok() { SlotState::Mapped } else { SlotState::Idle };
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_valid() {
        let m = naga::front::wgsl::parse_str(super::SRC).unwrap();
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap();
    }
}
