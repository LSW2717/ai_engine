//! 활성화 텐서 풀 — 로드 시 텐서당 STORAGE 버퍼를 생성, 프레임 루프 할당 0.
//!
//! ## 왜 단일 arena 버퍼가 아닌가
//! WebGPU 사용 규칙: 한 dispatch(synchronization scope) 안에서 같은 버퍼를
//! writable storage와 다른 용도(read-only storage 포함)로 동시에 쓸 수 없다 —
//! 범위가 겹치지 않아도 검증은 **버퍼 단위**다. 단일 arena에 입력(뷰)과 출력(뷰)이
//! 같이 살면 모든 conv dispatch가 이 규칙에 걸린다(Dawn/Chrome 하드 실패,
//! 네이티브 wgpu는 배리어 누락으로 조용히 오답). 그래서:
//! - **활성화 텐서 = 버퍼 풀(텐서당 1버퍼)** — usage scope가 버퍼 단위로 깨끗해지고
//!   wgpu가 RW→READ 전이 배리어를 dispatch마다 정확히 삽입한다.
//! - **가중치 = read-only 단일 버퍼 + 오프셋 바인딩**(weights.rs 아님, 로더 소관) —
//!   read-only usage는 합법적으로 병합된다.
//!
//! Phase 2 liveness 플래너는 "수명이 겹치지 않는 텐서들이 풀 슬롯을 공유"하는
//! 방식으로 같은 `TensorView` 계약 뒤에 드롭인된다(생산 dispatch가 마지막 소비
//! dispatch보다 뒤일 때만 재사용 → 같은 dispatch 안의 앨리어싱은 구조적으로 배제).

use ai_core::TensorDesc;

use crate::context::GpuContext;

/// 풀 안의 텐서 하나 — 슬롯 인덱스 + desc
#[derive(Clone, Copy, Debug)]
pub struct TensorView {
    pub desc: TensorDesc,
    pub slot: usize,
}

/// 로드 시 풀 구성을 계획한다 (Phase 2: liveness 기반 슬롯 공유가 여기 들어옴)
#[derive(Default)]
pub struct ArenaPlanner {
    sizes: Vec<u64>,
}

impl ArenaPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, desc: TensorDesc) -> TensorView {
        let slot = self.sizes.len();
        self.sizes.push(desc.size_bytes());
        TensorView { desc, slot }
    }

    /// 계획된 총 바이트 (통계/로그용)
    pub fn total(&self) -> u64 {
        self.sizes.iter().sum()
    }

    pub fn slots(&self) -> &[u64] {
        &self.sizes
    }
}

pub struct Arena {
    buffers: Vec<wgpu::Buffer>,
}

impl Arena {
    pub fn create(ctx: &GpuContext, planner: &ArenaPlanner) -> Result<Self, String> {
        let max = ctx.caps.limits.max_storage_buffer_binding_size as u64;
        let mut buffers = Vec::with_capacity(planner.slots().len());
        for (i, &size) in planner.slots().iter().enumerate() {
            if size > max {
                return Err(format!("텐서 슬롯 {i} 크기 {size}B가 바인딩 한도 {max}B 초과"));
            }
            buffers.push(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("tensor-{i}")),
                size: size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        Ok(Self { buffers })
    }

    /// 패킹된 텐서 바이트 업로드 (로드/입력 경로)
    pub fn upload(&self, ctx: &GpuContext, view: &TensorView, bytes: &[u8]) {
        debug_assert_eq!(bytes.len() as u64, view.desc.size_bytes());
        ctx.queue.write_buffer(&self.buffers[view.slot], 0, bytes);
    }

    pub fn buffer(&self, view: &TensorView) -> &wgpu::Buffer {
        &self.buffers[view.slot]
    }

    /// bind group entry용 바인딩
    pub fn binding(&self, view: &TensorView) -> wgpu::BindingResource<'_> {
        self.buffers[view.slot].as_entire_binding()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::DType;

    #[test]
    fn planner_tracks_slots_and_total() {
        let mut p = ArenaPlanner::new();
        let a = p.alloc(TensorDesc::new(1, 1, 1, DType::F32));
        let b = p.alloc(TensorDesc::new(3, 5, 6, DType::F32));
        assert_eq!(a.slot, 0);
        assert_eq!(b.slot, 1);
        assert_eq!(p.total(), 16 + 480);
    }
}
