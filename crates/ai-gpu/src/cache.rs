//! 파이프라인 캐시 — cache_key(codegen 입력 전체) → CompiledKernel.
//!
//! 같은 shape 시그니처의 레이어들이 파이프라인을 공유한다(RVM에서 shape 반복이 많음).
//! 로드 흐름: spec 수집 → 키 dedupe → 미스만 컴파일 → 워밍업 디스패치(백엔드 컴파일 강제).

use std::collections::HashMap;
use std::sync::Arc;

use crate::context::GpuContext;
use crate::kernel::{self, CompiledKernel, KernelSpec};

#[derive(Default)]
pub struct PipelineCache {
    map: HashMap<String, Arc<CompiledKernel>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get_or_compile(
        &mut self,
        ctx: &GpuContext,
        spec: &dyn KernelSpec,
    ) -> Result<Arc<CompiledKernel>, String> {
        let key = spec.cache_key(&ctx.caps);
        if let Some(k) = self.map.get(&key) {
            return Ok(k.clone());
        }
        let compiled = Arc::new(kernel::compile(ctx, spec).await?);
        self.map.insert(key, compiled.clone());
        Ok(compiled)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
