//! 파이프라인 일괄 컴파일 — 캐시 키 dedupe 후 네이티브는 스레드 팬아웃, wasm은 순차.
//!
//! Phase 1 실측: 파이프라인 생성 5~90ms/개 → RVM급(고유 수십 개)은 병렬화가 로드
//! 시간을 지배적으로 줄인다. 팬아웃 경로는 error scope를 쓰지 않는다(디바이스 전역
//! 스택이라 멀티스레드에서 얽힘) — 오류는 uncaptured 로그 + 워밍업에서 드러난다.

use std::collections::HashMap;
use std::sync::Arc;

use ai_gpu::kernel::{self, CompiledKernel, KernelSpec};
use ai_gpu::GpuContext;

use crate::error::RuntimeError;

pub async fn compile_all(
    ctx: &GpuContext,
    specs: &[&dyn KernelSpec],
) -> Result<HashMap<String, Arc<CompiledKernel>>, RuntimeError> {
    // dedupe
    let mut unique: Vec<&dyn KernelSpec> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for s in specs {
        let key = s.cache_key(&ctx.caps);
        if seen.insert(key) {
            unique.push(*s);
        }
    }
    log::info!("[ai-runtime] 파이프라인 {}개 (고유 {})", specs.len(), unique.len());

    let mut map = HashMap::new();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        let chunk = unique.len().div_ceil(threads.max(1));
        let compiled: Vec<CompiledKernel> = std::thread::scope(|s| {
            let handles: Vec<_> = unique
                .chunks(chunk.max(1))
                .map(|specs| {
                    s.spawn(move || {
                        specs
                            .iter()
                            .map(|spec| kernel::compile_unscoped(ctx, *spec))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for k in compiled {
            map.insert(k.key.clone(), Arc::new(k));
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        for spec in unique {
            let k = kernel::compile(ctx, spec)
                .await
                .map_err(RuntimeError::Gpu)?;
            map.insert(k.key.clone(), Arc::new(k));
        }
    }

    Ok(map)
}
