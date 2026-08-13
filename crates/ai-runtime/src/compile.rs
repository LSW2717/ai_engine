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
    // dedupe + 컨텍스트 캐시 히트 분리 (재로드·다중 모델은 컴파일을 건너뛴다)
    let mut unique: Vec<&dyn KernelSpec> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut map = HashMap::new();
    {
        let cache = ctx.kernel_cache.lock().unwrap();
        for s in specs {
            let key = s.cache_key(&ctx.caps);
            if !seen.insert(key.clone()) {
                continue;
            }
            if let Some(k) = cache.get(&key) {
                map.insert(key, k.clone());
            } else {
                unique.push(*s);
            }
        }
    }
    log::info!(
        "[ai-runtime] 파이프라인 {}개 (고유 신규 {}, 캐시 {})",
        specs.len(),
        unique.len(),
        map.len()
    );

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
        // ⚠ 여기가 콜드 로드의 전부다 (브라우저 실측: fetch 25ms vs 컴파일 2000ms).
        // wgpu에 async 파이프라인 생성이 없어(gfx-rs/wgpu#3794) createComputePipeline이
        // 동기로 불리고, 고유 파이프라인 수만큼 메인 스레드에서 직렬 컴파일된다.
        // 모듈을 먼저 다 만들어도 비싼 일(Tint→MSL→Metal)은 파이프라인 단계에 있어
        // 병렬화되지 않는다. AI_PIPE_TIMING 빌드에서 개당 시간을 찍어 어느 셰이더가
        // 비싼지 본다.
        // 모듈 생성(Tint 파싱·검증)과 파이프라인 생성(→MSL→Metal)을 따로 잰다.
        // 둘 다 재야 한다 — 한쪽만 재면 "컴파일이 공짜"라는 오답이 나온다.
        let mut t_mod = 0.0f64;
        let mut pending = Vec::with_capacity(unique.len());
        for s in &unique {
            let t0 = web_time::Instant::now();
            pending.push(kernel::begin_compile(ctx, *s));
            t_mod += t0.elapsed().as_secs_f64() * 1e3;
        }
        let mut timings: Vec<(f64, String)> = Vec::new();
        for p in pending {
            let t0 = web_time::Instant::now();
            let k = kernel::finish_compile(ctx, p);
            timings.push((t0.elapsed().as_secs_f64() * 1e3, k.key.clone()));
            map.insert(k.key.clone(), Arc::new(k));
        }
        timings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let t_pipe: f64 = timings.iter().map(|t| t.0).sum();
        log::info!(
            "[ai-runtime] 셰이더모듈 {t_mod:.0}ms + 파이프라인 {t_pipe:.0}ms = {:.0}ms / {}개",
            t_mod + t_pipe,
            timings.len()
        );
        for (ms, key) in timings.iter().take(10) {
            log::info!("[ai-runtime]   pipe {ms:7.1}ms  {key}");
        }
    }

    // 신규 컴파일분을 컨텍스트 캐시에 반영
    {
        let mut cache = ctx.kernel_cache.lock().unwrap();
        for (k, v) in &map {
            cache.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    Ok(map)
}
