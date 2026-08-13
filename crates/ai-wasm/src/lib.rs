//! ai-wasm — wasm-bindgen 경계.
//!
//! 이 크레이트만 JS를 안다. 엔진 컨텍스트는 스레드 로컬 Rc로 보관해
//! async 함수의 'static 요구를 만족시킨다(wasm은 단일 스레드).
//!
//! exports: `is_supported()` / `init_engine()` / `run_tests()` / `run_benchmarks()`
//! — init 실패는 구조화된 메시지로 JS에 전달되어 호스트가 폴백 티어를 결정한다.

use std::cell::RefCell;
use std::rc::Rc;

use ai_gpu::GpuContext;
// 캔버스 서피스는 wasm32 타깃에만 있다 (SurfaceTarget::Canvas가 cfg 게이트됨).
// 게이트하지 않으면 `cargo test --workspace`(네이티브)가 깨진다.
#[cfg(target_arch = "wasm32")]
mod present;

use wasm_bindgen::prelude::*;

thread_local! {
    static ENGINE: RefCell<Option<Rc<GpuContext>>> = const { RefCell::new(None) };
    static MODEL: RefCell<Option<ai_tasks::Segmenter>> = const { RefCell::new(None) };
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

fn engine() -> Result<Rc<GpuContext>, JsValue> {
    ENGINE
        .with(|e| e.borrow().clone())
        .ok_or_else(|| JsValue::from_str("엔진 미초기화 — init_engine()을 먼저 호출"))
}

#[derive(serde::Serialize)]
struct JsAdapterInfo {
    name: String,
    backend: String,
    f16: bool,
    timestamps: bool,
}

#[derive(serde::Serialize)]
struct JsCase {
    name: String,
    passed: bool,
    max_err: f32,
    tol: f32,
}

#[derive(serde::Serialize)]
struct JsBench {
    name: String,
    gpu_min_us: Option<f64>,
    gpu_median_us: Option<f64>,
    wall_us: f64,
    gflops: f64,
    pipeline_ms: f64,
}

/// WebGPU 사용 가능 여부 빠른 판정 (adapter 요청까지 수행)
#[wasm_bindgen]
pub async fn is_supported() -> bool {
    GpuContext::new().await.is_ok()
}

/// 엔진 초기화 — 성공 시 adapter 정보 객체 반환, 실패 시 구조화된 에러 메시지
#[wasm_bindgen]
pub async fn init_engine() -> Result<JsValue, JsValue> {
    let ctx = GpuContext::new().await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    let info = JsAdapterInfo {
        name: ctx.caps.info.name.clone(),
        backend: format!("{:?}", ctx.caps.info.backend),
        f16: ctx.caps.f16,
        timestamps: ctx.caps.timestamps,
    };
    ENGINE.with(|e| *e.borrow_mut() = Some(Rc::new(ctx)));
    serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// 공유 정확도 스위트 실행 — 네이티브 `cargo test`와 같은 케이스·같은 시드
#[wasm_bindgen]
pub async fn run_tests() -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let results =
        ai_gpu::testsuite::run_all(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
    let js: Vec<JsCase> = results
        .into_iter()
        .map(|r| JsCase { name: r.name, passed: r.passed, max_err: r.max_err, tol: r.tol })
        .collect();
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[derive(serde::Serialize)]
struct JsModelReport {
    name: String,
    ops: usize,
    unique_pipelines: usize,
    arena_mb: f64,
    weights_mb: f64,
}

#[derive(serde::Serialize)]
struct JsModelBench {
    ms_per_frame: f64,
    frames: u32,
    output_names: Vec<String>,
}

/// 프레임타임을 **CPU 인코딩**과 **GPU 대기**로 쪼갠 결과.
///
/// 우리는 프레임당 116 디스패치를 wasm↔JS 경계로 넘긴다(set_pipeline +
/// set_bind_group + dispatch = ~350 JS 호출). webgl2는 순수 JS 80 draw다.
/// "시간이 갈수록 우리만 느려진다"가 GPU 쪽인지 이 경계 쪽인지 합계로는 못 가른다.
#[derive(serde::Serialize)]
struct JsBenchSplit {
    /// 총 시간 / 프레임 (기존 bench_current와 같은 정의)
    ms_per_frame: f64,
    /// 제출까지 걸린 시간 / 프레임 = CPU 인코딩 + wasm↔JS 경계
    submit_ms: f64,
    /// 마지막 제출 후 GPU가 다 끝날 때까지 / 프레임
    wait_ms: f64,
}

/// .sw 모델 로드 (fetch한 바이트 전달)
#[wasm_bindgen]
pub async fn load_model(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let seg = ai_tasks::Segmenter::load(&ctx, &bytes)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let model = seg.model();
    let report = JsModelReport {
        name: model.sw.name.clone(),
        ops: model.report.ops,
        unique_pipelines: model.report.unique_pipelines,
        arena_mb: model.report.arena_bytes as f64 / 1e6,
        weights_mb: model.report.weights_bytes as f64 / 1e6,
    };
    MODEL.with(|m| *m.borrow_mut() = Some(seg));
    serde_wasm_bindgen::to_value(&report).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// 로드된 모델로 N프레임 추론 벤치 (시드 입력, 상태 ping-pong 포함)
#[wasm_bindgen]
pub async fn model_bench(frames: u32) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut seg = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;

    let result: Result<JsModelBench, JsValue> = async {
        // 첫 입력에 시드 데이터
        let t = &seg.model().sw.tensors[seg.model().sw.inputs[0] as usize];
        let elems = (t.h * t.w * t.c) as usize;
        let input = ai_core::rng::XorShift32::new(7).vec_f32(elems);
        seg.upload(&ctx, &input).map_err(|e| JsValue::from_str(&e.to_string()))?;

        for _ in 0..3 {
            seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;

        let t0 = web_time::Instant::now();
        for _ in 0..frames {
            seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / frames.max(1) as f64;

        let output_names: Vec<String> = seg
            .model()
            .sw
            .outputs
            .iter()
            .map(|&o| seg.model().sw.tensors[o as usize].name.clone())
            .collect();
        Ok(JsModelBench { ms_per_frame: ms, frames, output_names })
    }
    .await;

    MODEL.with(|m| *m.borrow_mut() = Some(seg));
    let bench = result?;
    serde_wasm_bindgen::to_value(&bench).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[derive(serde::Serialize)]
struct JsFrameInfo {
    h: u32,
    w: u32,
    c: u32,
    input: String,
    outputs: Vec<String>,
}

/// 프레임타임 분포 — 호스트의 **런타임 강등 판정 입력**.
///
/// 평균이 아니라 p90을 준다: 평균은 가끔 튀는 프레임을 감춘다.
/// 강등 기준은 "프레임 예산"이 아니라 "마스크 갱신율 하한"으로 잡아야 한다 —
/// 이 벽시계엔 GPU 큐 대기와 이벤트루프 대기가 섞이기 때문이다.
#[wasm_bindgen]
pub fn model_stats() -> Result<JsValue, JsValue> {
    MODEL.with(|m| {
        let b = m.borrow();
        let s = b.as_ref().ok_or_else(|| JsValue::from_str("모델 미로드"))?.stats();
        serde_wasm_bindgen::to_value(&JsStats {
            frames: s.frames,
            p50_ms: s.p50_ms,
            p90_ms: s.p90_ms,
            last_ms: s.last_ms,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

#[derive(serde::Serialize)]
struct JsStats {
    frames: u32,
    p50_ms: f32,
    p90_ms: f32,
    last_ms: f32,
}

/// 로드된 모델의 입력 크기·이름 (호스트가 프레임을 어떤 크기로 만들지 알아야 한다)
#[wasm_bindgen]
pub fn model_io() -> Result<JsValue, JsValue> {
    MODEL.with(|m| {
        let b = m.borrow();
        let model = b.as_ref().ok_or_else(|| JsValue::from_str("모델 미로드"))?.model();
        let t = &model.sw.tensors[model.sw.inputs[0] as usize];
        let info = JsFrameInfo {
            h: t.h,
            w: t.w,
            c: t.c,
            input: t.name.clone(),
            outputs: model
                .sw
                .outputs
                .iter()
                .map(|&o| model.sw.tensors[o as usize].name.clone())
                .collect(),
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// 프레임 1장 추론: 논리 NHWC(h*w*c, [0,1] RGB) 입력 → 지정 출력을 논리 NHWC로 반환.
/// 업로드·추론·리드백을 한 번에 도는 이유는 JS 왕복(=await 경계)을 최소화하기 위해서다.
/// 순환 상태는 엔진이 프레임 간 GPU 상주로 유지한다 — 호출자는 신경 쓸 필요 없다.
#[wasm_bindgen]
pub async fn infer_frame(rgb: Vec<f32>, output: String) -> Result<Vec<f32>, JsValue> {
    let ctx = engine()?;
    let mut seg = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;

    let result: Result<Vec<f32>, JsValue> = async {
        seg.upload(&ctx, &rgb).map_err(|e| JsValue::from_str(&e.to_string()))?;
        seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        seg.read_output(&ctx, &output).await.map_err(|e| JsValue::from_str(&e.to_string()))
    }
    .await;

    MODEL.with(|m| *m.borrow_mut() = Some(seg));
    result
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PRESENT: std::cell::RefCell<Option<present::Presenter>> =
        const { std::cell::RefCell::new(None) };
}

/// 출력 마스크를 그릴 캔버스를 붙인다 (WebGPU 서피스). 캔버스 크기는 모델 출력 크기로.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn attach_canvas(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    let ctx = engine()?;
    let p = present::Presenter::new(&ctx, canvas).map_err(|e| JsValue::from_str(&e))?;
    PRESENT.with(|c| *c.borrow_mut() = Some(p));
    Ok(())
}

/// 프레임 추론 후 지정 출력을 **리드백 없이** 캔버스에 바로 그린다.
/// 반환은 없다 — 마스크는 GPU에 머문 채 캔버스로 간다.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn infer_and_present(
    rgb: Vec<f32>,
    output: String,
    ch: u32,
    frame: web_sys::HtmlCanvasElement,
    mode: u32,
    bg: u32,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let mut seg = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;
    let result: Result<(), JsValue> = async {
        seg.upload(&ctx, &rgb).map_err(|e| JsValue::from_str(&e.to_string()))?;
        seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        let (buf, desc) = seg
            .output_storage(&output)
            .ok_or_else(|| JsValue::from_str(&format!("출력 아님: {output}")))?;
        PRESENT.with(|c| -> Result<(), JsValue> {
            let b = c.borrow();
            let p = b.as_ref().ok_or_else(|| JsValue::from_str("attach_canvas() 먼저"))?;
            // 카메라 프레임을 CPU 안 거치고 GPU 텍스처로 (copy_external_image_to_texture)
            let (fw, fh) = (frame.width(), frame.height());
            let srcinfo = ai_gpu::wgpu::wgt::CopyExternalImageSourceInfo {
                source: ai_gpu::wgpu::wgt::ExternalImageSource::HTMLCanvasElement(frame.clone()),
                origin: ai_gpu::wgpu::Origin2d::ZERO,
                flip_y: false,
            };
            p.upload_frame(&ctx, &srcinfo, fw, fh);
            p.draw(&ctx, buf, &desc, ai_tasks::CompositeOpts { channel: ch, mode, bg })
                .map_err(|e| JsValue::from_str(&e))
        })?;
        // 프레임 완료 대기 + 프레임타임 기록 (큐가 쌓이면 안 되는 이유는 finish_frame 참조)
        seg.finish_frame(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }
    .await;
    MODEL.with(|m| *m.borrow_mut() = Some(seg));
    result
}

/// 지금 올라가 있는 입력 그대로 N프레임 실행하고 **마지막에 한 번만** 동기화한다.
/// `model_bench`와 달리 입력을 시드 난수로 덮지 않아 라이브 데모 중에도 부를 수 있다.
/// 반환 = ms/frame. 상태 순환이 프레임 간 의존을 만들어 실제로 직렬 실행된다.
#[wasm_bindgen]
pub async fn bench_current(frames: u32) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut seg = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;
    let result: Result<JsValue, JsValue> = async {
        for _ in 0..2 {
            seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        let t0 = web_time::Instant::now();
        for _ in 0..frames {
            seg.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        // 제출만 끝난 시점 — 여기까지가 CPU(인코딩 + wasm↔JS 경계)
        let t_sub = t0.elapsed().as_secs_f64() * 1e3;
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        let t_all = t0.elapsed().as_secs_f64() * 1e3;
        let n = frames.max(1) as f64;
        serde_wasm_bindgen::to_value(&JsBenchSplit {
            ms_per_frame: t_all / n,
            submit_ms: t_sub / n,
            wait_ms: (t_all - t_sub) / n,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    .await;
    MODEL.with(|m| *m.borrow_mut() = Some(seg));
    result
}

/// 공유 벤치마크 실행 — 네이티브 ai-bench와 동일 루틴
#[wasm_bindgen]
pub async fn run_benchmarks() -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let results = ai_gpu::bench::run_benchmarks(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
    let js: Vec<JsBench> = results
        .into_iter()
        .map(|r| JsBench {
            name: r.name,
            gpu_min_us: r.gpu_min_ms.map(|v| v * 1e3),
            gpu_median_us: r.gpu_median_ms.map(|v| v * 1e3),
            wall_us: r.wall_ms * 1e3,
            gflops: r.gflops,
            pipeline_ms: r.pipeline_ms,
        })
        .collect();
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&e.to_string()))
}
