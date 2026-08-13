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
    static MODEL: RefCell<Option<ai_runtime::Model>> = const { RefCell::new(None) };
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

/// .sw 모델 로드 (fetch한 바이트 전달)
#[wasm_bindgen]
pub async fn load_model(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let model = ai_runtime::Model::load(&ctx, &bytes)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = JsModelReport {
        name: model.sw.name.clone(),
        ops: model.report.ops,
        unique_pipelines: model.report.unique_pipelines,
        arena_mb: model.report.arena_bytes as f64 / 1e6,
        weights_mb: model.report.weights_bytes as f64 / 1e6,
    };
    MODEL.with(|m| *m.borrow_mut() = Some(model));
    serde_wasm_bindgen::to_value(&report).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// 로드된 모델로 N프레임 추론 벤치 (시드 입력, 상태 ping-pong 포함)
#[wasm_bindgen]
pub async fn model_bench(frames: u32) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut model = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;

    let result: Result<JsModelBench, JsValue> = async {
        // 첫 입력에 시드 데이터
        let in_tid = model.sw.inputs[0];
        let t = &model.sw.tensors[in_tid as usize];
        let (name, elems) = (t.name.clone(), (t.h * t.w * t.c) as usize);
        let input = ai_core::rng::XorShift32::new(7).vec_f32(elems);
        model
            .upload_input(&ctx, &name, &input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        for _ in 0..3 {
            model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;

        let t0 = web_time::Instant::now();
        for _ in 0..frames {
            model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / frames.max(1) as f64;

        let output_names: Vec<String> = model
            .sw
            .outputs
            .iter()
            .map(|&o| model.sw.tensors[o as usize].name.clone())
            .collect();
        Ok(JsModelBench { ms_per_frame: ms, frames, output_names })
    }
    .await;

    MODEL.with(|m| *m.borrow_mut() = Some(model));
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

/// 로드된 모델의 입력 크기·이름 (호스트가 프레임을 어떤 크기로 만들지 알아야 한다)
#[wasm_bindgen]
pub fn model_io() -> Result<JsValue, JsValue> {
    MODEL.with(|m| {
        let b = m.borrow();
        let model = b.as_ref().ok_or_else(|| JsValue::from_str("모델 미로드"))?;
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
    let mut model = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;

    let result: Result<Vec<f32>, JsValue> = async {
        let in_name = model.sw.tensors[model.sw.inputs[0] as usize].name.clone();
        model
            .upload_input(&ctx, &in_name, &rgb)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        model.read_output(&ctx, &output).await.map_err(|e| JsValue::from_str(&e.to_string()))
    }
    .await;

    MODEL.with(|m| *m.borrow_mut() = Some(model));
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
    let mut model = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;
    let result: Result<(), JsValue> = async {
        let in_name = model.sw.tensors[model.sw.inputs[0] as usize].name.clone();
        model
            .upload_input(&ctx, &in_name, &rgb)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        let (buf, desc) = model
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
            p.upload_frame(&ctx, &srcinfo, fw, fh).map_err(|e| JsValue::from_str(&e))?;
            p.draw(&ctx, buf, &desc, ch, mode, bg).map_err(|e| JsValue::from_str(&e))
        })?;
        // ⚠ 제출만 하고 안 기다리면 큐가 무한정 쌓인다. 그러면 화면은 점점 과거
        // 프레임을 보여주고(마스크가 뒤처짐), 뒤에 도는 벤치의 wait_idle이 밀린
        // 큐 전체를 기다려 측정값이 계속 커진다. 사파리에서 실제로 그랬다
        // (추론 3.13 → 10.00ms 단조 증가). 프레임당 완료를 기다려 1프레임으로 묶는다.
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        Ok(())
    }
    .await;
    MODEL.with(|m| *m.borrow_mut() = Some(model));
    result
}

/// 지금 올라가 있는 입력 그대로 N프레임 실행하고 **마지막에 한 번만** 동기화한다.
/// `model_bench`와 달리 입력을 시드 난수로 덮지 않아 라이브 데모 중에도 부를 수 있다.
/// 반환 = ms/frame. 상태 순환이 프레임 간 의존을 만들어 실제로 직렬 실행된다.
#[wasm_bindgen]
pub async fn bench_current(frames: u32) -> Result<f64, JsValue> {
    let ctx = engine()?;
    let mut model = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("모델 미로드"))?;
    let result: Result<f64, JsValue> = async {
        for _ in 0..2 {
            model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        let t0 = web_time::Instant::now();
        for _ in 0..frames {
            model.infer(&ctx).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        ai_gpu::readback::wait_idle(&ctx).await.map_err(|e| JsValue::from_str(&e))?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / frames.max(1) as f64)
    }
    .await;
    MODEL.with(|m| *m.borrow_mut() = Some(model));
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
