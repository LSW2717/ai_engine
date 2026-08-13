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
#[cfg(target_arch = "wasm32")]
mod studio;

use wasm_bindgen::prelude::*;

thread_local! {
    static ENGINE: RefCell<Option<Rc<GpuContext>>> = const { RefCell::new(None) };
    static MODEL: RefCell<Option<ai_tasks::GpuSession>> = const { RefCell::new(None) };
    // CPU 폴백 티어 — GPU 엔진과 독립 (init_engine 실패 후에도 동작해야 한다)
    static CPU_MODEL: RefCell<Option<ai_tasks::CpuSession>> = const { RefCell::new(None) };
    // 핸들 기반 다중모델 풀 — vision 워커(det+lm+게이즈 상주)용.
    // 단일 슬롯(MODEL/CPU_MODEL)은 기존 데모·세그 경로 호환으로 남긴다.
    static GPU_MODELS: RefCell<ai_tasks::Pool<ai_tasks::GpuSession>> =
        RefCell::new(ai_tasks::Pool::new());
    static CPU_MODELS: RefCell<ai_tasks::Pool<ai_tasks::CpuSession>> =
        RefCell::new(ai_tasks::Pool::new());
    // 파이프라인 태스크 (상태: 트래킹 ROI + 필터) — 세션과 별도 수명
    static FACE_TASKS: RefCell<ai_tasks::Pool<ai_tasks::FaceTask>> =
        RefCell::new(ai_tasks::Pool::new());
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

/// 디바이스 유실 여부 — null(정상) 또는 사유 문자열("{reason}: {message}").
/// 호스트가 프레임 루프에서 폴링해 폴백 티어로 강등하는 근거.
/// (유실 후 infer 계열은 "디바이스 유실: ..." 에러로도 실패한다 — 이건 즉답 폴링용)
#[wasm_bindgen]
pub fn device_lost() -> Result<Option<String>, JsValue> {
    Ok(engine()?.lost_reason())
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
    let seg = ai_tasks::GpuSession::load(&ctx, &bytes)
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

/// CPU 폴백 모델 로드 — GPU 엔진 불필요 (init_engine 실패 시의 경로).
/// 어떤 모델(R11 등 경량)을 실을지는 호스트가 정한다 — 모델 조달은 호스트 몫.
#[wasm_bindgen]
pub fn load_model_cpu(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let seg = ai_tasks::CpuSession::load(&bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sw = seg.model().sw();
    let report = JsModelReport {
        name: sw.name.clone(),
        ops: sw.ops.len(),
        unique_pipelines: 0,
        arena_mb: 0.0,
        weights_mb: 0.0,
    };
    CPU_MODEL.with(|m| *m.borrow_mut() = Some(seg));
    serde_wasm_bindgen::to_value(&report).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// CPU 프레임 1장: 논리 NHWC 입력 → 추론 → 지정 출력 반환 (동기)
#[wasm_bindgen]
pub fn infer_frame_cpu(rgb: Vec<f32>, output: String) -> Result<Vec<f32>, JsValue> {
    CPU_MODEL.with(|m| {
        let mut b = m.borrow_mut();
        let seg = b.as_mut().ok_or_else(|| JsValue::from_str("CPU 모델 미로드"))?;
        seg.infer_frame(&rgb).map_err(|e| JsValue::from_str(&e.to_string()))?;
        seg.read_output(&output).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

thread_local! {
    /// infer_frame_cpu_view의 출력 스테이징 — 프레임마다 재사용
    static CPU_OUT: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// CPU 프레임 1장 — 출력을 **wasm 힙 뷰**로 반환 (JS로의 2차 복사 없음,
/// tflite HEAPF32 규약과 동일). 뷰는 다음 호출·wasm 메모리 성장 전까지만 유효 —
/// 받은 즉시 소비할 것.
#[wasm_bindgen]
pub fn infer_frame_cpu_view(rgb: &[f32], output: &str) -> Result<js_sys::Float32Array, JsValue> {
    CPU_MODEL.with(|m| {
        let mut b = m.borrow_mut();
        let seg = b.as_mut().ok_or_else(|| JsValue::from_str("CPU 모델 미로드"))?;
        seg.infer_frame(rgb).map_err(|e| JsValue::from_str(&e.to_string()))?;
        CPU_OUT.with(|o| {
            let mut o = o.borrow_mut();
            seg.read_output_into(output, &mut o)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            // SAFETY: 뷰 수명은 문서 규약(다음 호출 전 소비)이 지킨다
            Ok(unsafe { js_sys::Float32Array::view(&o) })
        })
    })
}

/// CPU 스텝별 반복 벤치 — [[라벨, ms], ...] (진단용, tools/profile_web.mjs --ops)
#[wasm_bindgen]
pub fn profile_cpu(reps: usize) -> Result<JsValue, JsValue> {
    CPU_MODEL.with(|m| {
        let mut b = m.borrow_mut();
        let seg = b.as_mut().ok_or_else(|| JsValue::from_str("CPU 모델 미로드"))?;
        serde_wasm_bindgen::to_value(&seg.bench_steps(reps))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// CPU 티어 프레임타임 분포 — GPU와 같은 강등 판정 입력 형식
#[wasm_bindgen]
pub fn model_stats_cpu() -> Result<JsValue, JsValue> {
    CPU_MODEL.with(|m| {
        let b = m.borrow();
        let s = b.as_ref().ok_or_else(|| JsValue::from_str("CPU 모델 미로드"))?.stats();
        serde_wasm_bindgen::to_value(&JsStats {
            frames: s.frames,
            p50_ms: s.p50_ms,
            p90_ms: s.p90_ms,
            last_ms: s.last_ms,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// CPU 모델의 입력 크기·이름 (model_io의 CPU 짝)
#[wasm_bindgen]
pub fn model_io_cpu() -> Result<JsValue, JsValue> {
    CPU_MODEL.with(|m| {
        let b = m.borrow();
        let sw = b.as_ref().ok_or_else(|| JsValue::from_str("CPU 모델 미로드"))?.model().sw();
        let t = &sw.tensors[sw.inputs[0] as usize];
        let info = JsFrameInfo {
            h: t.h,
            w: t.w,
            c: t.c,
            input: t.name.clone(),
            outputs: sw
                .outputs
                .iter()
                .map(|&o| sw.tensors[o as usize].name.clone())
                .collect(),
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
    })
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

// ───────── 핸들 기반 다중모델 API (vision 워커: det+lm+게이즈 상주) ─────────
//
// 단일 슬롯 API와 달리 로드가 기존 모델을 밀어내지 않는다. 핸들은 재사용되지
// 않으며(ai_tasks::Pool), 언로드된 핸들 접근은 구조화된 에러로 실패한다.

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[derive(serde::Serialize)]
struct JsLoaded {
    handle: u32,
    name: String,
    ops: usize,
    unique_pipelines: usize,
    arena_mb: f64,
    weights_mb: f64,
}

#[derive(serde::Serialize)]
struct JsDetection {
    score: f32,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    keypoints: Vec<[f32; 2]>,
}

fn js_detections(dets: Vec<ai_tasks::Detection>) -> Result<JsValue, JsValue> {
    let js: Vec<JsDetection> = dets
        .into_iter()
        .map(|d| JsDetection {
            score: d.score,
            xmin: d.xmin,
            ymin: d.ymin,
            xmax: d.xmax,
            ymax: d.ymax,
            keypoints: d.keypoints,
        })
        .collect();
    serde_wasm_bindgen::to_value(&js).map_err(js_err)
}

/// GPU 모델 로드 → 핸들 반환 (기존 모델은 유지 — 다중 상주)
#[wasm_bindgen]
pub async fn load_model_h(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let seg = ai_tasks::GpuSession::load(&ctx, &bytes).await.map_err(js_err)?;
    let model = seg.model();
    let mut report = JsLoaded {
        handle: 0,
        name: model.sw.name.clone(),
        ops: model.report.ops,
        unique_pipelines: model.report.unique_pipelines,
        arena_mb: model.report.arena_bytes as f64 / 1e6,
        weights_mb: model.report.weights_bytes as f64 / 1e6,
    };
    report.handle = GPU_MODELS.with(|p| p.borrow_mut().insert(seg));
    serde_wasm_bindgen::to_value(&report).map_err(js_err)
}

/// GPU 모델 언로드 — 버퍼·파이프라인 참조 해제
#[wasm_bindgen]
pub fn unload_model_h(handle: u32) -> Result<(), JsValue> {
    GPU_MODELS.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

/// 핸들 모델의 입력 크기·이름
#[wasm_bindgen]
pub fn model_io_h(handle: u32) -> Result<JsValue, JsValue> {
    GPU_MODELS.with(|p| {
        let b = p.borrow();
        let model = b.get(handle).map_err(js_err)?.model();
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
        serde_wasm_bindgen::to_value(&info).map_err(js_err)
    })
}

/// 핸들 모델의 프레임타임 분포 (강등 판정 입력 — 모델별로 따로 쌓인다)
#[wasm_bindgen]
pub fn model_stats_h(handle: u32) -> Result<JsValue, JsValue> {
    GPU_MODELS.with(|p| {
        let b = p.borrow();
        let s = b.get(handle).map_err(js_err)?.stats();
        serde_wasm_bindgen::to_value(&JsStats {
            frames: s.frames,
            p50_ms: s.p50_ms,
            p90_ms: s.p90_ms,
            last_ms: s.last_ms,
        })
        .map_err(js_err)
    })
}

/// 핸들 모델로 프레임 1장 (업로드→추론→리드백) — infer_frame의 핸들판
#[wasm_bindgen]
pub async fn infer_frame_h(
    handle: u32,
    rgb: Vec<f32>,
    output: String,
) -> Result<Vec<f32>, JsValue> {
    let ctx = engine()?;
    // RefCell 대여를 await 너머로 못 들고 간다 — 꺼냈다 되돌리는 규약 (Pool 참조)
    let mut seg = GPU_MODELS.with(|p| p.borrow_mut().take(handle)).map_err(js_err)?;
    let result: Result<Vec<f32>, JsValue> = async {
        seg.upload(&ctx, &rgb).map_err(js_err)?;
        seg.infer(&ctx).await.map_err(js_err)?;
        let out = seg.read_output(&ctx, &output).await.map_err(js_err)?;
        seg.finish_frame(&ctx).await.map_err(js_err)?;
        Ok(out)
    }
    .await;
    GPU_MODELS.with(|p| p.borrow_mut().put(handle, seg));
    result
}

/// 디텍터 프레임 1장 (GPU): 레터박스된 입력 → 검출 목록 (원본 프레임 정규화 좌표).
/// preset: "face"(BlazeFace short-range 128²) | "palm"(192²).
/// 호스트는 검정 캔버스에 비율 유지 중앙 정렬로 프레임을 그려 [-1,1]로 넘긴다.
#[wasm_bindgen]
pub async fn detect_gpu(
    handle: u32,
    preset: String,
    rgb: Vec<f32>,
    src_w: u32,
    src_h: u32,
) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let post = ai_tasks::detect::preset(&preset).map_err(js_err)?;
    let mut seg = GPU_MODELS.with(|p| p.borrow_mut().take(handle)).map_err(js_err)?;
    let result = seg.detect(&ctx, post, &rgb, src_w, src_h).await;
    GPU_MODELS.with(|p| p.borrow_mut().put(handle, seg));
    js_detections(result.map_err(js_err)?)
}

/// CPU 모델 로드 → 핸들 반환 (load_model_cpu의 다중 상주판)
#[wasm_bindgen]
pub fn load_model_cpu_h(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let seg = ai_tasks::CpuSession::load(&bytes).map_err(js_err)?;
    let sw = seg.model().sw();
    let mut report = JsLoaded {
        handle: 0,
        name: sw.name.clone(),
        ops: sw.ops.len(),
        unique_pipelines: 0,
        arena_mb: 0.0,
        weights_mb: 0.0,
    };
    report.handle = CPU_MODELS.with(|p| p.borrow_mut().insert(seg));
    serde_wasm_bindgen::to_value(&report).map_err(js_err)
}

#[wasm_bindgen]
pub fn unload_model_cpu_h(handle: u32) -> Result<(), JsValue> {
    CPU_MODELS.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

#[wasm_bindgen]
pub fn model_io_cpu_h(handle: u32) -> Result<JsValue, JsValue> {
    CPU_MODELS.with(|p| {
        let b = p.borrow();
        let sw = b.get(handle).map_err(js_err)?.model().sw();
        let t = &sw.tensors[sw.inputs[0] as usize];
        let info = JsFrameInfo {
            h: t.h,
            w: t.w,
            c: t.c,
            input: t.name.clone(),
            outputs: sw
                .outputs
                .iter()
                .map(|&o| sw.tensors[o as usize].name.clone())
                .collect(),
        };
        serde_wasm_bindgen::to_value(&info).map_err(js_err)
    })
}

#[wasm_bindgen]
pub fn model_stats_cpu_h(handle: u32) -> Result<JsValue, JsValue> {
    CPU_MODELS.with(|p| {
        let b = p.borrow();
        let s = b.get(handle).map_err(js_err)?.stats();
        serde_wasm_bindgen::to_value(&JsStats {
            frames: s.frames,
            p50_ms: s.p50_ms,
            p90_ms: s.p90_ms,
            last_ms: s.last_ms,
        })
        .map_err(js_err)
    })
}

/// 핸들 모델로 CPU 프레임 1장 (동기)
#[wasm_bindgen]
pub fn infer_frame_cpu_h(
    handle: u32,
    rgb: &[f32],
    output: &str,
) -> Result<Vec<f32>, JsValue> {
    CPU_MODELS.with(|p| {
        let mut b = p.borrow_mut();
        let seg = b.get_mut(handle).map_err(js_err)?;
        seg.infer_frame(rgb).map_err(js_err)?;
        seg.read_output(output).map_err(js_err)
    })
}

/// 디텍터 프레임 1장 (CPU) — detect_gpu와 같은 계약, 동기
#[wasm_bindgen]
pub fn detect_cpu(
    handle: u32,
    preset: String,
    rgb: &[f32],
    src_w: u32,
    src_h: u32,
) -> Result<JsValue, JsValue> {
    let post = ai_tasks::detect::preset(&preset).map_err(js_err)?;
    let dets = CPU_MODELS.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(handle).map_err(js_err)?.detect(post, rgb, src_w, src_h).map_err(js_err)
    })?;
    js_detections(dets)
}

// ───────── FaceTask (검출→ROI 트래킹→랜드마크 파이프라인) ─────────

#[derive(serde::Serialize)]
struct JsRoi {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    rotation: f32,
}

#[derive(serde::Serialize)]
struct JsFaceResult {
    presence: f32,
    /// 원본 프레임 정규화 [x,y,z] × 478
    points: Vec<[f32; 3]>,
    roi: JsRoi,
}

fn js_face_result(r: Option<ai_tasks::FaceResult>) -> Result<JsValue, JsValue> {
    match r {
        None => Ok(JsValue::NULL),
        Some(r) => serde_wasm_bindgen::to_value(&JsFaceResult {
            presence: r.presence,
            points: r.points,
            roi: JsRoi {
                cx: r.roi.cx,
                cy: r.roi.cy,
                w: r.roi.w,
                h: r.roi.h,
                rotation: r.roi.rotation,
            },
        })
        .map_err(js_err),
    }
}

/// FaceTask 생성 → 핸들. smoothing: OneEuroFilter 적용 (파라미터 검증 전 — 기본 false 권장)
#[wasm_bindgen]
pub fn face_task_new(smoothing: bool) -> u32 {
    FACE_TASKS.with(|p| p.borrow_mut().insert(ai_tasks::FaceTask::new(smoothing)))
}

#[wasm_bindgen]
pub fn face_task_free(handle: u32) -> Result<(), JsValue> {
    FACE_TASKS.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

/// 트래킹 상태 폐기 — 다음 프레임은 검출부터 (탭 전환·시킹 등 불연속 지점에서)
#[wasm_bindgen]
pub fn face_task_reset(handle: u32) -> Result<(), JsValue> {
    FACE_TASKS.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(handle).map_err(js_err)?.reset();
        Ok(())
    })
}

/// CPU 한 프레임: u8 RGB 프레임 → null | {presence, points[478][3], roi}.
/// det/lm은 CPU 세션 핸들. t_ms는 단조 증가 타임스탬프 (performance.now()).
#[wasm_bindgen]
pub fn face_task_cpu(
    task: u32,
    det: u32,
    lm: u32,
    frame: &[u8],
    img_w: u32,
    img_h: u32,
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    // 세션 둘을 풀에서 꺼내 태스크에 주입 (같은 핸들이면 두 번째 take가 실패한다)
    let mut det_s = CPU_MODELS.with(|p| p.borrow_mut().take(det)).map_err(js_err)?;
    let lm_r = CPU_MODELS.with(|p| p.borrow_mut().take(lm));
    let mut lm_s = match lm_r {
        Ok(s) => s,
        Err(e) => {
            CPU_MODELS.with(|p| p.borrow_mut().put(det, det_s));
            return Err(js_err(e));
        }
    };
    let result = FACE_TASKS.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(task)
            .map_err(js_err)?
            .process_cpu(&mut det_s, &mut lm_s, frame, img_w, img_h, t_ms)
            .map_err(js_err)
    });
    CPU_MODELS.with(|p| {
        let mut b = p.borrow_mut();
        b.put(det, det_s);
        b.put(lm, lm_s);
    });
    js_face_result(result?)
}

/// GPU 한 프레임 — face_task_cpu와 같은 계약, det/lm은 GPU 세션 핸들
#[wasm_bindgen]
pub async fn face_task_gpu(
    task: u32,
    det: u32,
    lm: u32,
    frame: Vec<u8>,
    img_w: u32,
    img_h: u32,
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut task_s = FACE_TASKS.with(|p| p.borrow_mut().take(task)).map_err(js_err)?;
    let mut det_s = match GPU_MODELS.with(|p| p.borrow_mut().take(det)) {
        Ok(s) => s,
        Err(e) => {
            FACE_TASKS.with(|p| p.borrow_mut().put(task, task_s));
            return Err(js_err(e));
        }
    };
    let mut lm_s = match GPU_MODELS.with(|p| p.borrow_mut().take(lm)) {
        Ok(s) => s,
        Err(e) => {
            FACE_TASKS.with(|p| p.borrow_mut().put(task, task_s));
            GPU_MODELS.with(|p| p.borrow_mut().put(det, det_s));
            return Err(js_err(e));
        }
    };
    let result = task_s
        .process_gpu(&ctx, &mut det_s, &mut lm_s, &frame, img_w, img_h, t_ms)
        .await;
    FACE_TASKS.with(|p| p.borrow_mut().put(task, task_s));
    GPU_MODELS.with(|p| {
        let mut b = p.borrow_mut();
        b.put(det, det_s);
        b.put(lm, lm_s);
    });
    js_face_result(result.map_err(js_err)?)
}

// ───────── studio — VideoPipeline 데모/게이트 (web/demo/studio.html) ─────────

#[cfg(target_arch = "wasm32")]
thread_local! {
    static STUDIO: RefCell<Option<studio::Studio>> = const { RefCell::new(None) };
}

/// VideoPipeline을 출력 캔버스(WebGPU 서피스)에 연결
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_attach(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    let ctx = engine()?;
    let s = studio::Studio::new(&ctx, canvas).map_err(js_err)?;
    STUDIO.with(|c| *c.borrow_mut() = Some(s));
    Ok(())
}

/// EffectsPatch JSON 적용 (없음=유지 / null=해제 / 값=설정)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_config(json: String) -> Result<(), JsValue> {
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        s.pipeline.apply_json(&json).map_err(js_err)
    })
}

/// 배경 이미지 업로드 (RGBA8)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_bg_image(rgba: &[u8], w: u32, h: u32) -> Result<(), JsValue> {
    let ctx = engine()?;
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        s.pipeline.set_background_image(&ctx, rgba, w, h);
        Ok(())
    })
}

/// 세그 세션 교체 후 필수 — 파이프라인 바인드그룹이 이전 모델 버퍼를 물고 있다
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_invalidate() -> Result<(), JsValue> {
    STUDIO.with(|c| {
        if let Some(s) = c.borrow_mut().as_mut() {
            s.pipeline.invalidate();
        }
        Ok(())
    })
}

/// 프레임 1장: 소스 캔버스 → GPU 전처리 → 세그 추론 → 마스크 스택 → 캔버스.
/// seg = GPU 세션 핸들. CPU 픽셀 왕복 0 — JS는 캔버스만 넘긴다.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn studio_frame(
    seg: u32,
    source: web_sys::HtmlCanvasElement,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let mut seg_s = GPU_MODELS.with(|p| p.borrow_mut().take(seg)).map_err(js_err)?;
    let mut st = STUDIO.with(|c| c.borrow_mut().take());
    let result = match st.as_mut() {
        Some(s) => s.frame(&ctx, &mut seg_s, &source).await.map_err(js_err),
        None => Err(JsValue::from_str("studio_attach 먼저")),
    };
    STUDIO.with(|c| *c.borrow_mut() = st);
    GPU_MODELS.with(|p| p.borrow_mut().put(seg, seg_s));
    result
}

// ───────── vb 픽셀 diff 게이트 — v-ai GLSL 파리티 (web/demo/vb-diff.html) ─────────

#[cfg(target_arch = "wasm32")]
thread_local! {
    static VB_GATE: RefCell<Option<ai_tasks::features::vb::GateHarness>> =
        const { RefCell::new(None) };
}

/// 게이트 하네스 생성/전체 리셋 (EffectsState·EMA·배경 이미지 전부 초기화)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn vb_gate_reset() -> Result<(), JsValue> {
    let ctx = engine()?;
    VB_GATE.with(|c| {
        *c.borrow_mut() = Some(ai_tasks::features::vb::GateHarness::new(&ctx));
    });
    Ok(())
}

/// EffectsPatch JSON 적용 (게이트 파이프라인)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn vb_gate_config(json: String) -> Result<(), JsValue> {
    VB_GATE.with(|c| {
        let mut b = c.borrow_mut();
        let g = b.as_mut().ok_or_else(|| JsValue::from_str("vb_gate_reset 먼저"))?;
        g.pipeline.apply_json(&json).map_err(js_err)
    })
}

/// 배경 이미지 업로드 (게이트 파이프라인, RGBA8)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn vb_gate_bg_image(rgba: &[u8], w: u32, h: u32) -> Result<(), JsValue> {
    let ctx = engine()?;
    VB_GATE.with(|c| {
        let mut b = c.borrow_mut();
        let g = b.as_mut().ok_or_else(|| JsValue::from_str("vb_gate_reset 먼저"))?;
        g.pipeline.set_background_image(&ctx, rgba, w, h);
        Ok(())
    })
}

/// 게이트: 프레이밍 크롭 강제 고정 (bbox·스무딩 우회 — 크롭 수학 파리티 검증용)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn vb_gate_framing(scale: f32, cx: f32, cy: f32) -> Result<(), JsValue> {
    VB_GATE.with(|c| {
        let mut b = c.borrow_mut();
        let g = b.as_mut().ok_or_else(|| JsValue::from_str("vb_gate_reset 먼저"))?;
        g.pipeline.set_framing_override(Some((scale, cx, cy)));
        Ok(())
    })
}

/// 프레임+마스크 주입 1장 → 이펙트 스택(추론 없음) → 최종 RGBA 반환.
/// ch=1 알파(RVM pha 등가), ch=2 로짓 [bg, person]. 마스크는 모델 마스크 해상도.
/// ema=false면 시간 상태를 끊는다 (공간 스택 결정적 게이트).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn vb_gate_frame(
    seg: u32,
    frame_rgba: &[u8],
    fw: u32,
    fh: u32,
    mask: &[f32],
    ch: u32,
    ema: bool,
) -> Result<Vec<u8>, JsValue> {
    let ctx = engine()?;
    let seg_s = GPU_MODELS.with(|p| p.borrow_mut().take(seg)).map_err(js_err)?;
    let mut g = VB_GATE.with(|c| c.borrow_mut().take());
    let result = match g.as_mut() {
        Some(g) => g.frame(&ctx, &seg_s, frame_rgba, fw, fh, mask, ch, ema).await.map_err(js_err),
        None => Err(JsValue::from_str("vb_gate_reset 먼저")),
    };
    VB_GATE.with(|c| *c.borrow_mut() = g);
    GPU_MODELS.with(|p| p.borrow_mut().put(seg, seg_s));
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
