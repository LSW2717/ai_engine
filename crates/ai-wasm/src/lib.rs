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
    // 집중도 태스크 (상태: OneEuro 필터 + 상태머신 + baseline) — FaceTask 랜드마크 소비
    static GAZE_TASKS: RefCell<ai_tasks::Pool<ai_tasks::GazeTask>> =
        RefCell::new(ai_tasks::Pool::new());
    // 손 태스크 (상태: 2손 트래킹 ROI) + 제스처 판정기 (상태: 쿨다운·홀드 카운터)
    static HAND_TASKS: RefCell<ai_tasks::Pool<ai_tasks::HandTask>> =
        RefCell::new(ai_tasks::Pool::new());
    static GESTURES: RefCell<ai_tasks::Pool<ai_tasks::features::hand::gesture::GestureClassifier>> =
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

// ───────── GazeTask (집중도 — FaceTask 랜드마크 소비, CNN 페이싱 내장) ─────────

/// FocusStatus → 웹 focus-tracker 문자열 (types.ts FocusStatus 1:1)
fn focus_status_str(s: ai_tasks::FocusStatus) -> &'static str {
    use ai_tasks::FocusStatus::*;
    match s {
        Initializing => "INITIALIZING",
        Focused => "FOCUSED",
        OtherMonitor => "OTHER_MONITOR",
        LookingAway => "LOOKING_AWAY",
        EyesClosed => "EYES_CLOSED",
        NoFace => "NO_FACE",
        MultipleFaces => "MULTIPLE_FACES",
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JsFocusResult {
    status: &'static str,
    attentive: bool,
    score: u32,
    monitor_index: i32,
    /// 필터 후 각도 (도) — 얼굴 소실 틱은 null
    yaw: Option<f32>,
    pitch: Option<f32>,
    /// 마지막 CNN 원시 각도 (필터 전) — 게이트/진단용
    raw_yaw: Option<f32>,
    raw_pitch: Option<f32>,
}

/// GazeTask 생성 → 핸들 (baseline 자동 수집 on)
#[wasm_bindgen]
pub fn gaze_task_new() -> u32 {
    GAZE_TASKS.with(|p| p.borrow_mut().insert(ai_tasks::GazeTask::default()))
}

#[wasm_bindgen]
pub fn gaze_task_free(handle: u32) -> Result<(), JsValue> {
    GAZE_TASKS.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

/// 필터·상태머신·baseline 전부 초기화 (스트림 파기 시)
#[wasm_bindgen]
pub fn gaze_task_reset(handle: u32) -> Result<(), JsValue> {
    GAZE_TASKS.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(handle).map_err(js_err)?.reset();
        Ok(())
    })
}

/// 비전 틱 1회 (GPU): u8 RGB 프레임 + FaceTask 랜드마크(정규화 flat [x,y]×N,
/// **비어 있으면 얼굴 소실 틱**) → {status, attentive, score, monitorIndex,
/// yaw, pitch, rawYaw, rawPitch}. gaze = GPU 세션 핸들(gaze.sw).
/// CNN은 내부 페이싱(83.3ms 하한)으로만 돈다 — 틱은 ~10fps 권장 (웹 visionFps).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn gaze_task_gpu(
    task: u32,
    gaze: u32,
    frame: Vec<u8>,
    img_w: u32,
    img_h: u32,
    landmarks: Vec<f32>,
    face_count: u32,
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut task_s = GAZE_TASKS.with(|p| p.borrow_mut().take(task)).map_err(js_err)?;
    let mut gaze_s = match GPU_MODELS.with(|p| p.borrow_mut().take(gaze)) {
        Ok(s) => s,
        Err(e) => {
            GAZE_TASKS.with(|p| p.borrow_mut().put(task, task_s));
            return Err(js_err(e));
        }
    };
    let pts: Vec<[f32; 2]> = landmarks.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let lm = if pts.is_empty() { None } else { Some(pts.as_slice()) };
    let result = task_s
        .process_gpu(
            &ctx,
            &mut gaze_s,
            &frame,
            img_w as usize,
            img_h as usize,
            lm,
            face_count as usize,
            t_ms,
        )
        .await;
    let filtered = task_s.last_filtered;
    let raw = task_s.last_cnn();
    GAZE_TASKS.with(|p| p.borrow_mut().put(task, task_s));
    GPU_MODELS.with(|p| p.borrow_mut().put(gaze, gaze_s));
    let r = result.map_err(js_err)?;
    serde_wasm_bindgen::to_value(&JsFocusResult {
        status: focus_status_str(r.status),
        attentive: r.attentive,
        score: r.score,
        monitor_index: r.monitor_index,
        yaw: filtered.map(|g| g.yaw),
        pitch: filtered.map(|g| g.pitch),
        raw_yaw: raw.map(|g| g.yaw),
        raw_pitch: raw.map(|g| g.pitch),
    })
    .map_err(js_err)
}

// 게이트 헬퍼 — features::gaze::preprocess 순수 함수의 1:1 노출 (gaze-ab.html)

/// 랜드마크(정규화 flat [x,y]×N) → 크롭 박스 [x0,y0,x1,y1] (없으면 null)
#[wasm_bindgen]
pub fn gaze_crop_box(landmarks: Vec<f32>, vw: f32, vh: f32) -> Option<Vec<f32>> {
    let pts: Vec<[f32; 2]> = landmarks.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    ai_tasks::features::gaze::preprocess::face_crop_box(&pts, vw, vh).map(|b| b.to_vec())
}

/// u8 RGB 프레임 + 크롭 박스 → 448² RGB f32 [0,1] 인터리브 (ImageNet 정규화 **전**)
#[wasm_bindgen]
pub fn gaze_crop_pixels(
    frame: &[u8],
    w: u32,
    h: u32,
    bx: &[f32],
) -> Result<Vec<f32>, JsValue> {
    if bx.len() != 4 {
        return Err(JsValue::from_str("크롭 박스는 [x0,y0,x1,y1]"));
    }
    let mut out = vec![0f32; 448 * 448 * 3];
    ai_tasks::features::gaze::preprocess::crop_resize_rgb(
        frame,
        w as usize,
        h as usize,
        [bx[0], bx[1], bx[2], bx[3]],
        448,
        &mut out,
    );
    Ok(out)
}

/// ImageNet 정규화 (RGB [0,1] 인터리브 → 모델 입력)
#[wasm_bindgen]
pub fn gaze_normalize(mut buf: Vec<f32>) -> Vec<f32> {
    ai_tasks::features::gaze::preprocess::imagenet_normalize(&mut buf);
    buf
}

/// 90bin 로짓 → 각도 (도): softmax 기댓값 ×4 − 180
#[wasm_bindgen]
pub fn gaze_decode_bins(logits: &[f32]) -> f32 {
    ai_tasks::features::gaze::preprocess::decode_bins(logits)
}

// ───────── HandTask (팜 det→ROI→lm, 2손 트래킹) + 제스처 ─────────

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JsHandResult {
    /// 손 존재 확률 [0,1] (모델 sigmoid 내장 — 로짓 아님)
    presence: f32,
    /// P(Left) [0,1] — >0.5면 Left
    handedness: f32,
    /// 원본 프레임 정규화 [x,y,z] × 21
    points: Vec<[f32; 3]>,
    /// 월드 랜드마크 (미터) × 21
    world: Vec<[f32; 3]>,
    roi: JsRoi,
}

fn js_hand_results(rs: Vec<ai_tasks::HandResult>) -> Result<JsValue, JsValue> {
    let out: Vec<JsHandResult> = rs
        .into_iter()
        .map(|r| JsHandResult {
            presence: r.presence,
            handedness: r.handedness,
            points: r.points,
            world: r.world,
            roi: JsRoi {
                cx: r.roi.cx,
                cy: r.roi.cy,
                w: r.roi.w,
                h: r.roi.h,
                rotation: r.roi.rotation,
            },
        })
        .collect();
    serde_wasm_bindgen::to_value(&out).map_err(js_err)
}

/// HandTask 생성 → 핸들 (num_hands: 1~2 — clap은 2 필요)
#[wasm_bindgen]
pub fn hand_task_new(num_hands: u32) -> u32 {
    HAND_TASKS.with(|p| p.borrow_mut().insert(ai_tasks::HandTask::new(num_hands as usize)))
}

#[wasm_bindgen]
pub fn hand_task_free(handle: u32) -> Result<(), JsValue> {
    HAND_TASKS.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

/// 트래킹 상태 폐기 — 다음 프레임은 검출부터
#[wasm_bindgen]
pub fn hand_task_reset(handle: u32) -> Result<(), JsValue> {
    HAND_TASKS.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(handle).map_err(js_err)?.reset();
        Ok(())
    })
}

/// CPU 한 프레임: u8 RGB 프레임 → [{presence, handedness, points, world, roi}]
/// (0~num_hands개). det/lm은 CPU 세션 핸들.
#[wasm_bindgen]
pub fn hand_task_cpu(
    task: u32,
    det: u32,
    lm: u32,
    frame: &[u8],
    img_w: u32,
    img_h: u32,
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    let mut det_s = CPU_MODELS.with(|p| p.borrow_mut().take(det)).map_err(js_err)?;
    let lm_r = CPU_MODELS.with(|p| p.borrow_mut().take(lm));
    let mut lm_s = match lm_r {
        Ok(s) => s,
        Err(e) => {
            CPU_MODELS.with(|p| p.borrow_mut().put(det, det_s));
            return Err(js_err(e));
        }
    };
    let result = HAND_TASKS.with(|p| {
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
    js_hand_results(result?)
}

/// GPU 한 프레임 — hand_task_cpu와 같은 계약, det/lm은 GPU 세션 핸들
#[wasm_bindgen]
pub async fn hand_task_gpu(
    task: u32,
    det: u32,
    lm: u32,
    frame: Vec<u8>,
    img_w: u32,
    img_h: u32,
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    let ctx = engine()?;
    let mut task_s = HAND_TASKS.with(|p| p.borrow_mut().take(task)).map_err(js_err)?;
    let mut det_s = match GPU_MODELS.with(|p| p.borrow_mut().take(det)) {
        Ok(s) => s,
        Err(e) => {
            HAND_TASKS.with(|p| p.borrow_mut().put(task, task_s));
            return Err(js_err(e));
        }
    };
    let mut lm_s = match GPU_MODELS.with(|p| p.borrow_mut().take(lm)) {
        Ok(s) => s,
        Err(e) => {
            HAND_TASKS.with(|p| p.borrow_mut().put(task, task_s));
            GPU_MODELS.with(|p| p.borrow_mut().put(det, det_s));
            return Err(js_err(e));
        }
    };
    let result = task_s
        .process_gpu(&ctx, &mut det_s, &mut lm_s, &frame, img_w, img_h, t_ms)
        .await;
    HAND_TASKS.with(|p| p.borrow_mut().put(task, task_s));
    GPU_MODELS.with(|p| {
        let mut b = p.borrow_mut();
        b.put(det, det_s);
        b.put(lm, lm_s);
    });
    js_hand_results(result.map_err(js_err)?)
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JsGestureEvent {
    gesture: &'static str,
    confidence: f32,
    handedness: &'static str,
    ts_ms: f64,
}

/// 제스처 판정기 생성 → 핸들 (clap 융합 브리지 + thumbsUp/handRaise)
#[wasm_bindgen]
pub fn gesture_new() -> u32 {
    GESTURES.with(|p| {
        p.borrow_mut().insert(ai_tasks::features::hand::gesture::GestureClassifier::default())
    })
}

#[wasm_bindgen]
pub fn gesture_free(handle: u32) -> Result<(), JsValue> {
    GESTURES.with(|p| p.borrow_mut().remove(handle)).map_err(js_err)?;
    Ok(())
}

#[wasm_bindgen]
pub fn gesture_reset(handle: u32) -> Result<(), JsValue> {
    GESTURES.with(|p| {
        let mut b = p.borrow_mut();
        b.get_mut(handle).map_err(js_err)?.reset();
        Ok(())
    })
}

/// 제스처 판정 1틱: HandTask 결과를 flat으로 —
/// hands_flat = [x,y]×21 × N (정규화), handed = P(Left) × N.
/// 반환: [{gesture: "clap"|"thumbsUp"|"handRaise", confidence, handedness, tsMs}]
#[wasm_bindgen]
pub fn gesture_classify(
    handle: u32,
    hands_flat: &[f32],
    handed: &[f32],
    t_ms: f64,
) -> Result<JsValue, JsValue> {
    use ai_tasks::features::hand::gesture::{Gesture, HandSnapshot, Handedness};
    if hands_flat.len() != handed.len() * 42 {
        return Err(JsValue::from_str("hands_flat은 손당 [x,y]×21 = 42개"));
    }
    let hands: Vec<HandSnapshot> = hands_flat
        .chunks_exact(42)
        .zip(handed)
        .map(|(c, &p)| {
            let mut landmarks = [[0f32; 2]; 21];
            for (i, xy) in c.chunks_exact(2).enumerate() {
                landmarks[i] = [xy[0], xy[1]];
            }
            let handedness =
                if p > 0.5 { Handedness::Left } else { Handedness::Right };
            HandSnapshot { landmarks, handedness }
        })
        .collect();
    let events = GESTURES.with(|p| {
        let mut b = p.borrow_mut();
        Ok::<_, JsValue>(b.get_mut(handle).map_err(js_err)?.classify(&hands, t_ms))
    })?;
    let out: Vec<JsGestureEvent> = events
        .into_iter()
        .map(|e| JsGestureEvent {
            gesture: match e.gesture {
                Gesture::ThumbsUp => "thumbsUp",
                Gesture::HandRaise => "handRaise",
                Gesture::Clap => "clap",
            },
            confidence: e.confidence,
            handedness: match e.handedness {
                ai_tasks::features::hand::gesture::Handedness::Left => "Left",
                ai_tasks::features::hand::gesture::Handedness::Right => "Right",
                ai_tasks::features::hand::gesture::Handedness::Unknown => "Unknown",
            },
            ts_ms: e.ts_ms,
        })
        .collect();
    serde_wasm_bindgen::to_value(&out).map_err(js_err)
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

// ── studio 3D 아이템 오버레이 (wgpu items3d — three.js 대체) ──

/// 현재 아이템 선택 (각각 종류명 또는 "none"). 첫 호출 시 오버레이 생성.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_items(hat: String, eyewear: String, beard: String) -> Result<(), JsValue> {
    let ctx = engine()?;
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        if s.items.is_none() {
            s.items = Some(
                ai_tasks::features::face::items3d::ItemsOverlay::new(&ctx).map_err(js_err)?,
            );
        }
        s.items.as_mut().unwrap().set_items(&hat, &eyewear, &beard);
        Ok(())
    })
}

/// GLB bytes 주입 (종류당 1회 — 호스트가 fetch). 오버레이가 없으면 생성.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_item_glb(kind: String, bytes: Vec<u8>) -> Result<(), JsValue> {
    let ctx = engine()?;
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        if s.items.is_none() {
            s.items = Some(
                ai_tasks::features::face::items3d::ItemsOverlay::new(&ctx).map_err(js_err)?,
            );
        }
        s.items.as_mut().unwrap().preload_glb(&ctx, &kind, &bytes).map_err(js_err)
    })
}

/// 최신 얼굴 포즈 — FaceTask points flat [x,y,z]×478 (정규화).
/// 빈 배열 = 얼굴 소실 (스무딩 리셋).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_items_pose(points: Vec<f32>) -> Result<(), JsValue> {
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        if let Some(items) = s.items.as_mut() {
            let pts: Vec<[f32; 3]> =
                points.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
            items.set_pose(if pts.is_empty() { None } else { Some(pts) });
        }
        Ok(())
    })
}

/// 씬 광원 프로브 — u8 RGB 프레임 (내부 8틱 스로틀, 웹 probeSceneLight 등가)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_items_probe(rgb: &[u8], w: u32, h: u32) -> Result<(), JsValue> {
    STUDIO.with(|c| {
        let mut b = c.borrow_mut();
        let s = b.as_mut().ok_or_else(|| JsValue::from_str("studio_attach 먼저"))?;
        if let Some(items) = s.items.as_mut() {
            items.renderer.probe_scene_light_rgb(rgb, w as usize, h as usize);
        }
        Ok(())
    })
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
/// ch=1 알파(RVM pha 등가), ch=2 로짓 [bg, person]. 마스크 해상도는 자유
/// (mask_w×mask_h). ema=false면 시간 상태를 끊는다 (공간 스택 결정적 게이트).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn vb_gate_frame(
    seg: u32,
    frame_rgba: &[u8],
    fw: u32,
    fh: u32,
    mask: &[f32],
    ch: u32,
    mask_w: u32,
    mask_h: u32,
    ema: bool,
) -> Result<Vec<u8>, JsValue> {
    let ctx = engine()?;
    let seg_s = GPU_MODELS.with(|p| p.borrow_mut().take(seg)).map_err(js_err)?;
    let mut g = VB_GATE.with(|c| c.borrow_mut().take());
    let result = match g.as_mut() {
        Some(g) => g
            .frame(&ctx, &seg_s, frame_rgba, fw, fh, mask, ch, mask_w, mask_h, ema)
            .await
            .map_err(js_err),
        None => Err(JsValue::from_str("vb_gate_reset 먼저")),
    };
    VB_GATE.with(|c| *c.borrow_mut() = g);
    GPU_MODELS.with(|p| p.borrow_mut().put(seg, seg_s));
    result
}

/// 제출된 GPU 작업 완료 시 resolve (onSubmittedWorkDone — **논블로킹**).
/// HUD 정직화의 주기 샘플용: JS가 fire-and-forget으로 걸고 .then에서 시각을 재면
/// 렌더 루프를 세우지 않고 실제 GPU 레이턴시를 얻는다.
#[wasm_bindgen]
pub async fn gpu_sync() -> Result<(), JsValue> {
    let ctx = engine()?;
    ai_gpu::readback::wait_idle(&ctx).await.map_err(js_err)
}

/// B티어 프레임: 소스 캔버스 + 외부(CPU 추론) 마스크 → 이펙트 스택 → 캔버스.
/// seg = GPU 세션 핸들(리소스 치수용 — 추론은 안 돈다). ch=2면 [bg, person] 로짓.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn studio_frame_mask(
    seg: u32,
    source: web_sys::HtmlCanvasElement,
    mask: &[f32],
    ch: u32,
    mask_w: u32,
    mask_h: u32,
) -> Result<(), JsValue> {
    let ctx = engine()?;
    let seg_s = GPU_MODELS.with(|p| p.borrow_mut().take(seg)).map_err(js_err)?;
    let mut st = STUDIO.with(|c| c.borrow_mut().take());
    let result = match st.as_mut() {
        Some(s) => {
            s.frame_mask(&ctx, &seg_s, &source, mask, ch, mask_w, mask_h).map_err(js_err)
        }
        None => Err(JsValue::from_str("studio_attach 먼저")),
    };
    STUDIO.with(|c| *c.borrow_mut() = st);
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
