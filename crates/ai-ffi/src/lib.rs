//! ai-ffi — 모바일 C ABI 바인딩.
//!
//! 표면은 vcxrust_ai에서 앱이 **실제로 쓰던 기능**을 따르되, 모양은 우리 엔진에
//! 맞춘다 (사용자 확정 원칙 — 파리티 강박 없음):
//! - 설정은 **단일 JSON 채널** `update_effects_config` — EffectsPatch(배경 hex/
//!   "image"·블러 0..1·밝기 1.0 기준·흑백 0..1·미러/회전·조명·프레이밍·터치업·
//!   메이크업) + 태스크 키(faceItems/handDetection/focusDetection).
//!   vcxrust의 VideoOptionsC(레거시 구조체 채널)는 재현하지 않는다.
//! - 배경 이미지는 base64 인코딩 대신 **바이너리 채널** `set_background_image`.
//! - 결과는 우리 타입 JSON: `get_focus_state`(FocusResult 7상태 전체),
//!   `poll_hand_gesture`(FIFO 16). 문자열은 `vcx_string_free`로 해제.
//! - 오케스트레이션은 전부 `ai_tasks::Director` — 이 크레이트에 남는 것은
//!   C 타입 변환, fs 조달(모델/GLB 경로), YUV 왕복(gpu_yuv.rs · yuv.rs), panic 방벽뿐.
//!
//! render_mask 계약(모바일): I420 in-place, stride 보존, width/height 음수 허용
//! (|절대값| 사용 — INTEGRATION.md §계약). passthrough(효과 전무)면 변환조차
//! 하지 않고 즉시 Success — analyzer-only(손/집중도만)면 세그·합성을 건너뛰고
//! 태스크만 돈다 (프레임 무수정).
//!
//! YUV↔RGB 는 GPU 가 평면에서 바로 한다(gpu_yuv.rs) — CPU 변환(yuv.rs)은 CPU 픽셀이
//! 필요한 프레임(B 티어 세그 · 태스크 픽셀 틱 · 합성 없는 뒤집기)과 C 티어만 쓴다.
//! ⚠ JNI 래퍼(Android)는 모바일 실연결 때 이 C 표면을 감싼다 (로드맵 E).

mod gpu_yuv;
pub mod yuv;

use gpu_yuv::{Dst, GpuYuv, Src};

// 안드로이드: C ABI 대신 JNI 심볼 (vcxrust_ai와 같은 플랫폼 분리 — 심볼은
// 이 크레이트의 C 표면을 그대로 감싼다, 로직 없음)
#[cfg(target_os = "android")]
pub mod java_api;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::Mutex;
use std::time::Instant;

use std::sync::atomic::Ordering;

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::audio::{Denoiser, Enhancer, Vad};
use ai_tasks::features::vb::SoftPipeline;
use ai_tasks::Director;

/// 0=Success, -1=Failure (vcxrust_ai VbResult와 동일 값 — 기존 호스트의
/// `== -1` 검사가 조용히 깨지지 않게 -1을 쓴다)
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VbResult {
    Success = 0,
    Failure = -1,
}

struct Offscreen {
    /// 뷰의 주인 — 읽고 쓰는 건 뷰로만 한다
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
}

struct Ffi {
    ctx: GpuContext,
    director: Director,
    target: Option<Offscreen>,
    rgba: Vec<u8>,
    rgb: Vec<u8>,
    epoch: Instant,
    /// mirror/degree — 프레임 변환은 호스트 몫 계약이라 엔진에 안 넘기고 여기 산다.
    /// 인물(프레임)만 뒤집고 배경 이미지는 화면에 고정된다 (모바일 제품 스펙).
    mirror: bool,
    degree: f32,
    /// B 티어 CPU 세그 (infer_mask만 사용 — 합성·EMA는 Director/GPU 소유)
    cpu_seg: Option<SoftPipeline>,
    /// I420 ↔ 프레임/타깃 텍스처 GPU 변환 — 첫 렌더 프레임에 만든다
    gpu_yuv: Option<GpuYuv>,
}

static STATE: Mutex<Option<Ffi>> = Mutex::new(None);

/// C 티어(소프트 합성) 상태 — 호스트가 set_render_tier(2)로 선언한 기기 전용.
/// GPU 심볼(wgpu 컨텍스트·셰이더 컴파일)을 일절 만들지 않는다 — 비호환 드라이버가
/// 컴파일 중 프로세스째 죽는(SIGSEGV) 기기가 진입 조건이라, 이 경로의 존재 이유가
/// "GPU를 건드리지 않음" 그 자체다.
struct Soft {
    pipe: SoftPipeline,
    rgba: Vec<u8>,
    mirror: bool,
    degree: f32,
}

use std::sync::atomic::AtomicI32;

/// 렌더 티어 (INTEGRATION.md §1.5): 0=A(GPU 추론+GPU 합성), 1=B(CPU 추론+GPU
/// 합성 — GPU는 살아있지만 추론이 느린 기기), 2=C(전부 CPU — GPU 자체가 불가).
static TIER: AtomicI32 = AtomicI32::new(0);
static SOFT: Mutex<Option<Soft>> = Mutex::new(None);

fn soft_state(guard: &mut Option<Soft>) -> &mut Soft {
    guard.get_or_insert_with(|| Soft {
        pipe: SoftPipeline::default(),
        rgba: Vec::new(),
        mirror: false,
        degree: 0.0,
    })
}

/// 렌더 티어 선언 — 0=A(GPU), 1=B(CPU 추론+GPU 합성), 2=C(GPU 완전 회피).
/// 모델 주입 전에 호출해야 한다. B/C 전환 시 세그 모델(경량 CPU용)을
/// set_video_stream_info로 다시 주입하는 것까지가 호스트 몫.
#[unsafe(no_mangle)]
pub extern "C" fn set_render_tier(tier: c_int) -> VbResult {
    if !(0..=2).contains(&tier) {
        return VbResult::Failure;
    }
    TIER.store(tier, Ordering::Relaxed);
    VbResult::Success
}

fn soft_tier() -> bool {
    TIER.load(Ordering::Relaxed) == 2
}

fn b_tier() -> bool {
    TIER.load(Ordering::Relaxed) == 1
}

fn ffi_log(msg: &str) {
    log::error!("[ai-ffi] {msg}");
    eprintln!("[ai-ffi] {msg}");
}

fn handle_panic() -> VbResult {
    ffi_log("panic recovered; returning Failure");
    VbResult::Failure
}

/// 컨텍스트+Director 지연 생성 — 어느 setter가 먼저 와도 동작한다
fn ensure_state(guard: &mut Option<Ffi>) -> Result<&mut Ffi, String> {
    if guard.is_none() {
        let ctx = GpuContext::new_blocking().map_err(|e| format!("GPU 컨텍스트: {e}"))?;
        let director = Director::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
        *guard = Some(Ffi {
            ctx,
            director,
            target: None,
            rgba: Vec::new(),
            rgb: Vec::new(),
            epoch: Instant::now(),
            mirror: false,
            degree: 0.0,
            cpu_seg: None,
            gpu_yuv: None,
        });
    }
    Ok(guard.as_mut().unwrap())
}

unsafe fn cstr(p: *const c_char) -> Result<String, String> {
    if p.is_null() {
        return Err("null 문자열".into());
    }
    Ok(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// (kind, 경로) 모델 파일을 읽어 Director에 주입
fn load_model_file(st: &mut Ffi, kind: &str, path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{kind}: read {path}: {e}"))?;
    st.director.set_model(kind, bytes)
}

// ───────── 모델 주입 ─────────

/// 세그(.sw) 모델 경로 — vcxrust_ai와 동일 시그니처(ptr+len, NUL 아님·UTF-8)
///
/// # Safety
/// `model_path_ptr..model_path_len`은 유효한 UTF-8 바이트여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_video_stream_info(
    model_path_ptr: *const u8,
    model_path_len: usize,
) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model_path_ptr.is_null() {
            ffi_log("set_video_stream_info: null path");
            return VbResult::Failure;
        }
        let path_bytes = unsafe { slice::from_raw_parts(model_path_ptr, model_path_len) };
        let Ok(path) = std::str::from_utf8(path_bytes) else {
            ffi_log("set_video_stream_info: path not utf-8");
            return VbResult::Failure;
        };
        if soft_tier() {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    ffi_log(&format!("set_video_stream_info(soft): read {path}: {e}"));
                    return VbResult::Failure;
                }
            };
            soft_state(&mut SOFT.lock().unwrap()).pipe.set_model(bytes);
            return VbResult::Success;
        }
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: {e}"));
                return VbResult::Failure;
            }
        };
        if b_tier() {
            // B 티어: 세그는 CPU — GPU 세그 세션(RVM 컴파일)을 만들지 않는다.
            // 합성/태스크는 Director(GPU)가 그대로 소유.
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    ffi_log(&format!("set_video_stream_info(b): read {path}: {e}"));
                    return VbResult::Failure;
                }
            };
            let mut pipe = SoftPipeline::default();
            pipe.set_model(bytes);
            st.cpu_seg = Some(pipe);
            st.director.reset(); // 기존 GPU 세그 세션/파이프라인 리소스 반납
            return VbResult::Success;
        }
        match load_model_file(st, "seg", path) {
            Ok(()) => VbResult::Success,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: {e}"));
                VbResult::Failure
            }
        }
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 얼굴 검출+랜드마크(.sw) 경로 — 아이템·터치업/메이크업·집중도의 전제
///
/// # Safety
/// det_path/lm_path는 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_face_model_info(
    det_path: *const c_char,
    lm_path: *const c_char,
) -> VbResult {
    set_model_pair("face_det", det_path, "face_lm", lm_path)
}

/// 손 검출+랜드마크(.sw) 경로 — handDetection(제스처)의 전제
///
/// # Safety
/// det_path/lm_path는 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_hand_model_info(
    det_path: *const c_char,
    lm_path: *const c_char,
) -> VbResult {
    set_model_pair("hand_det", det_path, "hand_lm", lm_path)
}

/// 게이즈 CNN(.sw) + blendshapes(.sw, **nullable** — 없으면 blink는 EAR 절반만)
/// — focusDetection의 전제
///
/// # Safety
/// gaze_path는 NUL 종료 C 문자열, bs_path는 같거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_gaze_model_info(
    gaze_path: *const c_char,
    bs_path: *const c_char,
) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if soft_tier() {
            return VbResult::Success;
        }
        let gaze = match unsafe { cstr(gaze_path) } {
            Ok(v) => v,
            Err(e) => {
                ffi_log(&format!("set_gaze_model_info: {e}"));
                return VbResult::Failure;
            }
        };
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_gaze_model_info: {e}"));
                return VbResult::Failure;
            }
        };
        if let Err(e) = load_model_file(st, "gaze", &gaze) {
            ffi_log(&format!("set_gaze_model_info: {e}"));
            return VbResult::Failure;
        }
        if !bs_path.is_null() {
            let bs = unsafe { CStr::from_ptr(bs_path) }.to_string_lossy().into_owned();
            if let Err(e) = load_model_file(st, "gaze_bs", &bs) {
                // bs는 선택 — 실패해도 EAR 절반으로 동작 (로그만)
                ffi_log(&format!("set_gaze_model_info(bs, 선택): {e}"));
            }
        }
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

fn set_model_pair(
    kind_a: &'static str,
    path_a: *const c_char,
    kind_b: &'static str,
    path_b: *const c_char,
) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // C 티어는 얼굴/손 스택이 없다 (효과별 최소 티어 = B 이상) — 조용히 성공
        if soft_tier() {
            return VbResult::Success;
        }
        let (a, b) = match (unsafe { cstr(path_a) }, unsafe { cstr(path_b) }) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                ffi_log(&format!("set_model({kind_a}/{kind_b}): {e}"));
                return VbResult::Failure;
            }
        };
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_model: {e}"));
                return VbResult::Failure;
            }
        };
        match load_model_file(st, kind_a, &a).and_then(|()| load_model_file(st, kind_b, &b)) {
            Ok(()) => VbResult::Success,
            Err(e) => {
                ffi_log(&format!("set_model: {e}"));
                VbResult::Failure
            }
        }
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 3D 아이템 GLB 디렉터리 — faceItems가 켜지면 `{dir}/{kind}.glb`를 지연 로드
///
/// # Safety
/// dir은 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_item_model_dir(dir: *const c_char) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // soft tier(C)여도 로더는 등록한다 — 등록은 클로저 저장뿐이라 GPU 를 안 건드리고,
        // 여기서 건너뛰면 이 세션에서 티어가 A/B 로 회복돼도 GLB 로더가 영영 비어
        // 3D 아이템이 조용히 안 뜬다. 실제 렌더 차단은 프레임 경로의 티어 게이트 몫.
        let dir = match unsafe { cstr(dir) } {
            Ok(v) => v,
            Err(e) => {
                ffi_log(&format!("set_item_model_dir: {e}"));
                return VbResult::Failure;
            }
        };
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_item_model_dir: {e}"));
                return VbResult::Failure;
            }
        };
        st.director.set_glb_loader(Box::new(move |kind| {
            // 아바타 프리셋(vrm/{model})은 .vrm 그대로 — 웹 에셋과 파일명·확장자 동일.
            // VRM 은 바이너리 glTF 컨테이너라 파서는 .glb 와 같은 경로를 탄다.
            let ext = if kind.starts_with("vrm/") { "vrm" } else { "glb" };
            let path = format!("{dir}/{kind}.{ext}");
            match std::fs::read(&path) {
                Ok(b) => Some(b),
                Err(e) => {
                    ffi_log(&format!("GLB {path}: {e}"));
                    None // 실패 캐시는 Director가 담당 (재시도 스팸 방지)
                }
            }
        }));
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

// ───────── 설정 ─────────

/// 단일 JSON 설정 — EffectsPatch + faceItems/handDetection/focusDetection.
/// 머지 규약: 없음=유지 / null=해제 / 값=설정.
///
/// # Safety
/// `json`은 유효한 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn update_effects_config(json: *const c_char) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let json = match unsafe { cstr(json) } {
            Ok(v) => v,
            Err(e) => {
                ffi_log(&format!("update_effects_config: {e}"));
                return VbResult::Failure;
            }
        };
        // mirror/degree — 프레임 변환은 호스트(ffi) 몫이라 값을 읽어 st에 흡수하되,
        // 키는 엔진에도 그대로 전달한다. 엔진 compose가 배경 샘플 좌표를 같은 변환으로
        // 보정해 "배경 포함 전체 거울"이 된다 (웹 wasm 스펙 동일 — 2026-08-27 통일).
        let mut mirror_patch: Option<bool> = None;
        let mut degree_patch: Option<f32> = None;
        let forwarded = match serde_json::from_str::<serde_json::Value>(&json) {
            Ok(v) => {
                if let Some(obj) = v.as_object() {
                    mirror_patch = obj.get("mirror").map(|m| m.as_bool().unwrap_or(false));
                    degree_patch = obj
                        .get("degree")
                        .map(|d| (d.as_f64().unwrap_or(0.0) as f32).rem_euclid(360.0));
                }
                v.to_string()
            }
            Err(_) => json, // 파싱 실패는 엔진 쪽 에러 메시지로 일원화
        };
        if soft_tier() {
            let mut guard = SOFT.lock().unwrap();
            let st = soft_state(&mut guard);
            if let Some(m) = mirror_patch {
                st.mirror = m;
            }
            if let Some(d) = degree_patch {
                st.degree = d;
            }
            return match st.pipe.apply_json(&forwarded) {
                Ok(()) => VbResult::Success,
                Err(e) => {
                    ffi_log(&format!("update_effects_config(soft): {e}"));
                    VbResult::Failure
                }
            };
        }
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("update_effects_config: {e}"));
                return VbResult::Failure;
            }
        };
        if let Some(m) = mirror_patch {
            st.mirror = m;
        }
        if let Some(d) = degree_patch {
            st.degree = d;
        }
        match st.director.apply_json(&forwarded) {
            Ok(()) => VbResult::Success,
            Err(e) => {
                ffi_log(&format!("update_effects_config: {e}"));
                VbResult::Failure
            }
        }
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 배경 이미지 (RGBA8, len = w*h*4) — EffectsPatch background:"image"가 소비.
/// vcxrust의 base64-인-JSON 대신 바이너리 채널 (인코딩 왕복 제거).
///
/// # Safety
/// rgba는 w*h*4 바이트의 유효한 버퍼여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_background_image(
    rgba: *const u8,
    width: i32,
    height: i32,
) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if rgba.is_null() || width <= 0 || height <= 0 {
            ffi_log("set_background_image: bad args");
            return VbResult::Failure;
        }
        let (w, h) = (width as u32, height as u32);
        let data = unsafe { slice::from_raw_parts(rgba, (w * h * 4) as usize) };
        if soft_tier() {
            soft_state(&mut SOFT.lock().unwrap())
                .pipe
                .set_background_image(data, w as usize, h as usize);
            return VbResult::Success;
        }
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_background_image: {e}"));
                return VbResult::Failure;
            }
        };
        let ctx = &st.ctx;
        st.director.set_background_image(ctx, data, w, h);
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 다중 모니터 레이아웃 JSON ({monitors:[...], targetIndex} | "null") —
/// focusDetection을 켠 뒤 호출 (레이아웃 조달은 호스트 몫)
///
/// # Safety
/// `json`은 유효한 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_focus_layout(json: *const c_char) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let json = match unsafe { cstr(json) } {
            Ok(v) => v,
            Err(e) => {
                ffi_log(&format!("set_focus_layout: {e}"));
                return VbResult::Failure;
            }
        };
        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else {
            ffi_log("set_focus_layout: 미초기화");
            return VbResult::Failure;
        };
        match st.director.set_focus_layout_json(&json) {
            Ok(()) => VbResult::Success,
            Err(e) => {
                ffi_log(&format!("set_focus_layout: {e}"));
                VbResult::Failure
            }
        }
    }));
    result.unwrap_or_else(|_| handle_panic())
}

// ───────── 프레임 ─────────

/// |v| (i32::MIN만 거부) — 모바일 계약: 음수 width/height/stride 허용
fn abs_dim(v: i32) -> Option<usize> {
    if v == i32::MIN {
        return None;
    }
    Some(v.unsigned_abs() as usize)
}

/// I420 in-place 처리: passthrough면 무가공, analyzer-only면 태스크만,
/// 아니면 세그+이펙트+아이템 합성 결과를 같은 평면에 되쓴다.
///
/// # Safety
/// y/u/v는 |stride|×|h|(/2) 크기의 유효한 가변 평면이어야 한다.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn render_mask(
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
    width: i32,
    height: i32,
    stride_y: i32,
    stride_u: i32,
    stride_v: i32,
) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims = (
            abs_dim(width),
            abs_dim(height),
            abs_dim(stride_y),
            abs_dim(stride_u),
            abs_dim(stride_v),
        );
        let (Some(w), Some(h), Some(sy), Some(su), Some(sv)) = dims else {
            ffi_log("render_mask: i32::MIN 치수");
            return VbResult::Failure;
        };
        if y.is_null() || u.is_null() || v.is_null() || w == 0 || h == 0 {
            ffi_log("render_mask: bad args");
            return VbResult::Failure;
        }
        if sy < w || su < w.div_ceil(2) || sv < w.div_ceil(2) {
            ffi_log(&format!("render_mask: stride<w ({sy},{su},{sv} vs {w})"));
            return VbResult::Failure;
        }
        let (yb, ub, vb) = unsafe {
            (
                slice::from_raw_parts_mut(y, sy * h),
                slice::from_raw_parts_mut(u, su * h.div_ceil(2)),
                slice::from_raw_parts_mut(v, sv * h.div_ceil(2)),
            )
        };

        if soft_tier() {
            return render_mask_soft(yb, ub, vb, w, h, sy, su, sv);
        }
        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else {
            ffi_log("render_mask: 미초기화 (set_video_stream_info 먼저)");
            return VbResult::Failure;
        };
        // mirror/degree 프레임 변환은 호스트 몫 계약 — ffi가 추론 전에 적용한다
        // (웹 워커 전처리 등가). 90/270은 in-place I420 치수 계약과 상충해 미지원.
        let (mirror, degree) = (st.mirror, st.degree);
        let rot180 = (degree - 180.0).abs() < 0.5;
        if degree != 0.0 && !rot180 {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                ffi_log(&format!("render_mask: degree {degree} 미지원 (0/180만) — 무시"));
            }
        }
        let (flip_h, flip_v) = (mirror ^ rot180, rot180);
        let flip = flip_h || flip_v;
        if st.director.passthrough() {
            // 완전 무가공 — 변환조차 하지 않는다 (최대 절약)
            if !flip {
                return VbResult::Success;
            }
            // mirror/180만 켠 경우 — GPU 없이 CPU 왕복만으로 뒤집어 되쓴다
            st.rgba.resize(w * h * 4, 0);
            yuv::i420_to_rgba(yb, ub, vb, w, h, sy, su, sv, &mut st.rgba);
            yuv::flip_rgba(&mut st.rgba, w, h, flip_h, flip_v);
            yuv::rgba_to_i420(&st.rgba, w, h, yb, ub, vb, sy, su, sv);
            return VbResult::Success;
        }
        let t_ms = st.epoch.elapsed().as_secs_f64() * 1e3;
        let needs = st.director.needs_render();
        let wants_pixels = st.director.wants_pixels(t_ms);
        // CPU 픽셀이 필요한 프레임만 CPU 로 바꾼다 — B 티어 세그 입력 · 태스크 픽셀 틱 ·
        // 합성 없이 뒤집힌 원본만 되쓰는 프레임. 나머지는 GPU 가 평면에서 바로 프레임
        // 텍스처를 채우고(gpu_yuv) 합성 결과도 평면으로 바로 내린다.
        let cpu_pixels = (b_tier() && needs) || wants_pixels || (!needs && flip);
        if cpu_pixels {
            st.rgba.resize(w * h * 4, 0);
            yuv::i420_to_rgba(yb, ub, vb, w, h, sy, su, sv, &mut st.rgba);
            if flip {
                yuv::flip_rgba(&mut st.rgba, w, h, flip_h, flip_v);
            }
        }

        // B 티어: 세그를 CPU에서 먼저 — 마스크 해상도가 프레임 리소스 ensure에 필요.
        // 마스크는 EMA 없이 원시로 넘긴다 (시간 상태는 GPU ingest 소유 — A와 동일 규약).
        let Ffi { ctx, director, rgba, rgb, target, cpu_seg, gpu_yuv, .. } = st;
        let gpu_yuv = gpu_yuv.get_or_insert_with(|| GpuYuv::new(ctx));
        let mut bmask: Option<(&[f32], u32, u32, u32)> = None;
        if b_tier() && needs {
            let Some(cpu) = cpu_seg.as_mut() else {
                ffi_log("render_mask(b): CPU 세그 미주입 (set_video_stream_info 먼저)");
                return VbResult::Failure;
            };
            match cpu.infer_mask(rgba, w, h) {
                Ok(m) => bmask = Some(m),
                Err(e) => {
                    ffi_log(&format!("render_mask(b): CPU 추론: {e}"));
                    return VbResult::Failure;
                }
            }
        }

        // 프레임 텍스처 채우기 (A: 세그 ensure / B: 마스크 해상도 ensure / 태스크: 세션리스)
        let fill = |tex: &wgpu::Texture| {
            if !cpu_pixels {
                let strides = [sy as u32, su as u32, sv as u32];
                let src = Src::I420 { y: yb, u: ub, v: vb, strides };
                gpu_yuv.upload(ctx, tex, src, w as u32, h as u32, 0, flip_h, flip_v);
                return;
            }
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w as u32 * 4),
                    rows_per_image: Some(h as u32),
                },
                wgpu::Extent3d {
                    width: w as u32,
                    height: h as u32,
                    depth_or_array_layers: 1,
                },
            );
        };
        let upload = if let Some((_, mw, mh, _)) = bmask {
            director.with_frame_mask(ctx, w as u32, h as u32, (mw, mh), fill)
        } else {
            pollster::block_on(director.with_frame(ctx, w as u32, h as u32, fill))
        };
        if let Err(e) = upload {
            ffi_log(&format!("render_mask: 프레임 업로드: {e}"));
            return VbResult::Failure;
        }

        // 태스크용 u8 RGB — 필요한 틱에만 (알파 제거)
        let rgb_opt = if wants_pixels {
            rgb.resize(w * h * 3, 0);
            for i in 0..w * h {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&rgba[i * 4..i * 4 + 3]);
            }
            Some(rgb.as_slice())
        } else {
            None
        };

        if needs {
            ensure_target(ctx, target, w as u32, h as u32);
        }
        let view = target.as_ref().filter(|_| needs).map(|t| &t.view);
        let framed = if let Some((mask, mw, mh, ch)) = bmask {
            // B 티어 — 외부 마스크 주입 (needs_render일 때만 bmask가 생긴다)
            let tv = view.expect("bmask는 needs_render 전제");
            pollster::block_on(director.frame_mask(
                ctx, mask, ch, mw, mh, w as u32, h as u32, tv, rgb_opt, t_ms,
            ))
        } else {
            pollster::block_on(director.frame(ctx, w as u32, h as u32, view, rgb_opt, t_ms))
        };
        if let Err(e) = framed {
            ffi_log(&format!("render_mask: {e}"));
            return VbResult::Failure;
        }
        if needs {
            let t = target.as_ref().unwrap();
            let dst = Dst::I420 { y: yb, u: ub, v: vb, strides: [sy, su, sv] };
            if let Err(e) = gpu_yuv.download(ctx, &t.view, w as u32, h as u32, dst) {
                ffi_log(&format!("render_mask: 리드백: {e}"));
                return VbResult::Failure;
            }
        } else if flip {
            // mirror/180만 켠 경우 — 합성 없이 뒤집힌 원본을 되쓴다
            yuv::rgba_to_i420(rgba, w, h, yb, ub, vb, sy, su, sv);
        }
        // analyzer-only: 프레임 무수정 (원본이 그대로 나간다 — vcxrust 고속경로 등가)
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 합성 타깃 확보 — 크기가 바뀔 때만 다시 만든다.
fn ensure_target(ctx: &GpuContext, target: &mut Option<Offscreen>, w: u32, h: u32) {
    if matches!(target, Some(t) if t.w == w && t.h == h) {
        return;
    }
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ai-ffi-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        // TEXTURE_BINDING — gpu_yuv 가 합성 결과를 읽어 평면으로 내린다
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    *target = Some(Offscreen { _tex: tex, view, w, h });
}

/// render_mask_nv12 · render_mask_i420_io 결과.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameResult {
    /// 출력 평면에 합성 결과를 썼다
    Written = 0,
    /// 프레임 무가공(passthrough · analyzer-only) — 출력 평면은 건드리지 않았다, 원본을 그대로 내보낸다
    Unchanged = 1,
    /// 이 프레임은 GPU 평면 경로로 못 한다(B/C 티어 · 태스크 픽셀 틱 · 합성 없는 뒤집기 ·
    /// 홀수 크기 회전) — 엔진 상태를 건드리지 않았으니 호출자는 같은 프레임을 예전 경로
    /// (호스트 회전 + render_mask I420)로 처리한다
    Fallback = 2,
    Failure = -1,
}

/// 입력 평면 → (카메라 회전) → 합성 → 출력 평면 — render_mask 의 GPU 경로를 평면 그대로 탄다.
/// 입력 `sw×sh` 는 읽기만 하고, 결과(회전 90/270 이면 `sh×sw`)는 `dst` 에 쓴다.
fn render_planes(name: &str, src: Src, sw: usize, sh: usize, rot: u32, dst: Dst) -> FrameResult {
    // CPU 픽셀이 필요한 티어는 예전 경로 몫
    if soft_tier() || b_tier() {
        return FrameResult::Fallback;
    }
    let mut guard = STATE.lock().unwrap();
    let Some(st) = guard.as_mut() else {
        ffi_log(&format!("{name}: 미초기화 (set_video_stream_info 먼저)"));
        return FrameResult::Failure;
    };
    // mirror/degree — render_mask 와 같은 규칙 (90/270 은 무시)
    let rot180 = (st.degree - 180.0).abs() < 0.5;
    let (flip_h, flip_v) = (st.mirror ^ rot180, rot180);
    let flip = flip_h || flip_v;
    if st.director.passthrough() {
        return if flip { FrameResult::Fallback } else { FrameResult::Unchanged };
    }
    let t_ms = st.epoch.elapsed().as_secs_f64() * 1e3;
    let needs = st.director.needs_render();
    // 아래부터 엔진 상태가 바뀐다 — 넘기는 판정은 전부 여기까지
    if st.director.wants_pixels(t_ms) || (!needs && flip) {
        return FrameResult::Fallback;
    }
    let (w, h) = if rot == 90 || rot == 270 { (sh as u32, sw as u32) } else { (sw as u32, sh as u32) };

    let Ffi { ctx, director, target, gpu_yuv, .. } = st;
    let gpu_yuv = gpu_yuv.get_or_insert_with(|| GpuYuv::new(ctx));
    let fill = |tex: &wgpu::Texture| {
        gpu_yuv.upload(ctx, tex, src, sw as u32, sh as u32, rot, flip_h, flip_v);
    };
    if let Err(e) = pollster::block_on(director.with_frame(ctx, w, h, fill)) {
        ffi_log(&format!("{name}: 프레임 업로드: {e}"));
        return FrameResult::Failure;
    }
    if needs {
        ensure_target(ctx, target, w, h);
    }
    let view = target.as_ref().filter(|_| needs).map(|t| &t.view);
    if let Err(e) = pollster::block_on(director.frame(ctx, w, h, view, None, t_ms)) {
        ffi_log(&format!("{name}: {e}"));
        return FrameResult::Failure;
    }
    if !needs {
        // analyzer-only: 프레임 무수정
        return FrameResult::Unchanged;
    }
    let t = target.as_ref().unwrap();
    if let Err(e) = gpu_yuv.download(ctx, &t.view, w, h, dst) {
        ffi_log(&format!("{name}: 리드백: {e}"));
        return FrameResult::Failure;
    }
    FrameResult::Written
}

fn frame_panic() -> FrameResult {
    handle_panic();
    FrameResult::Failure
}

/// NV12 입력(iOS 카메라 CVPixelBuffer 형식, 카메라 회전 `rotation` = 0/90/180/270, 시계 방향)
/// → 세운 NV12 출력 — 호스트의 NV12→I420 변환·회전과 되돌리는 인터리브 복사가 없어진다.
/// 입력 `width×height` 는 읽기만 하고(카메라 버퍼) 결과(90/270 이면 `height×width`)는 별도
/// 출력 평면(stride 보존)에 쓴다. 결과 바이트는 같은 프레임을 libyuv 로 세워 I420 으로
/// render_mask 에 넣은 것과 같다. 회전하는 홀수 크기는 Fallback.
///
/// # Safety
/// y_in/uv_in 은 |stride_in|×|height| · |stride_uv_in|×⌈|height|/2⌉ 크기의 읽기 평면,
/// y_out/uv_out 은 세운 크기 기준 같은 규칙의 쓰기 평면이어야 한다(입력과 겹치지 않을 것).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn render_mask_nv12(
    y_in: *const u8,
    uv_in: *const u8,
    width: i32,
    height: i32,
    stride_y_in: i32,
    stride_uv_in: i32,
    rotation: i32,
    y_out: *mut u8,
    uv_out: *mut u8,
    stride_y_out: i32,
    stride_uv_out: i32,
) -> FrameResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims = (
            abs_dim(width),
            abs_dim(height),
            abs_dim(stride_y_in),
            abs_dim(stride_uv_in),
            abs_dim(stride_y_out),
            abs_dim(stride_uv_out),
        );
        let (Some(w), Some(h), Some(syi), Some(suvi), Some(syo), Some(suvo)) = dims else {
            ffi_log("render_mask_nv12: i32::MIN 치수");
            return FrameResult::Failure;
        };
        let ptrs_ok = !(y_in.is_null() || uv_in.is_null() || y_out.is_null() || uv_out.is_null());
        if !ptrs_ok || w == 0 || h == 0 || ![0, 90, 180, 270].contains(&rotation) {
            ffi_log(&format!("render_mask_nv12: bad args (rotation {rotation})"));
            return FrameResult::Failure;
        }
        let rot = rotation as u32;
        // 크로마를 원본 좌표 절반으로 읽는 회전은 짝수 크기에서만 libyuv 평면 회전과 같다
        if rot != 0 && (w % 2 == 1 || h % 2 == 1) {
            return FrameResult::Fallback;
        }
        let (ow, oh) = if rot == 90 || rot == 270 { (h, w) } else { (w, h) };
        if syi < w || suvi < 2 * w.div_ceil(2) || syo < ow || suvo < 2 * ow.div_ceil(2) {
            ffi_log(&format!(
                "render_mask_nv12: stride<w ({syi},{suvi} / {syo},{suvo} vs {w}x{h} rot{rot})"
            ));
            return FrameResult::Failure;
        }
        let (src, dst) = unsafe {
            (
                Src::Nv12 {
                    y: slice::from_raw_parts(y_in, syi * h),
                    uv: slice::from_raw_parts(uv_in, suvi * h.div_ceil(2)),
                    strides: [syi as u32, suvi as u32],
                },
                Dst::Nv12 {
                    y: slice::from_raw_parts_mut(y_out, syo * oh),
                    uv: slice::from_raw_parts_mut(uv_out, suvo * oh.div_ceil(2)),
                    strides: [syo, suvo],
                },
            )
        };
        render_planes("render_mask_nv12", src, w, h, rot, dst)
    }));
    result.unwrap_or_else(|_| frame_panic())
}

/// I420 입력(카메라 회전 `rotation` = 0/90/180/270, 시계 방향) → 세운 I420 출력 — Android
/// 호스트의 회전(I420Rotate)·출력 버퍼로 옮기는 복사가 없어진다. 입력 `width×height` 는 읽기만
/// 하고, 결과(90/270 이면 `height×width`)는 별도 출력 평면(stride 보존)에 쓴다. 결과 바이트는
/// 같은 프레임을 libyuv 로 세워 render_mask 에 넣은 것과 같다. 회전하는 홀수 크기는 Fallback.
///
/// # Safety
/// 입력 평면은 |stride|×|height|(크로마 ⌈/2⌉) 크기의 읽기 평면, 출력 평면은 세운 크기 기준
/// 같은 규칙의 쓰기 평면이어야 한다(입력과 겹치지 않을 것).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn render_mask_i420_io(
    y_in: *const u8,
    u_in: *const u8,
    v_in: *const u8,
    width: i32,
    height: i32,
    stride_y_in: i32,
    stride_u_in: i32,
    stride_v_in: i32,
    rotation: i32,
    y_out: *mut u8,
    u_out: *mut u8,
    v_out: *mut u8,
    stride_y_out: i32,
    stride_u_out: i32,
    stride_v_out: i32,
) -> FrameResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims = (
            abs_dim(width),
            abs_dim(height),
            [abs_dim(stride_y_in), abs_dim(stride_u_in), abs_dim(stride_v_in)],
            [abs_dim(stride_y_out), abs_dim(stride_u_out), abs_dim(stride_v_out)],
        );
        let (Some(w), Some(h), [Some(syi), Some(sui), Some(svi)], [Some(syo), Some(suo), Some(svo)]) =
            dims
        else {
            ffi_log("render_mask_i420_io: i32::MIN 치수");
            return FrameResult::Failure;
        };
        let ptrs_ok = ![y_in, u_in, v_in].iter().any(|p| p.is_null())
            && ![y_out, u_out, v_out].iter().any(|p| p.is_null());
        if !ptrs_ok || w == 0 || h == 0 || ![0, 90, 180, 270].contains(&rotation) {
            ffi_log(&format!("render_mask_i420_io: bad args (rotation {rotation})"));
            return FrameResult::Failure;
        }
        let rot = rotation as u32;
        // 크로마를 원본 좌표 절반으로 읽는 회전은 짝수 크기에서만 libyuv 평면 회전과 같다
        if rot != 0 && (w % 2 == 1 || h % 2 == 1) {
            return FrameResult::Fallback;
        }
        let (ow, oh) = if rot == 90 || rot == 270 { (h, w) } else { (w, h) };
        let (cw, ch, ocw, och) = (w.div_ceil(2), h.div_ceil(2), ow.div_ceil(2), oh.div_ceil(2));
        if syi < w || sui < cw || svi < cw || syo < ow || suo < ocw || svo < ocw {
            ffi_log(&format!(
                "render_mask_i420_io: stride<w ({syi},{sui},{svi} / {syo},{suo},{svo} vs {w}x{h} rot{rot})"
            ));
            return FrameResult::Failure;
        }
        let (src, dst) = unsafe {
            (
                Src::I420 {
                    y: slice::from_raw_parts(y_in, syi * h),
                    u: slice::from_raw_parts(u_in, sui * ch),
                    v: slice::from_raw_parts(v_in, svi * ch),
                    strides: [syi as u32, sui as u32, svi as u32],
                },
                Dst::I420 {
                    y: slice::from_raw_parts_mut(y_out, syo * oh),
                    u: slice::from_raw_parts_mut(u_out, suo * och),
                    v: slice::from_raw_parts_mut(v_out, svo * och),
                    strides: [syo, suo, svo],
                },
            )
        };
        render_planes("render_mask_i420_io", src, w, h, rot, dst)
    }));
    result.unwrap_or_else(|_| frame_panic())
}

/// C 티어 프레임 처리 — CPU 추론 + 소프트 합성, GPU 관여 0.
/// mirror/180 호스트 변환은 GPU 티어와 동일 계약으로 여기서도 적용한다.
#[allow(clippy::too_many_arguments)]
fn render_mask_soft(
    yb: &mut [u8],
    ub: &mut [u8],
    vb: &mut [u8],
    w: usize,
    h: usize,
    sy: usize,
    su: usize,
    sv: usize,
) -> VbResult {
    let mut guard = SOFT.lock().unwrap();
    let st = soft_state(&mut guard);
    let rot180 = (st.degree - 180.0).abs() < 0.5;
    let (flip_h, flip_v) = (st.mirror ^ rot180, rot180);
    let flip = flip_h || flip_v;
    if st.pipe.passthrough() && !flip {
        return VbResult::Success;
    }
    st.rgba.resize(w * h * 4, 0);
    yuv::i420_to_rgba(yb, ub, vb, w, h, sy, su, sv, &mut st.rgba);
    if flip {
        yuv::flip_rgba(&mut st.rgba, w, h, flip_h, flip_v);
    }
    let Soft { pipe, rgba, .. } = st;
    match pipe.process(rgba, w, h) {
        Ok(processed) => {
            if processed || flip {
                yuv::rgba_to_i420(rgba, w, h, yb, ub, vb, sy, su, sv);
            }
            VbResult::Success
        }
        Err(e) => {
            ffi_log(&format!("render_mask(soft): {e}"));
            VbResult::Failure
        }
    }
}

// ───────── 결과 폴링 ─────────

/// 마지막 집중도 — {"status":"FOCUSED","attentive":true,"score":100,
/// "monitorIndex":0,"yaw":..,"pitch":..} (7상태 — 우리 FocusResult 그대로).
/// 반환 문자열은 `vcx_string_free`로 해제.
#[unsafe(no_mangle)]
pub extern "C" fn get_focus_state() -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let guard = STATE.lock().unwrap();
        let json = match guard.as_ref() {
            Some(st) => st.director.focus_json(),
            None => return std::ptr::null_mut(),
        };
        CString::new(json).map(|c| c.into_raw()).unwrap_or(std::ptr::null_mut())
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// 제스처 이벤트 하나 (FIFO 16) — 없으면 null.
/// {"gesture":"clap","confidence":0.973,"handedness":"left","tsMs":123.4}
/// 반환 문자열은 `vcx_string_free`로 해제.
#[unsafe(no_mangle)]
pub extern "C" fn poll_hand_gesture() -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = STATE.lock().unwrap();
        let json = match guard.as_mut().and_then(|st| st.director.poll_gesture_json()) {
            Some(j) => j,
            None => return std::ptr::null_mut(),
        };
        CString::new(json).map(|c| c.into_raw()).unwrap_or(std::ptr::null_mut())
    }));
    result.unwrap_or(std::ptr::null_mut())
}

// ───────── 수명 ─────────

/// 스트림 파기 = **리셋** (vcxrust 규약): GPU 세션·파이프라인 리소스는 반납하되
/// 컨텍스트·모델 바이트는 유지 — 다음 render_mask가 지연 재로드로 즉시 살아난다.
#[unsafe(no_mangle)]
pub extern "C" fn destroy_custom_video_stream() -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if let Some(st) = STATE.lock().unwrap().as_mut() {
            st.director.reset();
            st.target = None;
            st.mirror = false;
            st.degree = 0.0;
            if let Some(c) = st.cpu_seg.as_mut() {
                c.reset();
            }
        }
        if let Some(st) = SOFT.lock().unwrap().as_mut() {
            st.pipe.reset();
            st.mirror = false;
            st.degree = 0.0;
        }
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// C 문자열 해제 (get_focus_state/poll_hand_gesture 반환값)
///
/// # Safety
/// `s`는 이 라이브러리가 CString::into_raw로 만든 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vcx_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

// ───────── 오디오 (fastenhancer — CPU 고정, 비디오와 독립 수명) ─────────

/// 불투명 핸들 — None = 미지원 레이트 passthrough (프레임 480, 무가공)
pub struct FeHandle {
    inner: Option<Enhancer>,
    sample_rate: c_int,
}

const FE_PASSTHROUGH_LEN: usize = 480;

/// 생성 — sample_rate 16000/48000이면 `{model_dir}/fe16|fe48/graph.json`+
/// `weights.bin` 로드, 그 외 레이트는 passthrough 핸들 (프레임 480, 무가공 —
/// 호스트가 레이트 스위치 없이도 안전). 실패 시 null.
///
/// # Safety
/// model_dir은 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_create_c(
    sample_rate: c_int,
    model_dir: *const c_char,
) -> *mut FeHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let sub = match sample_rate {
            16000 => Some("fe16"),
            48000 => Some("fe48"),
            _ => None,
        };
        let inner = match sub {
            Some(sub) => {
                let dir = match unsafe { cstr(model_dir) } {
                    Ok(v) => v,
                    Err(e) => {
                        ffi_log(&format!("fe_create_c: {e}"));
                        return std::ptr::null_mut();
                    }
                };
                let graph = match std::fs::read(format!("{dir}/{sub}/graph.json")) {
                    Ok(b) => b,
                    Err(e) => {
                        ffi_log(&format!("fe_create_c: graph.json: {e}"));
                        return std::ptr::null_mut();
                    }
                };
                let weights = match std::fs::read(format!("{dir}/{sub}/weights.bin")) {
                    Ok(b) => b,
                    Err(e) => {
                        ffi_log(&format!("fe_create_c: weights.bin: {e}"));
                        return std::ptr::null_mut();
                    }
                };
                match Enhancer::new(&graph, &weights) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        ffi_log(&format!("fe_create_c: {e}"));
                        return std::ptr::null_mut();
                    }
                }
            }
            None => None,
        };
        Box::into_raw(Box::new(FeHandle { inner, sample_rate }))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `h`는 fe_create_c가 만든 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_free_c(h: *mut FeHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// process_frame 호출당 샘플 수 (hop) — passthrough 핸들은 480
///
/// # Safety
/// `h`는 유효한 FeHandle 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_get_in_frame_len(h: *const FeHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    match &unsafe { &*h }.inner {
        Some(e) => e.frame_len(),
        None => FE_PASSTHROUGH_LEN,
    }
}

/// 생성 시 레이트 (passthrough 핸들 판별: 미지원 레이트 그대로 반환)
///
/// # Safety
/// `h`는 유효한 FeHandle 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_get_sample_rate(h: *const FeHandle) -> c_int {
    if h.is_null() {
        return 0;
    }
    unsafe { &*h }.sample_rate
}

/// 한 hop 처리 — input/output은 frame_len 샘플 mono f32 [-1,1] (겹쳐도 안전).
/// 반환은 예약값 0.0 (vcxrust 규약 유지 — 활동도 자리)
///
/// # Safety
/// input/output은 frame_len 샘플의 유효 버퍼여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fe_process_frame(
    h: *mut FeHandle,
    input: *const c_float,
    output: *mut c_float,
) -> c_float {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if h.is_null() || input.is_null() || output.is_null() {
            return 0.0;
        }
        let handle = unsafe { &mut *h };
        let n = match &handle.inner {
            Some(e) => e.frame_len(),
            None => FE_PASSTHROUGH_LEN,
        };
        // 겹침(in-place) 허용 — 입력을 먼저 복사
        let inp: Vec<f32> = unsafe { slice::from_raw_parts(input, n) }.to_vec();
        let out = unsafe { slice::from_raw_parts_mut(output, n) };
        match &mut handle.inner {
            Some(e) => {
                if let Err(err) = e.process_frame(&inp, out) {
                    ffi_log(&format!("fe_process_frame: {err}"));
                    out.copy_from_slice(&inp); // 실패 시 원음 (무음보다 안전)
                }
            }
            None => out.copy_from_slice(&inp),
        }
        0.0
    }));
    result.unwrap_or(0.0)
}

// ───────── 오디오 (RNNoise — 기본 모드, CPU 고정, 비디오와 독립 수명) ─────────

/// 불투명 핸들 — None = 미지원 레이트 passthrough (프레임 480, 무가공)
pub struct RnHandle {
    inner: Option<Denoiser>,
}

const RN_FRAME_LEN: usize = 480;

/// 생성 — sample_rate 48000이면 RNNoise(가중치 내장 — 모델 파일 없음), 그 외 레이트는
/// passthrough 핸들 (프레임 480, 무가공 — 호스트가 레이트 스위치 없이도 안전).
#[unsafe(no_mangle)]
pub extern "C" fn rn_create_c(sample_rate: c_int) -> *mut RnHandle {
    let result = catch_unwind(|| {
        let inner = (sample_rate as u32 == ai_tasks::features::audio::denoiser::SAMPLE_RATE)
            .then(Denoiser::new);
        Box::into_raw(Box::new(RnHandle { inner }))
    });
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `h`는 rn_create_c가 만든 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rn_free_c(h: *mut RnHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// process_frame 호출당 샘플 수 — 활성·passthrough 모두 480 (10ms@48k)
///
/// # Safety
/// `h`는 유효한 RnHandle 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rn_get_in_frame_len(h: *const RnHandle) -> usize {
    if h.is_null() { 0 } else { RN_FRAME_LEN }
}

/// RNNoise가 실제로 도는 핸들인지 (0 = passthrough)
///
/// # Safety
/// `h`는 유효한 RnHandle 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rn_is_active(h: *const RnHandle) -> c_int {
    if h.is_null() {
        return 0;
    }
    unsafe { &*h }.inner.is_some() as c_int
}

/// 한 프레임 처리 — input/output은 480 샘플 mono f32 [-1,1] (겹쳐도 안전).
/// passthrough 핸들은 원음 그대로.
///
/// # Safety
/// input/output은 480 샘플의 유효 버퍼여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rn_process_frame(
    h: *mut RnHandle,
    input: *const c_float,
    output: *mut c_float,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if h.is_null() || input.is_null() || output.is_null() {
            return;
        }
        let handle = unsafe { &mut *h };
        // 겹침(in-place) 허용 — 입력을 먼저 복사 (스택 — 오디오 스레드에서 10ms마다 불린다)
        let mut inp = [0f32; RN_FRAME_LEN];
        inp.copy_from_slice(unsafe { slice::from_raw_parts(input, RN_FRAME_LEN) });
        let out = unsafe { slice::from_raw_parts_mut(output, RN_FRAME_LEN) };
        match &mut handle.inner {
            Some(d) => d.process_frame(&inp, out),
            None => out.copy_from_slice(&inp),
        }
    }));
}

// ───────── 음성 감지 (Silero VAD — CPU 고정, 비디오와 독립 수명) ─────────

/// 불투명 핸들 — 음소거 중 발화 감지용 음성 확률.
pub struct VadHandle {
    inner: Vad,
}

/// 한 번에 넘길 수 있는 최대 샘플 수 (40ms@48k) — 오디오 스레드에서 스택 버퍼로 받는다.
pub const VAD_MAX_FRAME: usize = 1920;

/// 생성 — `{model_dir}/vad16/graph.json` + `weights.bin` 로드. sample_rate = 넣을 PCM 의
/// 레이트(16000 그대로, 48000 등은 내부에서 16k 로 내린다). 실패 시 null.
///
/// # Safety
/// model_dir은 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vad_create_c(
    sample_rate: c_int,
    model_dir: *const c_char,
) -> *mut VadHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dir = match unsafe { cstr(model_dir) } {
            Ok(v) => v,
            Err(e) => {
                ffi_log(&format!("vad_create_c: {e}"));
                return std::ptr::null_mut();
            }
        };
        let read = |name: &str| std::fs::read(format!("{dir}/vad16/{name}"));
        let (graph, weights) = match (read("graph.json"), read("weights.bin")) {
            (Ok(g), Ok(w)) => (g, w),
            (Err(e), _) | (_, Err(e)) => {
                ffi_log(&format!("vad_create_c: {dir}/vad16: {e}"));
                return std::ptr::null_mut();
            }
        };
        match Vad::new(&graph, &weights, sample_rate.max(0) as u32) {
            Ok(inner) => Box::into_raw(Box::new(VadHandle { inner })),
            Err(e) => {
                ffi_log(&format!("vad_create_c: {e}"));
                std::ptr::null_mut()
            }
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `h`는 vad_create_c가 만든 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vad_free_c(h: *mut VadHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// 상태 초기화 (듣기를 새로 시작할 때).
///
/// # Safety
/// `h`는 유효한 VadHandle 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vad_reset_c(h: *mut VadHandle) {
    if let Some(handle) = unsafe { h.as_mut() } {
        handle.inner.reset();
    }
}

/// mono f32 [-1,1] `len` 샘플(≤ VAD_MAX_FRAME)을 넣는다 — 이번 호출에서 끝난 청크
/// (32ms@16k)들의 음성 확률 평균. 끝난 청크가 없거나 실패면 −1.
///
/// # Safety
/// input은 len 샘플의 유효 버퍼여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vad_process(
    h: *mut VadHandle,
    input: *const c_float,
    len: usize,
) -> c_float {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if h.is_null() || input.is_null() {
            return -1.0;
        }
        let handle = unsafe { &mut *h };
        let n = len.min(VAD_MAX_FRAME);
        match handle
            .inner
            .process(unsafe { slice::from_raw_parts(input, n) })
        {
            Ok(p) => p.unwrap_or(-1.0),
            Err(err) => {
                ffi_log(&format!("vad_process: {err}"));
                -1.0
            }
        }
    }));
    result.unwrap_or(-1.0)
}
