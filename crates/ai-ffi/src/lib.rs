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
//!   C 타입 변환, fs 조달(모델/GLB 경로), YUV 왕복(yuv.rs), panic 방벽뿐.
//!
//! render_mask 계약(모바일): I420 in-place, stride 보존, width/height 음수 허용
//! (|절대값| 사용 — INTEGRATION.md §계약). passthrough(효과 전무)면 변환조차
//! 하지 않고 즉시 Success — analyzer-only(손/집중도만)면 세그·합성을 건너뛰고
//! 태스크만 돈다 (프레임 무수정).
//!
//! ⚠ YUV↔RGB CPU 왕복은 임시 — GPU 상주 변환이 모바일 필수 카드 (yuv.rs 헤더).
//! ⚠ JNI 래퍼(Android)는 모바일 실연결 때 이 C 표면을 감싼다 (로드맵 E).

pub mod yuv;

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

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::audio::Enhancer;
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
    tex: wgpu::Texture,
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
}

static STATE: Mutex<Option<Ffi>> = Mutex::new(None);

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
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: {e}"));
                return VbResult::Failure;
            }
        };
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
            let path = format!("{dir}/{kind}.glb");
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
        let mut guard = STATE.lock().unwrap();
        let st = match ensure_state(&mut guard) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("update_effects_config: {e}"));
                return VbResult::Failure;
            }
        };
        match st.director.apply_json(&json) {
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

        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else {
            ffi_log("render_mask: 미초기화 (set_video_stream_info 먼저)");
            return VbResult::Failure;
        };
        // 완전 무가공 — 변환조차 하지 않는다 (최대 절약)
        if st.director.passthrough() {
            return VbResult::Success;
        }
        let t_ms = st.epoch.elapsed().as_secs_f64() * 1e3;
        st.rgba.resize(w * h * 4, 0);
        yuv::i420_to_rgba(yb, ub, vb, w, h, sy, su, sv, &mut st.rgba);

        // 프레임 텍스처 채우기 (비디오 경로면 세그 ensure, 아니면 세션리스)
        let Ffi { ctx, director, rgba, rgb, target, .. } = st;
        let upload = pollster::block_on(director.with_frame(ctx, w as u32, h as u32, |tex| {
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
        }));
        if let Err(e) = upload {
            ffi_log(&format!("render_mask: 프레임 업로드: {e}"));
            return VbResult::Failure;
        }

        // 태스크용 u8 RGB — 필요한 틱에만 (알파 제거)
        let rgb_opt = if director.wants_pixels(t_ms) {
            rgb.resize(w * h * 3, 0);
            for i in 0..w * h {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&rgba[i * 4..i * 4 + 3]);
            }
            Some(rgb.as_slice())
        } else {
            None
        };

        let needs = director.needs_render();
        if needs {
            let stale = !matches!(target, Some(t) if t.w == w as u32 && t.h == h as u32);
            if stale {
                let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("ai-ffi-target"),
                    size: wgpu::Extent3d {
                        width: w as u32,
                        height: h as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                *target = Some(Offscreen { tex, view, w: w as u32, h: h as u32 });
            }
        }
        let view = target.as_ref().filter(|_| needs).map(|t| &t.view);
        if let Err(e) = pollster::block_on(director.frame(
            ctx,
            w as u32,
            h as u32,
            view,
            rgb_opt,
            t_ms,
        )) {
            ffi_log(&format!("render_mask: {e}"));
            return VbResult::Failure;
        }
        if needs {
            let t = target.as_ref().unwrap();
            match readback_rgba(ctx, &t.tex, w as u32, h as u32, rgba) {
                Ok(()) => yuv::rgba_to_i420(rgba, w, h, yb, ub, vb, sy, su, sv),
                Err(e) => {
                    ffi_log(&format!("render_mask: 리드백: {e}"));
                    return VbResult::Failure;
                }
            }
        }
        // analyzer-only: 프레임 무수정 (원본이 그대로 나간다 — vcxrust 고속경로 등가)
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 타깃 텍스처 → RGBA (256 정렬 패딩 행 제거)
fn readback_rgba(
    ctx: &GpuContext,
    tex: &wgpu::Texture,
    w: u32,
    h: u32,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let bpr = (w * 4).next_multiple_of(256);
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ai-ffi-staging"),
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let bytes = pollster::block_on(ai_gpu::readback::read_buffers(ctx, &[&staging]))?
        .remove(0);
    out.resize((w * h * 4) as usize, 0);
    for row in 0..h as usize {
        let src = row * bpr as usize;
        let dst = row * (w * 4) as usize;
        out[dst..dst + (w * 4) as usize]
            .copy_from_slice(&bytes[src..src + (w * 4) as usize]);
    }
    Ok(())
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
