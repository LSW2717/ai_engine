//! ai-ffi — 모바일 C ABI 바인딩 (vcxrust_ai vcx-segmentation 표면 재현).
//!
//! 목표: 모바일이 **.so/.a 교체 + 모델 파일 교체**만으로 ncnn 스택에서 ai_engine으로
//! 갈아탈 수 있게 함수 이름·시그니처를 기존 C ABI와 동일하게 유지한다
//! (JNI/Swift 브리지 무수정 — INTEGRATION.md §1.6 seam).
//!
//! 규칙 (ARCHITECTURE.md): 이 크레이트에 분기(if)가 생기면 로직이고 ai-tasks로
//! 내려간다. 여기 남는 것: C 타입 변환, YUV 프레임 임포트(yuv.rs), 전역 상태
//! 수명, panic 방벽(catch_unwind — C 경계로 unwind가 새면 UB).
//!
//! 구현 상태 (뼈대 1차, 2026-08-14):
//!   ✅ set_video_stream_info(.sw 경로) / update_effects_config(JSON) /
//!      render_mask(I420 in-place — 추론+이펙트 스택) / destroy / vcx_string_free
//!   ⬜ update_video_config(VideoOptionsC) — 미러/회전 등은 effects JSON으로 우선
//!   ⬜ set_face/hand/item_model_info, get_focus_state, poll_hand_gesture —
//!      태스크(Face/Hand/Gaze/ItemsOverlay)는 ai-tasks에 완비, 배선은 모바일
//!      실연결 때 (로드맵 E)
//! ⚠ YUV↔RGB CPU 왕복은 임시 — GPU 상주 변환이 모바일 필수 카드 (yuv.rs 헤더).

pub mod yuv;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::Mutex;

use ai_gpu::GpuContext;
use ai_tasks::features::vb::GateHarness;
use ai_tasks::GpuSession;

/// vcxrust_ai VbResult와 같은 표현 (0=Success, 1=Failure)
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VbResult {
    Success = 0,
    Failure = 1,
}

struct Ffi {
    ctx: GpuContext,
    seg: GpuSession,
    /// 오프스크린 렌더+리드백 하네스 — frame_infer(실추론 경로)를 쓴다
    harness: GateHarness,
    rgba: Vec<u8>,
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

/// 세그 모델(.sw) 로드 + GPU 컨텍스트/파이프라인 준비.
/// vcxrust_ai와 동일 시그니처 (경로 ptr+len — CStr 아님).
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
        let path = match std::str::from_utf8(path_bytes) {
            Ok(v) => v.to_string(),
            Err(_) => {
                ffi_log("set_video_stream_info: path not utf-8");
                return VbResult::Failure;
            }
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: read {path}: {e}"));
                return VbResult::Failure;
            }
        };
        let ctx = match GpuContext::new_blocking() {
            Ok(c) => c,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: GPU 컨텍스트: {e}"));
                return VbResult::Failure;
            }
        };
        let seg = match pollster::block_on(GpuSession::load(&ctx, &bytes)) {
            Ok(s) => s,
            Err(e) => {
                ffi_log(&format!("set_video_stream_info: 모델 로드: {e}"));
                return VbResult::Failure;
            }
        };
        let harness = GateHarness::new(&ctx);
        *STATE.lock().unwrap() = Some(Ffi { ctx, seg, harness, rgba: Vec::new() });
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// EffectsPatch JSON 적용 (없음=유지 / null=해제 / 값=설정 — 웹 studio_config와
/// 같은 계약, ai_tasks::features::vb::params 참조).
///
/// # Safety
/// `json`은 유효한 NUL 종료 C 문자열이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn update_effects_config(json: *const c_char) -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if json.is_null() {
            ffi_log("update_effects_config: null json");
            return VbResult::Failure;
        }
        let json = unsafe { CStr::from_ptr(json) }.to_string_lossy();
        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else {
            ffi_log("update_effects_config: set_video_stream_info 먼저");
            return VbResult::Failure;
        };
        match st.harness.pipeline.apply_json(&json) {
            Ok(()) => VbResult::Success,
            Err(e) => {
                ffi_log(&format!("update_effects_config: {e}"));
                VbResult::Failure
            }
        }
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// I420 프레임 in-place 처리: 추론(세그) + 이펙트 스택 → 결과를 같은 평면에 되쓴다.
/// vcxrust_ai render_mask와 동일 시그니처.
///
/// # Safety
/// y/u/v는 stride×h(/2) 크기의 유효한 가변 평면이어야 한다.
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
        if y.is_null() || u.is_null() || v.is_null() || width <= 0 || height <= 0 {
            ffi_log("render_mask: bad args");
            return VbResult::Failure;
        }
        let (w, h) = (width as usize, height as usize);
        let (sy, su, sv) = (stride_y as usize, stride_u as usize, stride_v as usize);
        let (yb, ub, vb) = unsafe {
            (
                slice::from_raw_parts_mut(y, sy * h),
                slice::from_raw_parts_mut(u, su * h.div_ceil(2)),
                slice::from_raw_parts_mut(v, sv * h.div_ceil(2)),
            )
        };

        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else {
            ffi_log("render_mask: set_video_stream_info 먼저");
            return VbResult::Failure;
        };
        st.rgba.resize(w * h * 4, 0);
        yuv::i420_to_rgba(yb, ub, vb, w, h, sy, su, sv, &mut st.rgba);
        let out = match pollster::block_on(st.harness.frame_infer(
            &st.ctx,
            &mut st.seg,
            &st.rgba,
            w as u32,
            h as u32,
        )) {
            Ok(o) => o,
            Err(e) => {
                ffi_log(&format!("render_mask: {e}"));
                return VbResult::Failure;
            }
        };
        yuv::rgba_to_i420(&out, w, h, yb, ub, vb, sy, su, sv);
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// 스트림 파기 — 모델·파이프라인·컨텍스트 해제
#[unsafe(no_mangle)]
pub extern "C" fn destroy_custom_video_stream() -> VbResult {
    let result = catch_unwind(AssertUnwindSafe(|| {
        *STATE.lock().unwrap() = None;
        VbResult::Success
    }));
    result.unwrap_or_else(|_| handle_panic())
}

/// C 문자열 해제 (get_focus_state/poll_hand_gesture 반환값용 — 표면 유지)
///
/// # Safety
/// `s`는 이 라이브러리가 CString::into_raw로 만든 포인터이거나 null이어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vcx_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { std::ffi::CString::from_raw(s) });
    }
}
