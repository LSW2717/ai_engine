//! 안드로이드 JNI 레이어 — Kotlin `com.cloudwebrtc.webrtc.AiEngine`(가칭)이
//! `System.loadLibrary("ai_ffi")`로 소비한다. **로직 없음**: 전부 이 크레이트의
//! C 표면을 그대로 감싼다 (vcxrust_ai의 c_api/java_api 분리와 같은 구조 —
//! 안드로이드에선 C ABI 대신 이 심볼들만 쓴다).
//!
//! Kotlin 시그니처 (external fun — 클래스/패키지명은 실연결 때 앱에 맞춰 심볼만
//! 리네임하면 된다):
//!   setVideoStreamInfo(path: String): Int
//!   setFaceModelInfo(det: String, lm: String): Int
//!   setGazeModelInfo(gaze: String, bs: String?): Int
//!   setHandModelInfo(det: String, lm: String): Int
//!   setItemModelDir(dir: String): Int
//!   updateEffectsConfig(json: String): Int
//!   setBackgroundImage(rgba: ByteBuffer/*direct*/, w: Int, h: Int): Int
//!   setFocusLayout(json: String): Int
//!   renderMask(y,u,v: ByteBuffer/*direct*/, w,h, sy,su,sv: Int): Int
//!   getFocusState(): String?
//!   pollHandGesture(): String?
//!   destroyCustomVideoStream(): Int
//!   feCreate(sampleRate: Int, modelDir: String): Long
//!   feFree(handle: Long) / feFrameLength(handle: Long): Int /
//!   feSampleRate(handle: Long): Int /
//!   feProcessFrame(handle: Long, input: FloatArray): FloatArray
//!   rnCreate(sampleRate: Int): Long
//!   rnFree(handle: Long) / rnFrameLength(handle: Long): Int /
//!   rnIsActive(handle: Long): Boolean /
//!   rnProcessFrame(handle: Long, input: FloatArray, output: FloatArray)
//!   vadCreate(sampleRate: Int, modelDir: String): Long
//!   vadFree(handle: Long) / vadReset(handle: Long) /
//!   vadProcess(handle: Long, input: FloatArray, length: Int): Float (음성 확률, 청크 미완 −1)
//!
//! 반환 Int는 VbResult (0=Success, -1=Failure). JNI 문자열 반환은 JVM 소유 —
//! vcx_string_free는 안드로이드에서 쓰지 않는다.

use std::ffi::CString;

use jni::objects::{JByteBuffer, JClass, JFloatArray, JString};
use jni::sys::{jboolean, jfloat, jfloatArray, jint, jlong, jstring};
use jni::JNIEnv;

use crate::VbResult;

/// logcat 로거 — 라이브러리 로드 시 1회. 없으면 log:: 매크로(ffi_log 포함)가 전부
/// no-op 이라 GPU 초기화/모델 로드 실패 원인을 기기에서 볼 방법이 없다 (프로브가
/// ok=false 만 남기고 이유는 사라졌던 실사고 2026-08-21). 태그 "ai-ffi".
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(
    _vm: *mut std::ffi::c_void,
    _reserved: *mut std::ffi::c_void,
) -> jint {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("ai-ffi"),
    );
    log::info!("ai-ffi loaded");
    jni::sys::JNI_VERSION_1_6
}

fn jstr(env: &mut JNIEnv, s: &JString) -> Option<String> {
    env.get_string(s).ok().map(|v| v.into())
}

/// String 경로 인자를 CString으로 — 실패는 Failure
macro_rules! cpath {
    ($env:expr, $s:expr) => {
        match jstr($env, &$s).and_then(|v| CString::new(v).ok()) {
            Some(c) => c,
            None => return VbResult::Failure as jint,
        }
    };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setRenderTier(
    _env: JNIEnv,
    _cls: JClass,
    tier: jint,
) -> jint {
    crate::set_render_tier(tier) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setVideoStreamInfo(
    mut env: JNIEnv,
    _cls: JClass,
    path: JString,
) -> jint {
    let Some(p) = jstr(&mut env, &path) else { return VbResult::Failure as jint };
    (unsafe { crate::set_video_stream_info(p.as_ptr(), p.len()) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setFaceModelInfo(
    mut env: JNIEnv,
    _cls: JClass,
    det: JString,
    lm: JString,
) -> jint {
    let (d, l) = (cpath!(&mut env, det), cpath!(&mut env, lm));
    (unsafe { crate::set_face_model_info(d.as_ptr(), l.as_ptr()) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setGazeModelInfo(
    mut env: JNIEnv,
    _cls: JClass,
    gaze: JString,
    bs: JString,
) -> jint {
    let g = cpath!(&mut env, gaze);
    // bs는 nullable — Kotlin String?이 null이면 JString이 null 객체
    let bs_c = if bs.is_null() { None } else { jstr(&mut env, &bs).and_then(|v| CString::new(v).ok()) };
    let bs_ptr = bs_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
    (unsafe { crate::set_gaze_model_info(g.as_ptr(), bs_ptr) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setHandModelInfo(
    mut env: JNIEnv,
    _cls: JClass,
    det: JString,
    lm: JString,
) -> jint {
    let (d, l) = (cpath!(&mut env, det), cpath!(&mut env, lm));
    (unsafe { crate::set_hand_model_info(d.as_ptr(), l.as_ptr()) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setItemModelDir(
    mut env: JNIEnv,
    _cls: JClass,
    dir: JString,
) -> jint {
    let d = cpath!(&mut env, dir);
    (unsafe { crate::set_item_model_dir(d.as_ptr()) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_updateEffectsConfig(
    mut env: JNIEnv,
    _cls: JClass,
    json: JString,
) -> jint {
    let j = cpath!(&mut env, json);
    (unsafe { crate::update_effects_config(j.as_ptr()) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setBackgroundImage(
    env: JNIEnv,
    _cls: JClass,
    rgba: JByteBuffer,
    w: jint,
    h: jint,
) -> jint {
    let Ok(ptr) = env.get_direct_buffer_address(&rgba) else {
        return VbResult::Failure as jint;
    };
    (unsafe { crate::set_background_image(ptr, w, h) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_setFocusLayout(
    mut env: JNIEnv,
    _cls: JClass,
    json: JString,
) -> jint {
    let j = cpath!(&mut env, json);
    (unsafe { crate::set_focus_layout(j.as_ptr()) }) as jint
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_renderMask(
    env: JNIEnv,
    _cls: JClass,
    y: JByteBuffer,
    u: JByteBuffer,
    v: JByteBuffer,
    w: jint,
    h: jint,
    sy: jint,
    su: jint,
    sv: jint,
) -> jint {
    let (Ok(yp), Ok(up), Ok(vp)) = (
        env.get_direct_buffer_address(&y),
        env.get_direct_buffer_address(&u),
        env.get_direct_buffer_address(&v),
    ) else {
        return VbResult::Failure as jint;
    };
    (unsafe { crate::render_mask(yp, up, vp, w, h, sy, su, sv) }) as jint
}

/// render_mask_i420_io — 입력 I420(카메라 회전 포함) → 세운 출력 I420. 반환 = FrameResult 값
/// (0 Written · 1 Unchanged · 2 Fallback · -1 Failure).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_renderMaskI420Io(
    env: JNIEnv,
    _cls: JClass,
    y_in: JByteBuffer,
    u_in: JByteBuffer,
    v_in: JByteBuffer,
    w: jint,
    h: jint,
    sy_in: jint,
    su_in: jint,
    sv_in: jint,
    rotation: jint,
    y_out: JByteBuffer,
    u_out: JByteBuffer,
    v_out: JByteBuffer,
    sy_out: jint,
    su_out: jint,
    sv_out: jint,
) -> jint {
    let (Ok(yi), Ok(ui), Ok(vi), Ok(yo), Ok(uo), Ok(vo)) = (
        env.get_direct_buffer_address(&y_in),
        env.get_direct_buffer_address(&u_in),
        env.get_direct_buffer_address(&v_in),
        env.get_direct_buffer_address(&y_out),
        env.get_direct_buffer_address(&u_out),
        env.get_direct_buffer_address(&v_out),
    ) else {
        return crate::FrameResult::Failure as jint;
    };
    (unsafe {
        crate::render_mask_i420_io(
            yi, ui, vi, w, h, sy_in, su_in, sv_in, rotation, yo, uo, vo, sy_out, su_out, sv_out,
        )
    }) as jint
}

/// C 문자열 반환을 JVM 문자열로 옮기고 즉시 해제
fn own_cstring_to_jstring(env: &JNIEnv, ptr: *mut std::os::raw::c_char) -> jstring {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    unsafe { crate::vcx_string_free(ptr) };
    env.new_string(s).map(|j| j.into_raw()).unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_getFocusState(
    env: JNIEnv,
    _cls: JClass,
) -> jstring {
    own_cstring_to_jstring(&env, crate::get_focus_state())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_pollHandGesture(
    env: JNIEnv,
    _cls: JClass,
) -> jstring {
    own_cstring_to_jstring(&env, crate::poll_hand_gesture())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_destroyCustomVideoStream(
    _env: JNIEnv,
    _cls: JClass,
) -> jint {
    crate::destroy_custom_video_stream() as jint
}

// ───────── 오디오 ─────────

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_feCreate(
    mut env: JNIEnv,
    _cls: JClass,
    sample_rate: jint,
    model_dir: JString,
) -> jlong {
    let Some(d) = jstr(&mut env, &model_dir).and_then(|v| CString::new(v).ok()) else {
        return 0;
    };
    (unsafe { crate::fe_create_c(sample_rate, d.as_ptr()) }) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_feFree(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    unsafe { crate::fe_free_c(handle as *mut crate::FeHandle) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_feFrameLength(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jint {
    (unsafe { crate::fe_get_in_frame_len(handle as *const crate::FeHandle) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_feSampleRate(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jint {
    unsafe { crate::fe_get_sample_rate(handle as *const crate::FeHandle) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_feProcessFrame(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    input: JFloatArray,
) -> jfloatArray {
    let h = handle as *mut crate::FeHandle;
    let n = unsafe { crate::fe_get_in_frame_len(h) };
    if n == 0 {
        return std::ptr::null_mut();
    }
    let mut inp = vec![0f32; n];
    if env.get_float_array_region(&input, 0, &mut inp).is_err() {
        return std::ptr::null_mut();
    }
    let mut out = vec![0f32; n];
    unsafe { crate::fe_process_frame(h, inp.as_ptr(), out.as_mut_ptr()) };
    let Ok(arr) = env.new_float_array(n as i32) else {
        return std::ptr::null_mut();
    };
    if env.set_float_array_region(&arr, 0, &out).is_err() {
        return std::ptr::null_mut();
    }
    arr.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_rnCreate(
    _env: JNIEnv,
    _cls: JClass,
    sample_rate: jint,
) -> jlong {
    crate::rn_create_c(sample_rate) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_rnFree(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    unsafe { crate::rn_free_c(handle as *mut crate::RnHandle) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_rnFrameLength(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jint {
    (unsafe { crate::rn_get_in_frame_len(handle as *const crate::RnHandle) }) as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_rnIsActive(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jboolean {
    (unsafe { crate::rn_is_active(handle as *const crate::RnHandle) } != 0) as jboolean
}

/// 출력은 호출측 배열에 기록 — 10ms마다 새 배열을 만들지 않는다 (feProcessFrame과 다른 점).
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_rnProcessFrame(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    input: JFloatArray,
    output: JFloatArray,
) {
    let h = handle as *mut crate::RnHandle;
    if h.is_null() {
        return;
    }
    let mut inp = [0f32; 480];
    if env.get_float_array_region(&input, 0, &mut inp).is_err() {
        return;
    }
    let mut out = [0f32; 480];
    unsafe { crate::rn_process_frame(h, inp.as_ptr(), out.as_mut_ptr()) };
    let _ = env.set_float_array_region(&output, 0, &out);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_vadCreate(
    mut env: JNIEnv,
    _cls: JClass,
    sample_rate: jint,
    model_dir: JString,
) -> jlong {
    let Some(d) = jstr(&mut env, &model_dir).and_then(|v| CString::new(v).ok()) else {
        return 0;
    };
    (unsafe { crate::vad_create_c(sample_rate, d.as_ptr()) }) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_vadFree(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    unsafe { crate::vad_free_c(handle as *mut crate::VadHandle) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_vadReset(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    unsafe { crate::vad_reset_c(handle as *mut crate::VadHandle) }
}

/// 입력은 스택 버퍼로 받는다 — 오디오 스레드에서 10ms마다 불린다.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_cloudwebrtc_webrtc_AiEngine_vadProcess(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    input: JFloatArray,
    length: jint,
) -> jfloat {
    let h = handle as *mut crate::VadHandle;
    if h.is_null() || length <= 0 {
        return -1.0;
    }
    let n = (length as usize).min(crate::VAD_MAX_FRAME);
    let mut buf = [0f32; crate::VAD_MAX_FRAME];
    if env
        .get_float_array_region(&input, 0, &mut buf[..n])
        .is_err()
    {
        return -1.0;
    }
    unsafe { crate::vad_process(h, buf.as_ptr(), n) }
}
