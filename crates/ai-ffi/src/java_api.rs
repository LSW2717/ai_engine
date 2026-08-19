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
//!
//! 반환 Int는 VbResult (0=Success, -1=Failure). JNI 문자열 반환은 JVM 소유 —
//! vcx_string_free는 안드로이드에서 쓰지 않는다.

use std::ffi::CString;

use jni::objects::{JByteBuffer, JClass, JFloatArray, JString};
use jni::sys::{jfloatArray, jint, jlong, jstring};
use jni::JNIEnv;

use crate::VbResult;

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
