#ifndef AI_ENGINE_FFI_H
#define AI_ENGINE_FFI_H

/* 자동 생성 — make ffi-header. 손으로 고치지 말 것. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * 0=Success, -1=Failure (vcxrust_ai VbResult와 동일 값 — 기존 호스트의
 * `== -1` 검사가 조용히 깨지지 않게 -1을 쓴다)
 */
typedef enum VbResult {
  VB_RESULT_SUCCESS = 0,
  VB_RESULT_FAILURE = -1,
} VbResult;

/**
 * 불투명 핸들 — None = 미지원 레이트 passthrough (프레임 480, 무가공)
 */
typedef struct FeHandle FeHandle;

/**
 * 세그(.sw) 모델 경로 — vcxrust_ai와 동일 시그니처(ptr+len, NUL 아님·UTF-8)
 *
 * # Safety
 * `model_path_ptr..model_path_len`은 유효한 UTF-8 바이트여야 한다.
 */
enum VbResult set_video_stream_info(const uint8_t *model_path_ptr, uintptr_t model_path_len);

/**
 * 얼굴 검출+랜드마크(.sw) 경로 — 아이템·터치업/메이크업·집중도의 전제
 *
 * # Safety
 * det_path/lm_path는 NUL 종료 C 문자열이어야 한다.
 */
enum VbResult set_face_model_info(const char *det_path, const char *lm_path);

/**
 * 손 검출+랜드마크(.sw) 경로 — handDetection(제스처)의 전제
 *
 * # Safety
 * det_path/lm_path는 NUL 종료 C 문자열이어야 한다.
 */
enum VbResult set_hand_model_info(const char *det_path, const char *lm_path);

/**
 * 게이즈 CNN(.sw) + blendshapes(.sw, **nullable** — 없으면 blink는 EAR 절반만)
 * — focusDetection의 전제
 *
 * # Safety
 * gaze_path는 NUL 종료 C 문자열, bs_path는 같거나 null이어야 한다.
 */
enum VbResult set_gaze_model_info(const char *gaze_path, const char *bs_path);

/**
 * 3D 아이템 GLB 디렉터리 — faceItems가 켜지면 `{dir}/{kind}.glb`를 지연 로드
 *
 * # Safety
 * dir은 NUL 종료 C 문자열이어야 한다.
 */
enum VbResult set_item_model_dir(const char *dir);

/**
 * 단일 JSON 설정 — EffectsPatch + faceItems/handDetection/focusDetection.
 * 머지 규약: 없음=유지 / null=해제 / 값=설정.
 *
 * # Safety
 * `json`은 유효한 NUL 종료 C 문자열이어야 한다.
 */
enum VbResult update_effects_config(const char *json);

/**
 * 배경 이미지 (RGBA8, len = w*h*4) — EffectsPatch background:"image"가 소비.
 * vcxrust의 base64-인-JSON 대신 바이너리 채널 (인코딩 왕복 제거).
 *
 * # Safety
 * rgba는 w*h*4 바이트의 유효한 버퍼여야 한다.
 */
enum VbResult set_background_image(const uint8_t *rgba, int32_t width, int32_t height);

/**
 * 다중 모니터 레이아웃 JSON ({monitors:[...], targetIndex} | "null") —
 * focusDetection을 켠 뒤 호출 (레이아웃 조달은 호스트 몫)
 *
 * # Safety
 * `json`은 유효한 NUL 종료 C 문자열이어야 한다.
 */
enum VbResult set_focus_layout(const char *json);

/**
 * I420 in-place 처리: passthrough면 무가공, analyzer-only면 태스크만,
 * 아니면 세그+이펙트+아이템 합성 결과를 같은 평면에 되쓴다.
 *
 * # Safety
 * y/u/v는 |stride|×|h|(/2) 크기의 유효한 가변 평면이어야 한다.
 */
enum VbResult render_mask(uint8_t *y,
                          uint8_t *u,
                          uint8_t *v,
                          int32_t width,
                          int32_t height,
                          int32_t stride_y,
                          int32_t stride_u,
                          int32_t stride_v);

/**
 * 마지막 집중도 — {"status":"FOCUSED","attentive":true,"score":100,
 * "monitorIndex":0,"yaw":..,"pitch":..} (7상태 — 우리 FocusResult 그대로).
 * 반환 문자열은 `vcx_string_free`로 해제.
 */
char *get_focus_state(void);

/**
 * 제스처 이벤트 하나 (FIFO 16) — 없으면 null.
 * {"gesture":"clap","confidence":0.973,"handedness":"left","tsMs":123.4}
 * 반환 문자열은 `vcx_string_free`로 해제.
 */
char *poll_hand_gesture(void);

/**
 * 스트림 파기 = **리셋** (vcxrust 규약): GPU 세션·파이프라인 리소스는 반납하되
 * 컨텍스트·모델 바이트는 유지 — 다음 render_mask가 지연 재로드로 즉시 살아난다.
 */
enum VbResult destroy_custom_video_stream(void);

/**
 * C 문자열 해제 (get_focus_state/poll_hand_gesture 반환값)
 *
 * # Safety
 * `s`는 이 라이브러리가 CString::into_raw로 만든 포인터이거나 null이어야 한다.
 */
void vcx_string_free(char *s);

/**
 * 생성 — sample_rate 16000/48000이면 `{model_dir}/fe16|fe48/graph.json`+
 * `weights.bin` 로드, 그 외 레이트는 passthrough 핸들 (프레임 480, 무가공 —
 * 호스트가 레이트 스위치 없이도 안전). 실패 시 null.
 *
 * # Safety
 * model_dir은 NUL 종료 C 문자열이어야 한다.
 */
struct FeHandle *fe_create_c(int sample_rate, const char *model_dir);

/**
 * # Safety
 * `h`는 fe_create_c가 만든 포인터이거나 null이어야 한다.
 */
void fe_free_c(struct FeHandle *h);

/**
 * process_frame 호출당 샘플 수 (hop) — passthrough 핸들은 480
 *
 * # Safety
 * `h`는 유효한 FeHandle 포인터여야 한다.
 */
uintptr_t fe_get_in_frame_len(const struct FeHandle *h);

/**
 * 생성 시 레이트 (passthrough 핸들 판별: 미지원 레이트 그대로 반환)
 *
 * # Safety
 * `h`는 유효한 FeHandle 포인터여야 한다.
 */
int fe_get_sample_rate(const struct FeHandle *h);

/**
 * 한 hop 처리 — input/output은 frame_len 샘플 mono f32 [-1,1] (겹쳐도 안전).
 * 반환은 예약값 0.0 (vcxrust 규약 유지 — 활동도 자리)
 *
 * # Safety
 * input/output은 frame_len 샘플의 유효 버퍼여야 한다.
 */
float fe_process_frame(struct FeHandle *h, const float *input, float *output);

#endif  /* AI_ENGINE_FFI_H */
