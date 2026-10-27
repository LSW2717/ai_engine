#ifndef AI_ENGINE_FFI_H
#define AI_ENGINE_FFI_H

/* 자동 생성 — make ffi-header. 손으로 고치지 말 것. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * 한 번에 넘길 수 있는 최대 샘플 수 (40ms@48k) — 오디오 스레드에서 스택 버퍼로 받는다.
 */
#define VAD_MAX_FRAME 1920

/**
 * render_mask_nv12 · render_mask_i420_io 결과.
 */
typedef enum FrameResult {
  /**
   * 출력 평면에 합성 결과를 썼다
   */
  FRAME_RESULT_WRITTEN = 0,
  /**
   * 프레임 무가공(passthrough · analyzer-only) — 출력 평면은 건드리지 않았다, 원본을 그대로 내보낸다
   */
  FRAME_RESULT_UNCHANGED = 1,
  /**
   * 이 프레임은 GPU 평면 경로로 못 한다(B/C 티어 · 태스크 픽셀 틱 · 합성 없는 뒤집기 ·
   * 홀수 크기 회전) — 엔진 상태를 건드리지 않았으니 호출자는 같은 프레임을 예전 경로
   * (호스트 회전 + render_mask I420)로 처리한다
   */
  FRAME_RESULT_FALLBACK = 2,
  FRAME_RESULT_FAILURE = -1,
} FrameResult;

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
 * 불투명 핸들 — None = 미지원 레이트 passthrough (프레임 480, 무가공)
 */
typedef struct RnHandle RnHandle;

/**
 * 불투명 핸들 — 음소거 중 발화 감지용 음성 확률.
 */
typedef struct VadHandle VadHandle;

/**
 * 렌더 티어 선언 — 0=A(GPU), 1=B(CPU 추론+GPU 합성), 2=C(GPU 완전 회피).
 * 모델 주입 전에 호출해야 한다. B/C 전환 시 세그 모델(경량 CPU용)을
 * set_video_stream_info로 다시 주입하는 것까지가 호스트 몫.
 */
enum VbResult set_render_tier(int tier);

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
 * NV12 입력(iOS 카메라 CVPixelBuffer 형식, 카메라 회전 `rotation` = 0/90/180/270, 시계 방향)
 * → 세운 NV12 출력 — 호스트의 NV12→I420 변환·회전과 되돌리는 인터리브 복사가 없어진다.
 * 입력 `width×height` 는 읽기만 하고(카메라 버퍼) 결과(90/270 이면 `height×width`)는 별도
 * 출력 평면(stride 보존)에 쓴다. 결과 바이트는 같은 프레임을 libyuv 로 세워 I420 으로
 * render_mask 에 넣은 것과 같다. 회전하는 홀수 크기는 Fallback.
 *
 * # Safety
 * y_in/uv_in 은 |stride_in|×|height| · |stride_uv_in|×⌈|height|/2⌉ 크기의 읽기 평면,
 * y_out/uv_out 은 세운 크기 기준 같은 규칙의 쓰기 평면이어야 한다(입력과 겹치지 않을 것).
 */
enum FrameResult render_mask_nv12(const uint8_t *y_in,
                                  const uint8_t *uv_in,
                                  int32_t width,
                                  int32_t height,
                                  int32_t stride_y_in,
                                  int32_t stride_uv_in,
                                  int32_t rotation,
                                  uint8_t *y_out,
                                  uint8_t *uv_out,
                                  int32_t stride_y_out,
                                  int32_t stride_uv_out);

/**
 * I420 입력(카메라 회전 `rotation` = 0/90/180/270, 시계 방향) → 세운 I420 출력 — Android
 * 호스트의 회전(I420Rotate)·출력 버퍼로 옮기는 복사가 없어진다. 입력 `width×height` 는 읽기만
 * 하고, 결과(90/270 이면 `height×width`)는 별도 출력 평면(stride 보존)에 쓴다. 결과 바이트는
 * 같은 프레임을 libyuv 로 세워 render_mask 에 넣은 것과 같다. 회전하는 홀수 크기는 Fallback.
 *
 * # Safety
 * 입력 평면은 |stride|×|height|(크로마 ⌈/2⌉) 크기의 읽기 평면, 출력 평면은 세운 크기 기준
 * 같은 규칙의 쓰기 평면이어야 한다(입력과 겹치지 않을 것).
 */
enum FrameResult render_mask_i420_io(const uint8_t *y_in,
                                     const uint8_t *u_in,
                                     const uint8_t *v_in,
                                     int32_t width,
                                     int32_t height,
                                     int32_t stride_y_in,
                                     int32_t stride_u_in,
                                     int32_t stride_v_in,
                                     int32_t rotation,
                                     uint8_t *y_out,
                                     uint8_t *u_out,
                                     uint8_t *v_out,
                                     int32_t stride_y_out,
                                     int32_t stride_u_out,
                                     int32_t stride_v_out);

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

/**
 * 생성 — sample_rate 48000이면 RNNoise(가중치 내장 — 모델 파일 없음), 그 외 레이트는
 * passthrough 핸들 (프레임 480, 무가공 — 호스트가 레이트 스위치 없이도 안전).
 */
struct RnHandle *rn_create_c(int sample_rate);

/**
 * # Safety
 * `h`는 rn_create_c가 만든 포인터이거나 null이어야 한다.
 */
void rn_free_c(struct RnHandle *h);

/**
 * process_frame 호출당 샘플 수 — 활성·passthrough 모두 480 (10ms@48k)
 *
 * # Safety
 * `h`는 유효한 RnHandle 포인터여야 한다.
 */
uintptr_t rn_get_in_frame_len(const struct RnHandle *h);

/**
 * RNNoise가 실제로 도는 핸들인지 (0 = passthrough)
 *
 * # Safety
 * `h`는 유효한 RnHandle 포인터여야 한다.
 */
int rn_is_active(const struct RnHandle *h);

/**
 * 한 프레임 처리 — input/output은 480 샘플 mono f32 [-1,1] (겹쳐도 안전).
 * passthrough 핸들은 원음 그대로.
 *
 * # Safety
 * input/output은 480 샘플의 유효 버퍼여야 한다.
 */
void rn_process_frame(struct RnHandle *h, const float *input, float *output);

/**
 * 생성 — `{model_dir}/vad16/graph.json` + `weights.bin` 로드. sample_rate = 넣을 PCM 의
 * 레이트(16000 그대로, 48000 등은 내부에서 16k 로 내린다). 실패 시 null.
 *
 * # Safety
 * model_dir은 NUL 종료 C 문자열이어야 한다.
 */
struct VadHandle *vad_create_c(int sample_rate, const char *model_dir);

/**
 * # Safety
 * `h`는 vad_create_c가 만든 포인터이거나 null이어야 한다.
 */
void vad_free_c(struct VadHandle *h);

/**
 * 상태 초기화 (듣기를 새로 시작할 때).
 *
 * # Safety
 * `h`는 유효한 VadHandle 포인터이거나 null이어야 한다.
 */
void vad_reset_c(struct VadHandle *h);

/**
 * mono f32 [-1,1] `len` 샘플(≤ VAD_MAX_FRAME)을 넣는다 — 이번 호출에서 끝난 청크
 * (32ms@16k)들의 음성 확률 평균. 끝난 청크가 없거나 실패면 −1.
 *
 * # Safety
 * input은 len 샘플의 유효 버퍼여야 한다.
 */
float vad_process(struct VadHandle *h,
                  const float *input,
                  uintptr_t len);

#endif  /* AI_ENGINE_FFI_H */
