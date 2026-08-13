# v-ai 통합 설계 (2026-08-13 정찰 기반)

정찰 범위: v-ai 전체(모놀리스 3741줄 정독 + face/gaze/hand/배관/오디오) + vcxrust_ai 전체.
상세 지도는 메모리 `vai-runtime-map`, 근거 file:line은 거기에 있다.

## 0. 정찰이 바꾼 것 — 문제의 정의

정찰 전 가정: "웹과 모바일에 **다른** 구현 두 벌이 있다."
실제: **같은 알고리즘의 TS판/Rust판 두 벌**이다. vcxrust_ai는 주석에 "웹 이식"이
명시된 문자 그대로의 포팅이고, 후처리 상수(JBF/refine/coverage/spill/EMA/프레이밍)
까지 동일하다. 그리고 두 벌 다 "추론 + 후처리 12스테이지 + 합성"이 한 덩어리다.

따라서 **"추론만 ai_engine으로 교체"는 문제를 못 푼다** — 이원화(파이프라인
로직 두 벌)와 모바일의 구조적 성능 문제(ncnn↔wgpu 컨텍스트 분리로 프레임당
CPU↔GPU 왕복 2회 + 동기 블로킹)가 그대로 남는다. ai_engine이 **비디오
파이프라인 전체의 코어**가 되어야 두 문제가 같이 풀린다.

## 1. 핵심 설계 결정

### D1. ai-tasks에 파이프라인 층 신설 — 마스크 소비 스택 전체를 WGSL 한 벌로

세그 마스크의 소비 스택(softmax+시간EMA → joint bilateral 업샘플 → 엣지 정제 →
배경 3모드 합성 + 조명 + 프레이밍 + spill/edge + 밝기/흑백 + 터치업/메이크업
오버레이)이 웹 GLSL, 모바일 WGSL로 두 벌이다. **모바일 것이 이미 WGSL**
(video_kernels.wgsl 772줄, 12 compute pass)이므로 ai-tasks로 이관하는 비용이
낮고, 상수가 양쪽에서 문서화돼 있어 "어느 쪽이 정답" 논쟁이 없다.

```
ai-tasks 층 구조 (확정안)
├─ Session층 (있음)      GpuSession / CpuSession — 모델 인스턴스
├─ Task층 (진행 중)      FaceTask ✓ / SegmentTask / HandTask / GazeTask
│                        = 추론 + 전·후처리 + 트래킹 (모델 여러 개 묶음)
├─ Pipeline층 (신설)     VideoPipeline = 마스크 소비 스택 WGSL + 상태(EMA history,
│                        프레이밍) + TaskParams. Compositor를 여기로 흡수·확장
└─ 순수 로직층 (신설·이관) framing 스무딩(데드밴드+2s커밋+활강), Horn 피팅,
                         제스처 판정, 터치업/메이크업 기하, OneEuro ✓, focus 상태머신
                         — 전부 순수 Rust: 웹(wasm)·모바일(ffi)이 같은 코드
```

경계 밖 유지: three.js GLB 렌더(웹), 2D 캔버스 아바타(웹). 단 모바일 items3d가
이미 wgpu PBR 자체구현이므로 장기적으로 Pipeline층 편입 후보(지금 결정 안 함).

### D2. 연결 seam은 이미 있다 — 새 배관 설계 금지

- **웹**: ai_engine의 WebGL2 엔진이 이미 세그 티어 0으로 프로덕션 가동 중(바이트
  동일 사본). 정식 연결 = `VbEngine` 3함수(`configCustomVideoStream` /
  `destroyCustomVideoStream` / `processWorkerFrame(bitmap, timeSec) →
  {bitmap, passthrough}`)를 ai_engine(WebGPU)으로 구현해 4번째 엔진으로 꽂는 것.
  프레임 반입/반출 배관(worker-stream.ts)은 엔진 불가지론 — **무수정**.
  티어: **ai_engine(WebGPU) → gl-rvm(WebGL2, 기존) → lite(2D)** — WebGPU 없는
  저사양은 기존 gl-rvm이 그대로 받친다.
- **모바일**: ai-ffi가 vcxrust_ai의 기존 표면(`renderMask` YUV420 in-place +
  `updateVideoConfig` + `updateEffectsConfig` JSON + 폴링 2종)을 **그대로 구현**
  → react-native-webrtc 무수정 교체. YUV↔엔진 변환은 GPU 커널로(현행 CPU 전처리
  보다 이득). ncnn 제거 = 왕복 2회+poll(Wait) 소멸이 모바일 최대 성능 카드.
- **오디오**: 웹 audio.worker.ts의 `Engine` 클래스({block, delay, reset,
  processBlock}) 4멤버만 구현하면 교체 끝. 모바일 rustfft STFT는 이미 Rust —
  ai-tasks로 이관해 웹·모바일 공유(#6 선행 자산).

### D3. 설정 프로토콜 = EffectsPatch 규약 채택

vcxrust_ai의 `Option<Option<T>>` JSON 머지(없음=유지 / null=해제 / 값=설정)가
이미 검증된 확장 채널이다. ai-tasks TaskParams를 이 규약으로 정의하고 웹
VBOptions도 여기로 수렴시킨다 (옵션 스키마 3곳 동시 수정 문제도 1곳으로).

### D4. 모델 통일 (품질 비대칭 해소가 통합의 사용자 가치)

| 태스크 | 통일 모델 | 지금의 비대칭 |
|---|---|---|
| 얼굴 | face_landmarker.task 478pt (+blendshape) | 모바일은 FaceMesh 468, blendshape 없음 → 아바타 미지원 |
| 게이즈 | MobileOne-S0 448² (gaze.sw 있음) | **모바일은 CNN 자체가 없음** — 랜드마크 기하 프록시 |
| 손 | hand_landmarker.task 21pt+handedness | 모바일 handedness 없음 → clap 판정이 아예 다른 로직 |
| 세그 | R11/mnv4 계열 (진행 중) | 웹 R18/R11/RVM vs 모바일 v679 |
| 오디오 | fastenhancer 16k/48k (동일) | 런타임만 상이 (ORT vs ncnn) |

추가 이득(웹): face-effects와 focus-tracker가 같은 task 파일을 **두 인스턴스로
이중 로드** 중 — vision 워커의 얼굴 lm 1회 추론 공유로 통합. 기능별 제각각인
delegate 정책(GPU/CPU)도 워커 토폴로지 + p90 강등 사다리 하나로 수렴.

### D5. 신규 갭 과제 (정찰로 확정)

1. **blendshape 모델 개통** — 아바타(eyeBlink/jawOpen 필수)와 blink 판정 소비.
   face_blendshapes.tflite 자산 있음(§3.5에서 "잡op 많아 별도 판단"이었으나
   소비처 확인으로 **필요 확정**). facialTransformationMatrix는 불필요(웹 3D가
   Horn 자체 피팅 — matrix는 depth 힌트일 뿐, 동공간격 폴백 있음).
2. **GazeTask 크롭 규약 주의** — FaceTask의 MediaPipe 회전 크롭이 아니라 웹
   focus-tracker 규약(478 bbox+margin 0.18/0.22, **비회전, 종횡비 무시** 448²
   리사이즈)과 파리티. L2CS가 그 크롭으로 학습됨.
3. **Mali-G57 축소 셰이더 분기** 이관 (대형 WGSL이 드라이버 SIGSEGV).
4. 모바일 clap 분기(1↔2손 전이)는 통일 모델(handedness) 확보 후 재평가.

## 1.5 백엔드 정책 — GPU/CPU를 어떻게 섞나 (2026-08-13 판단)

**핵심: 축이 2개다 — 추론 백엔드 × 합성 백엔드. 이 둘은 독립이고, 섞는 단위는
모델/스테이지 단위다 (op 단위 분할 금지 — 경계 리드백이 이득을 압도, 기확정).**

현 제품의 "GPU 안 되면 효과 전부 끔"(lite 티어)은 이분법이 아니라 구현 단순화다.
마스크는 작다(256×144 u8 ≈ 37KB) — **추론이 CPU로 떨어져도 합성 GPU가 살아있으면
효과 대부분이 산다.** 반대 방향(추론 GPU + 합성 CPU)은 프레임 해상도 리드백이라
금지.

| 티어 | 추론 | 합성 | 효과 | 진입 조건 |
|---|---|---|---|---|
| **A** | GPU(wgpu) | GPU(**같은 디바이스**, zero-copy) | 전부 | WebGPU/네이티브 wgpu 정상 |
| **B** | CPU(ai-cpu) | GPU | 전부 (게이즈만 N프레임 페이싱) | 추론이 GPU에서 밀림(세그 p90) 또는 GPU 약함 — 마스크 업로드 37KB/프레임뿐 |
| **C** | CPU | Canvas2D(웹 lite) / 소프트 합성 | 배경·블러·밝기·흑백·미러·회전·**프레이밍**(2D transform) | 합성 GPU 불가(소프트웨어 래스터 등) |

- 웹 사다리: ai_engine(WebGPU, A/B) → **기존 webgl2 파이프라인(gl-rvm) 유지** → lite(C).
  WebGPU 없는 저사양은 검증된 WebGL2 자산이 그대로 받친다 — 버리지 않는다.
- 모바일: wgpu(Vulkan/Metal)는 거의 항상 가용 → A/B가 기본. Mali-G57류는 축소
  셰이더 분기. **현 모바일은 CPU 폴백이 아예 없음(GPU 실패=전기능 죽음)** —
  ai-cpu(NEON)가 B/C 티어를 신규 제공하는 것 자체가 개선.
- 합성(마스크 소비 스택)은 프레임 해상도 이미지 처리라 **사실상 GPU 전용**으로
  설계한다(WGSL 12패스). C 티어의 윤곽은 canvas blur 근사(현 lite 방식)로 대체.
- 강등 신호는 기확정 규약 유지: 마스크 사이클 p90 > 66ms 2윈도우 연속, 승격 없음.

**효과별 최소 티어**: 배경/블러/밝기/흑백/미러/회전/프레이밍=C까지 전부 ·
윤곽 정제(JBF+refine)/조명/spill=B 이상 · 터치업/메이크업/3D/아바타=B 이상+얼굴 lm ·
게이즈=B에서 페이싱.

**배경 윤곽(엣지) 처리 판단**: 저해상도 추론 마스크 + 프레임 해상도 **joint
bilateral 가이드 업샘플 + 엣지 정제** 스택(현 웹·모바일 공통 방식)이 정답 —
마스크 해상도를 올리는 것보다 싸고 품질이 좋다. 시간 EMA(동적 α)는 마스크
해상도에서 수행하므로 어느 티어든 유지.

## 1.6 아바타(사람→아바타 교체, 표정 포함) 대비 설계

사용자 로드맵: 얼굴 위 2D 아바타(현행)를 넘어 **인물 전체를 표정 연동 아바타로
교체**. 엔진에 요구되는 것:

1. **Expression Stream API** — FaceTask 출력을 `{points 478, blendshapes 52,
   pose(quat·t·scale — Horn 피팅), presence}`로 확장. 어떤 렌더러(three.js/wgpu/
   네이티브)든 이 스트림만 소비하면 표정 리타게팅이 된다. **blendshape 모델
   개통과 Horn 피팅의 ai-tasks 이관이 아바타의 전제조건** — 3D 아이템과 공유.
2. **Pipeline에 person-replacement 합성 모드** — 현 합성은 `mix(bg, person, mask)`
   하나뿐. 아바타 모드는 인물 레이어를 **제거**(배경으로 채움)하고 아바타 레이어를
   얹는다. 합성 모드를 enum(composite/replace)으로 설계해 두면 스테이지 추가 없이
   열린다.
3. 아바타 렌더 자체(리깅+모프타깃)는 경계 밖에서 시작(웹 three.js) — 단 모바일
   items3d가 이미 wgpu PBR이라, 리깅/모프를 얹으면 Pipeline층 내장 렌더러로
   승격 가능(웹·모바일 한 벌). 이건 expression stream이 안정된 뒤 결정.

## 2. 지켜야 할 계약 (게이트 체크리스트)

- mirror/degree는 **랜드마크 추론 전** 적용 — 좌표계 == 출력 화면 (아니면 이펙트 전부 반전).
- brightness/grayscale는 **배경에만** 적용.
- 프레이밍 활성 시 이펙트 좌표에 크롭 transform 보정.
- `processWorkerFrame`: 1프레임 직렬화 전제, passthrough=입력 비트맵 제로카피 반환.
- `destroyCustomVideoStream`은 **모델을 파기하지 않는다** (웜 워커 재사용 — 카메라 토글 지연).
- 시간 EMA(diff>0.3 ? 0.9 : 0.03) — RVM류(pha 직출)는 0.6/0.9.
- 프레이밍 스무딩: 데드밴드 0.045/0.055 + 2s 지속이탈 커밋 + EMA 0.5s + 슬루 0.35/s.
- 모바일 renderMask: in-place, stride 보존, width/height 음수 허용.
- 프로세스당 1인스턴스 제약(모바일)은 다중 스트림 로드맵 확정 전까지 유지 가능.

## 3. 실행 계획 — 데모 HTML 주도 (사용자 제안 채택, 2026-08-13)

방법론: **web/demo/studio.html 하나를 제품 UI 미니어처로 두고, 단계마다 효과를
하나씩 켜면서 개발한다.** 옵션 패널(배경 이미지/색·블러·밝기·흑백·미러·회전·
조명 2개·프레이밍·터치업·메이크업·아바타 토글) + 카메라 + 티어 표시 + 프레임타임
HUD. 각 단계는 헤드리스 게이트(픽셀 diff 또는 좌표 diff)와 눈검증을 같이 간다 —
face-ab.html에서 검증된 방식.

**P1. VideoPipeline 코어** — vcxrust_ai `video_kernels.wgsl`(이미 WGSL) 이관:
시간EMA → JBF → 마스크 blur+refine → 합성 3모드 → 조명 → 프레이밍(GPU bbox
리덕션+순수 Rust 스무딩) → spill/edge → 밝기/흑백. TaskParams=EffectsPatch 규약.
GPU 상주 전처리(리사이즈/정규화 커널)로 프레임당 CPU 픽셀 0.
게이트: 같은 프레임+마스크를 현 웹 파이프라인과 픽셀 diff. studio.html 초판
(배경/블러/밝기/조명/프레이밍까지).

**P2. 티어 정책 (§1.5 매트릭스)** — A/B 티어 전환 구현(ai-cpu 추론+마스크 업로드
경로), p90 강등, 효과별 최소 티어 게이팅. studio.html에 티어 강제 토글 + 강등
시뮬레이션.

**P3. 얼굴 스택** — blendshape 모델 개통(변환기 op 게이트) → FaceTask 확장
(Expression Stream: points+blendshapes+pose), Horn 피팅 이관, 터치업/메이크업
기하 이관(vcxrust_ai Rust 코드 기반 — 이미 Rust) → Pipeline 오버레이.
게이트: blendshape는 MediaPipe 대비 계수 diff, 메이크업/터치업은 픽셀 diff.
studio.html에 터치업/메이크업/lm·pose 시각화.

**P4. 웹 연결** — VbEngine 3함수(ai-wasm) 구현, v-ai 4번째 티어로 A/B
(processWorkerFrame 계약: 1프레임 직렬화·passthrough 제로카피·destroy≠모델 파기).

**P5. 아바타 스파이크** — person-replacement 합성 모드 + expression stream으로
현행 2D 아바타 호환 구동 → 리깅 3D는 스트림 안정 후.

**P6+. GazeTask/HandTask → ai-ffi(ncnn 제거) → 오디오(#6)** — 기존 순서 유지.
