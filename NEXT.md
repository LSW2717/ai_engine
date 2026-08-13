# 다음 세션 시작점 (2026-08-13 마감)

## 5분 안에 상태 확인

```sh
cargo test --workspace --release        # 46개 스위트, 전부 ok 여야 한다
make build-wasm                         # web/pkg 갱신
node tools/run_web.mjs compare/index.html   # webgl2 vs 우리 (GPU 다른 앱 닫고)
```

기대: ORT 게이트 `fp32 max_err <5e-5 / 가중치f16 6e-4 / 전경 18.5%`,
compare에서 `webgl2 ~1.57ms / ai_engine 가중치fp16 ~2.01ms`.

---

## 확정된 로드맵

| # | 항목 | 상태 |
|---|---|---|
| 0 | `ai-tasks` 크레이트 (공개 API 본체) | **진행 중 — 아래 참조** |
| 1 | 폴백 트리거 3종 | 다음 |
| 2 | `ai-cpu` (SIMD128/NEON + 스레드) + R11 vs tflite-simd A/B | |
| 3 | `Reshape` canon + `PRELU` + `MAX_POOL_2D` | |
| 4 | 랜드마크 스파이크 (face_detector → 좌표 diff) | |
| 5 | 파이프라인 로직 (ROI 트래킹 / 회전 정규화 / OneEuroFilter / Horn 피팅) | |
| 6 | 오디오 (DFT / 1D conv / 복소) | |

성능 추격(webgl2 대비 1.28배)은 **mnv4 기반 RVM으로 교체한 뒤에** 재개하기로 함.

---

## 0. `ai-tasks` — 지금까지 한 것 / 남은 것

**끝난 것**
- 크레이트 생성, 워크스페이스 등록
- `Compositor` (`composite.rs` + `shaders/composite.wgsl`) — 합성 셰이더·업샘플·
  스테이징 텍스처·바인드그룹 캐시가 **플랫폼 무관**으로 이관됨.
  `ai-wasm/present.rs`는 442 → 109줄 (서피스 획득 + 웹 프레임 임포트만).
- `Segmenter` (`segmenter.rs`) — 모델 수명 + 프레임 루프 + **프레임타임 링버퍼(p50/p90)**.
  `ai-wasm`의 `MODEL` thread_local이 이걸 담는다. 전 export 경로가 이걸 통과한다.
- `TaskError` — `NoGpu` / `DeviceLost` / `Runtime` / `Gpu` 구조화 (폴백 판정 근거).
- `model_stats()` wasm export 추가 (p50/p90/last/frames).

**남은 것**
- `Segmenter::process()` 하나로 묶기 — 지금은 `upload → infer → (호스트 합성) → finish_frame`을
  바인딩이 순서대로 부른다. 합성이 플랫폼 서피스에 걸려 있어 아직 안 묶었다.
  `ai-ffi` 만들 때 같은 순서를 또 쓰게 되면 그때 묶는다.
- `ai-ffi` 뼈대 (C ABI, `repr(C)`, opaque handle). 모바일 실연결은 나중.

**지켜야 할 규칙 (ARCHITECTURE.md에도 박아둠)**
> `ai-wasm` / `ai-ffi`에는 분기(`if`)가 없다. 분기가 생겼다면 로직이고 `ai-tasks`로 내려간다.
> 플랫폼마다 진짜 다른 것만 바인딩에 남긴다: 서피스 획득, 프레임 임포트, 스레드 모델, 모델 바이트 조달.

---

## 1. 폴백 트리거 3종 (바로 다음)

지금 있는 것: `is_supported()`, `InitError{NoAdapter, RequestDevice}`.
**없는 것**:

1. **device lost 구독** — `ai-gpu/context.rs`에서 `device.lost` 미구독.
   wasm은 `on_uncaptured_error`조차 미등록(wgpu 30 웹 백엔드 `Error::from_js` panic 회피).
   → `TaskError::DeviceLost`로 올려 호스트가 강등하게.
2. **프레임타임 노출** — `model_stats()`는 만들었다. 남은 건 호스트가 이걸 **쓰게** 하는 것.
3. **v-ai 연결** — 여기가 진짜 구멍이다:
   ```
   vcxreact/packages/v-ai/src/virtual-background/video-worker-webgl2.js
     :444  _recordCycle(dt)        — p90 > 66ms 2윈도우 연속이면 _demoteSegTier
     :488  _recordCycle 호출       — ★ _createOnnxAdapter 경로에만 있다
     :3577 _demoteSegTier('engine init 실패')  — 엔진 티어는 init 실패만
   ```
   즉 **`gl-rvm`(자체 엔진) 티어는 아무리 느려도 강등되지 않는다.**
   `es.render()` 경로에도 `_recordCycle`을 걸어야 한다. ai_engine을 티어로 넣어도 같은 상태.

---

## 2. `ai-cpu` — 측정된 출발점

`CpuExec`(순진 스칼라 f32, 1스레드) 실측, `segm_mnv4s050_s2_160x288`(0.134 GMAC):

> **119.1 ms (1.1 GMAC/s)**

M2 Pro 코어 1개 NEON f32 피크가 ~14 GMAC/s → 지금 SIMD를 전혀 못 쓰고 있다.
현실적 목표: +SIMD128 ~30~40ms → +4스레드 ~10~15ms.
참고선(ncnn, 손튜닝, M2 Pro, RVM 144×256): CPU 4T fp16 12.3ms / fp32 19.6ms.

재현:
```sh
AI_ONNX=<onnx> [AI_REPS=5] cargo test --release -p ai-convert --test bench_cpu -- --ignored --nocapture
```

**ai-cpu의 값어치는 속도가 아니라 의존성 제거다** — v-ai가 지금 싣는 AI 런타임이
**66MB** (ORT 50MB + mediapipe 11MB + tflite 5.2MB). 우리 wasm은 0.65MB.

---

## 3~5. 랜드마크 — 조사 끝, 결론

`.task`는 **zip**이고 MediaPipe가 실제로 쓰는 tflite가 그대로 들어 있다:

| face_landmarker.task | | hand_landmarker.task | |
|---|---|---|---|
| face_detector.tflite | 230KB (128², 896앵커) | hand_detector.tflite | 2.34MB (192², 2016앵커) |
| face_landmarks_detector.tflite | 2.55MB (256², 478pt) | hand_landmarks_detector.tflite | 5.48MB (224², 21pt) |
| face_blendshapes.tflite | 955KB | | |
| geometry_pipeline_metadata_landmarks.binarypb | 19KB | | |

**커스텀 op이 하나도 없다** (전부 표준 빌트인). 우리 엔진 대비 빠진 것:

> **`PRELU` + `MAX_POOL_2D` + `Reshape` canon** — 이 3개면 핵심 4모델이 열린다.
> `FULLY_CONNECTED`는 M=1 GEMM = 이미 있는 Gemv. `MEAN`은 gpool.
> `DEQUANTIZE`(74~251개)는 fp16 가중치 디코드 — 변환기가 오프라인 처리(이미 지원).
> blendshapes만 잡op(transpose/strided_slice/rsqrt) 많아 별도 판단.

임포터 경로: **(A) tf2onnx로 tflite→onnx 스파이크 → 되면 (B) `ai-convert`에 tflite
flatbuffer 임포터**(400~600줄, `.task` 직접 소비 → MediaPipe 버전업이 파일 교체로 끝남).

품질 파리티의 나머지 절반은 **calculator 그래프**다: 이전 프레임 ROI 트래킹(검출은
놓쳤을 때만), 회전 정규화, attention mesh, **OneEuroFilter 스무딩**. ncnn 포팅이
나빴던 원인이 여기일 가능성이 크다. 이건 순수 Rust라 **웹·모바일이 한 벌 공유** —
지금 웹(mediapipe-wasm)/모바일(vcxrust_ai) 두 벌인 걸 합치는 게 통합의 본체다.

검증은 RVM 때와 같은 방식: **MediaPipe wasm과 같은 프레임 먹여 좌표 diff**. 숫자로
파리티가 증명되기 전엔 교체 안 함.

추출본: `unzip <task> -d <dir>` (한 번 뽑아두면 됨).

---

## 6. 오디오

`fastenhancer_b_48k` → `SplitToSequence` 미지원, `_16k` → `DFT` 미지원.
STFT → 네트 → iSTFT 구조라 **1D conv·복소수·시퀀스 = 새 커널 계열**.
그리고 **AudioWorklet에는 WebGPU가 없다** → 오디오는 무조건 CPU.
즉 **`ai-cpu`가 선행 조건**이다.

---

## 경계 결정 (3D 에셋 / 영상처리)

| ai-tasks 안 | 밖 |
|---|---|
| 추론, 전처리, 후처리(softmax·EMA·NMS·앵커·OneEuroFilter), 배경 합성 | glTF 로딩·스키닝·PBR·씬 그래프 (three.js / 네이티브) |
| **밝기·대비·채도·LUT, 블러, 샤픈** — 이미 프레임 텍스처를 만지므로 공짜 | 2D 메이크업 드로잉 (캔버스 2D) |
| **랜드마크 → 3D 유사변환 피팅 (Horn/Procrustes + EMA/slerp)** — 지금 웹에만 있음, 모바일도 필요 | |

⚠ 웹에서 ai_engine은 WebGPU, three.js는 WebGL2 → **텍스처 공유 불가**, 프레임당 복사.
제로카피로 가려면 three.js `WebGPURenderer` + ai_engine이 `GPUDevice`를 JS에 노출해야 하는데,
wgpu 30 웹 백엔드에서 실제 `GPUDevice`를 꺼낼 수 있는지 **미확인**. present 경로를
"캔버스에 직접 그린다"에 고정하지 말고 **출력 텍스처를 돌려줄 수 있는 형태**로 둘 것.

---

## 알려진 잠재 버그 (지금은 안 막음)

`segm_mnv4_w100m64`(R18) 변환 시 **op[5]가 아직 생산되지 않은 tid 6을 읽는다**
(`fuse_concat`이 concat을 conv에 접으면서 소비자를 생산자 앞에 둠).
CPU 실행기는 즉시 죽고 **GPU는 안 죽고 0 초기화 슬롯을 읽어 조용히 틀린다.**
R11·RVM은 안 걸린다. **R18은 안 쓰기로 했으므로 지금은 보류. 단 mnv4-RVM은 같은
concat→conv 패턴이라 걸릴 확률이 높으니 그 작업 시작할 때 같이 잡을 것.**
재현: `cargo run -p ai-convert -- <r18.onnx> -o /dev/null --dump-json`으로 op 순서 확인.

---

## 측정 규율 (반복해서 데인 것)

- **per-op은 `prof_isolated`로.** `profile_infer`는 op마다 컴퓨트 패스를 새로 열어
  Metal에서 ~17µs 가짜 바닥이 붙는다. `Model::bench_op`은 패스 1개에 디스패치 N개.
- **min-of-3.** 단발은 ±20% 흔들린다. 총합·프레임타임은 머신이 바쁘면 신호가 아니다 —
  **정책이 바꾸는 op만 diff**하면 SNR이 훨씬 높다.
- **브라우저 벤치와 네이티브 테스트를 동시에 돌리지 말 것** (같은 GPU를 뺏는다).
- **셰이더 만졌으면 즉시 `cargo test -p ai-gpu --lib`** — naga 검증이 0.2초에 잡는다.
  `writer::fill`은 매칭 안 된 `//@슬롯`을 조용히 삭제해서, 슬롯 추가하고 Rust 공급을
  빼먹으면 `@compute` 줄이 사라지고 **마스크가 전부 0**이 된다 (실제로 당함).
- **커널·정책 바꿨으면 `gate_ort_frame` 필수.** `rvm_e2e`는 엔진 자기 CPU 실행기라
  GPU/CPU가 같이 틀리면 통과한다 (tanh 오버플로가 그렇게 빠져나갔다).
- 데모 눈검증은 사용자가 한다 — `make build-wasm`까지 하고 `ab.html` 확인 요청.
