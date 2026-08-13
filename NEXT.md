# 다음 세션 시작점 (2026-08-13 마감)

## 5분 안에 상태 확인

```sh
cargo test --workspace --release        # 52개 스위트, 전부 ok 여야 한다
make build-wasm                         # web/pkg 갱신 (+simd128, 725KB)
node tools/run_web.mjs compare/index.html   # webgl2 vs 우리 (GPU 다른 앱 닫고)
node tools/run_web.mjs demo/cpu-ab.html --camera   # R11 CPU 3자 A/B: ai-cpu ~6.8ms < tflite 6.9 / ort 11 (§2.5)
node tools/profile_web.mjs 'demo/cpu-ab.html?only=ours' --ops   # wasm 스텝별 예산표
# per-op 예산표 (커널 만졌으면 여기부터):
AI_ONNX=$(pwd)/models/segm_mnv4s050_s2_160x288_nhwc.onnx \
  cargo test --release -p ai-cpu --test prof_cpu -- --ignored --nocapture
# ai-cpu 정확도 (R11) — AI_ONNX는 절대경로 (test CWD가 크레이트 루트):
AI_ONNX=$(pwd)/models/segm_mnv4s050_s2_160x288_nhwc.onnx \
  cargo test --release -p ai-cpu --test oracle_real -- --ignored --nocapture   # max_err ~1.4e-5
```

기대: ORT 게이트 `fp32 max_err <5e-5 / 가중치f16 6e-4 / 전경 18.5%`,
compare에서 `webgl2 ~1.57ms / ai_engine 가중치fp16 ~2.01ms`.

---

## 바로 다음 — 로드맵 #4 (랜드마크 스파이크)

모델 5종은 CPU·GPU 양쪽에서 다 돈다(§3.5) — 하지만 **calculator 그래프 없이는
기능이 아니다** (디텍터 출력은 날 앵커/로짓, 랜드마크는 ROI 크롭 입력 전제).
순서:
1. **다중모델 핸들 API** (ai-wasm/ai-tasks — vision 워커에 det+lm+게이즈 상주)
2. **face_detector 후처리**: 앵커 디코드(896×16, SSD 앵커) + NMS → 얼굴 박스
3. **ROI 파이프라인**: 박스→회전 정규화 크롭→face_landmarks→역변환, 이전 프레임
   ROI 트래킹(검출은 놓쳤을 때만) + OneEuroFilter
4. **게이트: MediaPipe wasm과 같은 프레임 좌표 diff** — 파리티 증명 전 교체 금지
(1~2는 스파이크로 face만 먼저 — hand는 같은 골격 복제.)

**fastenhancer(#6, 오디오)는 그다음**: 선행조건(ai-cpu)은 풀렸지만 DFT·1D
conv·복소 = 새 커널 계열이라 스파이크 단위가 크고, #4는 이미 변환해둔 5모델을
기능으로 바꾸는 마지막 조각이라 레버리지가 더 크다. (#3 잔여인 tflite 직접
임포터도 #4 뒤로.)
**(B) 브라우저 R11 A/B 배선 — 완료 (2026-08-13 저녁, §2 실측 참조).**
   `web/demo/cpu-ab.html` — 같은 프레임을 ai-cpu/tflite-simd/ORT wasm 셋에 먹여
   마스크 3장 + p50/p90 + ai-cpu 대비 diff. ORT와 diff 0.0000(fp32 수치 일치) 확인.
   COOP/COEP 없는 서버라 셋 다 1T = v-ai 배포 기본 조건과 동일(공정).
   처음 발견된 2.6배 열세(17.9 vs 6.9ms)는 **같은 날 밤 커널 스프린트로 해소 —
   wasm 7.1ms = tflite-simd(6.9)와 동률, §2.5 참조.**

v-ai 정찰 전체 지도(런타임·티어·플래그·file:line)는 메모리 `vai-runtime-map` 참조.

---

## 확정된 로드맵

| # | 항목 | 상태 |
|---|---|---|
| 0 | `ai-tasks` 크레이트 (공개 API 본체) | **진행 중 — 아래 참조** |
| 1 | 폴백 트리거 3종 | **엔진 쪽 완료** (2026-08-13) — v-ai 연결만 보류, §1 참조 |
| 2 | `ai-cpu` (SIMD128/NEON + 스레드) + R11 vs tflite-simd A/B | **완료** (2026-08-13) — 브라우저 A/B + 커널 스프린트로 **tflite-simd 동률**, §2·§2.5 |
| 3 | `Reshape` canon + `PRELU` + `MAX_POOL_2D` | **완료** (2026-08-13) — 5모델 전부 개통+ORT 게이트+벤치, §3.5 |
| 4 | 랜드마크 스파이크 (face_detector → 좌표 diff) | ← 다음 (모델은 다 돈다 — calculator 그래프 차례) |
| 5 | 파이프라인 로직 (ROI 트래킹 / 회전 정규화 / OneEuroFilter / Horn 피팅) | |
| 6 | 오디오 (DFT / 1D conv / 복소) | |

**폴백 사다리 (사용자 확정, 2026-08-13)**: ①로드 시 GPU 체크(`NoGpu`) → ②돌렸을 때
프레임 실제로 나오는지(DeviceLost는 됨, "조용히 안 나옴" 헬스체크는 미구현) →
③안 되면 R11을 CPU로(`ai-cpu`) → ④CPU도 안 되면 구조화 에러 → 호스트가 "호환 안 됨"
팝업. 모든 비전 태스크 공통. 전환 스위치는 호스트(모델 조달이 호스트 몫).

**워커 토폴로지·백엔드 배치 (사용자 확정, 2026-08-13)**:

| 워커(웹)/스레드(모바일) | 모델 | 백엔드 |
|---|---|---|
| video 워커 | 세그(가상배경) | **GPU 고정** (매 프레임, 최우선) |
| vision 워커 | face det+lm, 게이즈 | GPU 기본 → **세그 p90 악화 시 CPU 강등** |
| audio 워커 | fastenhancer(#6) | CPU 고정 (AudioWorklet엔 WebGPU 없음) |

- 근거: 워커 = 스레드 1개 + wasm 인스턴스/메모리 분리 → **CPU 강등이 다른 워커에
  무영향 + COOP/COEP 없이 멀티코어** (SAB 불필요). 단 GPU 큐는 워커가 나뉘어도
  하나 — "후달림" 신호는 **세그 모델의 p90**(v-ai p90>66ms 규칙), `model_stats`/
  `model_stats_cpu`가 그 입력.
- vision 워커는 det→lm→게이즈가 한 프레임 순차 파이프라인이라 **핸들 기반 다중모델
  API**가 필요 (지금은 워커당 GPU 1 + CPU 1 슬롯). 이 확장이 로드맵 #4의 첫 삽.
- 웹 CPU는 워커당 1T (게이즈 48ms급은 강등 시 N프레임에 1회 페이싱 — 파이프라인
  로직 #5에서). 모바일은 스레드 자유 + CPU 4T라 강등 페널티가 작다.
- op 단위 GPU/CPU 분할은 안 한다 — 경계 리드백이 이득을 압도 (모델 단위 배치만).

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

## 1. 폴백 트리거 3종 — 엔진 쪽 끝 (2026-08-13)

1. ✅ **device lost 구독** — `GpuContext::new()`가 `set_device_lost_callback` 등록
   (웹은 `device.lost` 프라미스 경로라 `Error::from_js` panic 이슈와 무관 — 소스로 확인).
   `ctx.lost_reason()` 폴링, `Segmenter` 프레임 경로 3곳에서 체크 → `TaskError::DeviceLost`,
   wasm `device_lost()` export (`null | "사유"`). 테스트: `ai-gpu/tests/device_lost.rs`.
2. **프레임타임 노출** — `model_stats()` + `model_stats_cpu()` 있음. 호스트가 쓰게 하는 건 v-ai 연결과 한 묶음.
3. **v-ai 연결 — 사용자가 "나중에"라고 보류함. 착수 전 확인 필수.** 구멍은 그대로:
   ```
   vcxreact/packages/v-ai/src/virtual-background/video-worker-webgl2.js
     :444  _recordCycle(dt)        — p90 > 66ms 2윈도우 연속이면 _demoteSegTier
     :488  _recordCycle 호출       — ★ _createOnnxAdapter 경로에만 있다
     :3577 _demoteSegTier('engine init 실패')  — 엔진 티어는 init 실패만
   ```
   즉 **`gl-rvm`(자체 엔진) 티어는 아무리 느려도 강등되지 않는다.**
   `es.render()` 경로에도 `_recordCycle`을 걸어야 한다. ai_engine을 티어로 넣어도 같은 상태.

---

## 2. `ai-cpu` — 구현 완료 (2026-08-13), 네이티브 실측 끝

구조 (ARCHITECTURE.md "CPU 백엔드 짝" 절차 참조):
- `simd.rs` F32x4 (NEON/SIMD128/스칼라 — 커널은 core::arch 모름), `plan.rs`(로드 시
  가중치 재패킹 + last_use 슬롯 재사용 계획), `exec.rs`(프레임 중 분석·할당 없음),
  `kernels/`(conv=브로드캐스트 GEMM 4px×8cout, dw=채널 벡터화, 나머지 콜드 스칼라).
- alias/concat 융합은 채널 스트라이드 뷰로 무복사. 상태 ping-pong은 프레임 시작 swap.
- 스레드: rayon 행 밴드(네이티브 전용, `set_threads`). wasm 스레드는 COOP/COEP 필요 — 미착수.
- `ai-tasks::CpuSegmenter` + wasm export: `load_model_cpu/infer_frame_cpu/model_stats_cpu/model_io_cpu`.
  Makefile `+simd128` 추가됨. wasm 725KB (vs v-ai 런타임 66MB).

**M2 Pro 실측** (오라클 diff CpuExec 대비 max_err ≤2e-5 전 해상도):

| R11 (mnv4s050_s2) | 1T | 4T | 순진 CpuExec |
|---|---|---|---|
| 144×256 | 8.07ms | **2.94ms** | — |
| 160×288 | 10.08ms | **3.57ms** | 119.1ms (33×) |
| 192×320 | 13.29ms | **4.51ms** | — |

참고선(ncnn 손튜닝, RVM 144×256): 4T fp32 19.6ms. 목표(10~15ms) 초과 달성.

재현:
```sh
AI_ONNX=<onnx> cargo test --release -p ai-cpu --test oracle_real -- --ignored --nocapture  # 정확도
AI_ONNX=<onnx> AI_THREADS=4 AI_REPS=100 cargo test --release -p ai-cpu --test bench_cpu -- --ignored --nocapture
# R11 160×288 로컬 사본: models/segm_mnv4s050_s2_160x288_nhwc.onnx (gitignore — 원본은 v-ai assets/models)
# 다른 해상도(144×256/192×320)는 M2 Pro 머신: /Users/foxcom/Desktop/segm-ft/export/mnv4/r11/
```

**브라우저 A/B 완료 (2026-08-13 저녁, M1 Pro)**: `web/demo/cpu-ab.html` + `cpu-ab.js` —
같은 카메라 프레임(288×160, RGB [0,1] 한 벌)을 셋에 먹이고 2채널 로짓을 각자
softmax(사람확률 = sigmoid(사람−배경), v-ai softmax 스테이지와 같은 식).
자산: `make convert-r11-web`(V_AI ?= ../../vcxreact/packages/v-ai에서 tflite 복사),
런타임 glue는 web/demo/tflite/, 모델 3종은 web/models/(gitignore).

| 헤드리스 크로미움 1T (COOP/COEP 없음 = v-ai 배포 기본) | p50 |
|---|---|
| ai-cpu (wasm SIMD128) | **17.9ms** (벽시계 18.2) |
| tflite-simd (XNNPACK, f16 가중치) | **6.9ms** |
| ORT wasm (동일 fp32 onnx) | **11.0ms** |

정확도: ORT vs ai-cpu **diff 0.0000**(같은 fp32 그래프 수치 일치 — 배선 정합 증명),
tflite는 f16 가중치라 0.02~0.06. 이 머신 네이티브 1T 10.89ms → wasm 갭 1.64배(정상 범위).
재현: `node tools/run_web.mjs demo/cpu-ab.html --camera` (수치), `--headed`(눈검증).

**남은 것**: ①wasm 스레드(코옵 헤더 요구라 v-ai 배포 환경 확인 먼저), ②"조용히
프레임 안 나옴" 첫 프레임 헬스체크(NaN/전부0/시간 상한), ③프로덕션 로더(ai-tasks/
v-ai 연결)에 relaxed-simd 이중 빌드 선택 로직 이식 — 지금은 cpu-ab.js에만 있다.

---

## 2.5. XNNPACK 추격 스프린트 — 완료 (2026-08-13 밤): 2.6배 열세 → **역전**

**결과 (R11 160×288, 1T)**: 네이티브 10.89 → **~5.1ms** (4T 3.57 → **~2.2ms**),
Chrome wasm 17.9 → **6.8ms 벽시계 / 6.6 순추론 = tflite-simd(XNNPACK) 6.9를 제침**,
ORT wasm 11.0 대비 1.6배. 정확도 게이트 전 과정 유지: oracle max_err 1.4e-5,
브라우저 ORT diff 0.0000, 전 스위트(89) 그린.

**방법론**: ① 네이티브 — `ai-cpu/tests/prof_cpu.rs` per-op 예산표(rep별 min vs
이론하한, `CpuModel::infer_profiled`). ② **wasm — `node tools/profile_web.mjs
'demo/cpu-ab.html?only=ours' --ops`** (`CpuModel::bench_steps` = 스텝당 N회 합산으로
100µs 타이머 양자화 우회; V8 CDP 샘플 프로파일은 전 커널이 한 함수로 인라인돼 무용).
**네이티브 예산표로 wasm을 추정하면 틀린다** — 스템은 wasm에서 1.74배, concat 1.5배로
뒤틀렸었다. wasm 최적화는 반드시 wasm 스텝벤치를 근거로.

**적용한 것 (효과 순)**:
1. **conv 마이크로커널 = XNNPACK gemm-splat 구조** (소스 대조: scratchpad에 받아 확인):
   A를 픽셀당 벡터 1로드 + lane 브로드캐스트 fma 4회 (NEON `fmla by lane` /
   wasm `i32x4_shuffle` — splat 로드 4개 → 로드 1개). 경계검사 제거(`load_splat`/`load`
   무검사), 에필로그 벡터화(act `apply4` + residual + 벡터 store, 부분블록 nc≥4 포함).
2. **MR 확장**: 4px→MR_BIG px 마이크로커널(`mr_n::<MR>` const generic). **네이티브 8 /
   wasm 6** — 누산 체인 2×MR개가 FMA 지연 은닉. 디코더 conv 46→74 GF/s.
3. **relaxed-simd 이중 빌드**: `web/pkg-relaxed`(+relaxed-simd, `f32x4_relaxed_madd`=FMLA)
   + 기본 `web/pkg`. 로더는 relaxed 먼저 import하고 CompileError면 폴백(Safari) —
   cpu-ab.js. wasm -9%.
4. **dw 재작성**: conv_std식 3분할(행별 ky목록 + interior ox구간) + (행,채널블록)당
   tap 오프셋·가중치 테이블 + 4px 동시(체인 4개). 5.2 → 20+ GF/s (c44 1.75→0.37ms).
5. **resize**: ox 좌표 테이블(행 불변) + 채널 벡터화 + **c2 픽셀페어 패킹**(`low2_concat`,
   세그 마스크 최종 업샘플 0.35→0.067ms).
6. **elementwise/mix/concat(copy_view_into) 벡터화**, **슬롯 +4 패딩**(4레인
   오버리드를 crate 전체에서 안전화 — exec.rs 참조).
7. **역전 마무리 3종 (wasm 스텝벤치 근거)**: ⓐ **ConvStem** — cin≤4 k>1 conv를
   im2row 패치(`[px][k_pad]`, K=(ky,kx,ic) 연속)로 펴서 1x1 GEMM화 (스템 wasm
   1.03→0.58ms; im2row.rs, 오버런 규약은 파일 머리 주석). ⓑ **PwDot** — cout≤4
   1x1 헤드를 채널 내적으로 (NR8 레인 6개 버리던 것; 0.17→0.107). ⓒ **NR16
   (wasm 전용)** — cout 16배수 블록은 mr4_nr16: 셔플 1개가 madd 4개 서빙 (NR8은
   2개). ⓓ **출력 zero-copy**: `infer_frame_cpu_view` — wasm 힙 뷰 반환(tflite
   HEAPF32 규약, 368KB/프레임 2차 복사 제거). 뷰는 다음 호출 전 소비.

**실패/무효 실험 (재시도 금지)**: ①**MR8 wasm = 스필 경계** (V8 arm64 가용 vreg ~28,
acc16+A8+W2=26; MR6이 7.6→7.1ms, MR8은 7.6) — 네이티브(NEON 32reg)만 8. ②스템
K패딩(제로패딩만으로는)은 네이티브·wasm 모두 무반응 — im2row(⑦ⓐ)가 정답이었다.
③wasm에서 MR8 단독(relaxed 없이)은 무반응 — wasm은 FMA 없인 처리량 병목이라 체인
수가 안 도움. ④V8 CDP 샘플 프로파일로 wasm 커널 분해 — 전부 한 함수로 인라인돼
못 씀 (bench_steps가 대안).

**남은 여지 (지금 안 함)**: 디코더 conv(wasm 2.18ms, 네이티브 대비 1.35배 — 셔플
잔여세. dot-product 재구성(MR2NR8, hsum)이 이론상 남은 카드나 리스크 있음), 소형맵
dw(@18x10 에지 지배), gpool/segate 스칼라(µs급). tflite 역전 달성으로 우선순위 하락.

---

## 3.5. MediaPipe 5모델 개통 — 완료 (2026-08-13 밤)

**결과**: `.task`의 tflite 4종(tf2onnx 경유) + 게이즈 onnx 전부 **변환·ORT 게이트·벤치
통과 — CPU·GPU 양 백엔드** (2일째 밤에 GPU 개통 + CPU 추가 개선 반영).

| 모델 | 네이티브 CPU 1T | wasm ai-cpu | **WebGPU** | ORT wasm 1T | GPU 네이티브 |
|---|---|---|---|---|---|
| gaze 448² (1.10 GMAC) | 36.0ms | 48.3 | **3.9** | 67.5 | 10.3 |
| face_detector 128² | 1.7ms | 2.4 | **0.63** | 3.3 | 1.5 |
| face_landmarks 256² | 6.8ms | 8.4 | **1.72** | 10.4 | 2.5 |
| hand_detector 192² | 16.6ms | 20.0 | **2.32** | 34.2 | 3.0 |
| hand_landmarks 224² | 21.1→**14.2ms** | 17.4 | **3.18** | 32.6 | 4.0 |

CPU는 ORT 대비 전 모델 1.24~1.87배, **WebGPU는 ORT wasm 대비 10~17배**.
MediaPipe tasks-vision e2e 12.1ms vs 우리 GPU face 파이프라인(det+lm) 2.4ms.
(⚠ WebGPU가 네이티브 GPU보다 빠른 건 Tint MSL — RVM 때와 같은 현상.
GPU 프레임타임 측정은 **동기화 필수**: infer는 제출만 한다 — gate_models_gpu가
마지막 리드백으로 강제. 동기화 없으면 0.3ms 허수가 나온다.)

**추가된 op/canon**: SwOp::Maxpool(+pad_c=BlazeFace 채널패드 접기, GPU/CPU 커널),
BinaryOp::Prelu(cvec slope, wgsl_expr 함수형 코드젠), Activation::Relu6,
canon 4종(reshape→chcopy 실체화·gemm→1x1 Conv·pad 접기/k1s1 maxpool 폴백·
transpose flat_ok 마킹), shape 추론 5종(Reshape/Squeeze/Gemm/Pad/PRelu),
desc_of rank-3 flatten, 임포터 견고화.

**잡은 잠복버그 2개 (기존 코드)**: ① dce가 res attr(fuse_residual)을 읽기로
추적 안 함 — res 유일 소비 생산자가 통째로 죽음 (facelm maxpool 전멸로 발견).
② fuse_act가 모르는 act 태그를 "none"으로 뭉갬 — relu6 증발, 출력 1e9 폭주.
디버깅은 층별 이등분: `dump_all.rs` + `tools/diff_dump.py` (ORT --intermediates
대조, NCHW/NHWC라 정렬비교, act 융합은 Clip후 매핑).

**GPU 개통에서 잡은 것 (2일째)**: ①desc_of가 [1,1,1,N]을 (h1,wN,c1)로 잡아
GPU C4 레인패딩과 평탄 버퍼가 어긋남 → (1,1,N) 채널벡터로 접음. ②지오메트리를
바꾸는 chcopy(flatten)는 c%4≠0에서 레인 재배치 필요 → **flatten 커널 신설**
(ai-gpu/kernels/flatten.rs, 런타임이 out_c≠n일 때 라우팅). CPU는 밀집이라 무관.
디버깅 교훈: **GPU 중간텐서 이등분은 AI_RT_NO_REUSE=1 필수** — 버퍼 재사용
때문에 안 걸면 tid1부터 전부 허수 발산으로 보인다 (gate_models_gpu AI_BISECT=1).

**CPU 추가 개선 (2일째)**: ①k1s1 conv 지오메트리 접기(전 픽셀 한 행 — 7×7 소형맵
mr1 낭비 제거, hand_lm 21→14ms) ②elementwise dense 플랫 루프. PRelu 단독 op는
메모리 대역 한계 — **conv+PRelu(cvec) 에필로그 융합이 다음 CPU 카드** (face_lm
~1.5ms어치). dw k5 슬라이딩 윈도우(x-로드 재사용)도 미착수 카드 (hand_det 6.4ms어치).

**도구**: `tools/prep_mediapipe.py`(.task→onnx, tf2onnx+후처리),
`tools/ort_dump.py`(게이트 기준값, --intermediates), `gate_ort_models.rs`(CPU)·
`gate_models_gpu.rs`(GPU, AI_BISECT 층별 이등분), `dump_all.rs`+`tools/diff_dump.py`,
`web/demo/models-bench.html`(5모델 CPU/GPU/ORT 벤치). tf2onnx는 /usr/bin/python3 --user.

**남은 것**: ①tflite flatbuffer 직접 임포터(로드맵의 (B) — 지금은 tf2onnx 경유),
②커널 여지 — hand 계열이 15~20 GMAC/s로 낮음(PRelu 비융합 69개, dw 비중,
Gemm@1x1 — conv+prelu 융합이 1순위), ③MediaPipe e2e 정밀 A/B(같은 프레임 좌표
diff — 로드맵 #4의 검증 게이트와 한 몸), ④face_blendshapes(잡op 많음, 별도 판단).

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
