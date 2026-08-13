# ai_engine 아키텍처

wgpu(WGSL 컴퓨트 셰이더) 기반 범용 비전 추론 엔진. 웹(wasm+WebGPU)/데스크탑/모바일을 하나의 코어로 커버한다.
설계 목표: RVM(MNv3/MNv4 백본)·facelandmark·handdetector·focustracker 등 **모델 불문** 실행,
webgl2-engine-span.js(~3ms)와 ORT-web WebGPU(~2ms)를 넘는 추론 성능, memcpy 수준의 모델 로드.

## 크레이트 구성

| 크레이트 | 역할 | 비고 |
|---|---|---|
| `ai-core` | GPU 무의존 코어: TensorDesc(NHWC-C4), pack/unpack, 활성화 정의, CPU 레퍼런스, 그래프 IR·컨테이너 포맷 타입 | rlib. 변환기·런타임이 공유하는 계약 |
| `ai-gpu` | wgpu 30 실행 계층: context/caps, arena, params, 커널, 캐시, 프로파일러 | 타겟별 의존성 테이블(native=metal/vulkan/dx12, wasm=webgpu) |
| `ai-runtime` | 그래프 executor: 컨테이너 로드 → lowering → 상태 텐서 ping-pong | |
| `ai-tasks` | **공개 API 본체** — 태스크(세그멘테이션/랜드마크/노이즈억제) + 전·후처리 + 합성 + 폴백 정책 | 플랫폼 무관. 바인딩이 여기를 감싸기만 한다 |
| `ai-convert` | ONNX → 컨테이너 CLI: BN 폴딩·사전 패킹·융합 마킹 | |
| `ai-bench` | 네이티브 벤치 러너 | wasm 데모와 동일 루틴 |
| `ai-wasm` | wasm-bindgen 경계 (`cdylib`) — **웹 API** | 서피스·프레임 임포트·타입 변환만 |
| (향후) `ai-cpu` | SIMD128/NEON + 스레드 실행 계층 | GPU 미지원·저사양 폴백 |
| (향후) `ai-ffi` | 모바일 C ABI (`staticlib`+`cdylib`) — **모바일 API** | crate-type은 leaf가 소유 — 플랫폼별 수동 편집 금지 |

의존 그래프: `ai-core ← ai-gpu ← ai-runtime ← ai-tasks ← {ai-wasm, ai-ffi}`, `ai-convert ← ai-core`.

**바인딩 규칙**: `ai-wasm`/`ai-ffi`에는 분기(`if`)가 없다. 분기가 생겼다면 그건 로직이고
`ai-tasks`로 내려가야 한다. 이 규칙이 깨지면 웹과 모바일 동작이 갈라진다 — 지금 랜드마크가
웹(mediapipe-wasm)과 모바일(vcxrust_ai/ncnn) 두 벌로 갈라져 있는 것과 같은 실수를
한 층 아래에서 반복하게 된다. 플랫폼마다 **진짜** 다른 것만 바인딩에 남긴다:
서피스 획득, 프레임 임포트, 스레드 모델, 모델 바이트 조달.

## 설계 원칙 (요약)

1. **오프라인 변환기가 무거운 일 전부** — BN 폴딩, NHWC-C4 사전 패킹, dtype 변환. 런타임 로드 = memcpy.
2. **융합이 기본값** — conv+bias+activation(+residual)은 단일 커널 에필로그. 별도 activation 디스패치 금지.
3. **NHWC-C4 레이아웃** — `idx(h,w,cg) = (h*W+w)*cg_count + cg` (vec4 단위). C만 4패딩, W/H 무제약.
4. **디스패치 규율** — 프레임당 인코더 1개, 그래프 중간 CPU 리드백 금지, bind group/파이프라인은 로드 시 전부 생성.
5. **shape은 셰이더 상수** — WGSL 템플릿에 리터럴로 주입, 유니폼 분기 없음. 캐시 키 = codegen 입력 전체.
6. **정밀도는 축(axis)** — dtype{F32, F16(누산 f32), 예약 Int8}이 캐시 키에 포함. 기기별 기본값은 런타임 정책(웹/데스크탑 f32, 모바일 f16).
7. **상태 텐서는 일반 기능** — RVM의 r1~r4 같은 순환 피드백은 그래프 선언(`input "rN"` ↔ `output "rNo"`)으로 표현, 엔진이 GPU 상주 ping-pong 자동 관리.

## 파일 조직 원칙: "하나의 개념 = 하나의 파일"

커널·op·CPU 레퍼런스·테스트 모두 계열별 개별 파일. 셰이더는 Rust 문자열 인라인 금지 —
커널별 `.wgsl` 템플릿 파일(`include_str!` + 슬롯 치환: `//@CONSTS`, `//@LOOP_BODY`, `//@EPILOGUE` 등).

각 `ai-gpu/src/kernels/<이름>.rs`는 반드시 다음을 담는다:
1. `<이름>Spec` 구조체 (커널의 모든 codegen 입력)
2. 변형 선택 정책 (순수 함수, 예: M<512 → small 변형)
3. `KernelSpec` impl (`cache_key` / `wgsl` / `bindings` / `workgroups`)
4. 자체 naga 검증 테스트 (`#[cfg(test)]` — GPU 없이 WGSL 파싱·검증)

## 새 커널 추가 절차

1. `ai-gpu/src/kernels/shaders/<이름>.wgsl` — 정적 골격 + 슬롯 마커
2. `ai-gpu/src/kernels/<이름>.rs` — Spec + KernelSpec impl + naga 테스트 (위 규약)
3. `ai-core/src/reference/<계열>.rs` — CPU 레퍼런스 (없으면 추가)
4. `ai-gpu/src/testsuite.rs` — GPU vs CPU 케이스를 그리드에 등록
5. (Phase 2+) `ai-runtime/src/lowering/<op>.rs` — op → KernelSpec 매핑

## 새 모델 추가 절차 (Phase 2+)

1. `ai-convert`로 ONNX → 컨테이너 변환 (부족한 op는 변환기가 목록으로 보고)
2. 부족한 op만 "새 커널 추가 절차"로 증설
3. 모델별 전·후처리 파이프라인 파일 1개 (검출→크롭→랜드마크 같은 다중 스테이지는 파이프라인 레이어)

## 폴백 전략

- init은 즉시 판정: `is_supported()` + 구조화된 `InitError{NoWebGPU|NoAdapter|LimitsInsufficient|DeviceLost}`
- 웹 티어: ai_engine(WebGPU) → 기존 webgl2 엔진(변환기의 plan.json 호환 출력으로 동일 모델 재사용) → ORT wasm/tflite
- 네이티브/모바일: adapter-info 기반 capability 티어, 최후엔 CPU 백엔드(후속 Phase)
- 런타임 강등 신호(p90 프레임타임 초과, device lost)를 이벤트로 노출

## 빌드/검증

```sh
make setup-wasm     # 최초 1회
make test           # 네이티브 정확도 (CPU 오라클 + naga + GPU vs CPU)
make bench          # 네이티브 벤치
make build-wasm     # wasm-pack → web/pkg/
# web/index.html을 VS Code Live Server로 열어 브라우저 검증
```
