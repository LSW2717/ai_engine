# 다음 세션 시작점 (2026-08-14 마감)

## ⚡ 야간 자율 작업 지시 (사용자, 2026-08-14 밤 — 중단 없이 순서대로)

1. ~~blendshape 마감~~ **완료** (§C-6)
2. **집중도 모드(GazeTask) — 코어 완료 (2026-08-15)**: `features/gaze/` —
   preprocess(비회전 크롭: bbox 전체 min/max + margin 0.18X/0.22Y **×bbox 크기**,
   비대칭 클램프·재센터 없음·8px 게이트, cv2 반픽셀 bilinear, ImageNet 정규화
   **인터리브 NHWC — 엔진 업로드 규약**, 90bin softmax 기댓값 ×4−180) +
   one_euro(freq 없는 변형: minCutoff 1.3/beta 0.45/dCutoff 1.0, dt=타임스탬프차
   하한 1ms) + state(집중 판정은 원뿔이 아니라 **비대칭 박스** |yaw|≤24°·pitch
   [−22,+16]°, 히스테리시스 집중→이탈 350ms/이탈→집중 250ms/무얼굴 600ms/눈감김
   450ms, **무얼굴 600ms 미만은 직전 상태 유지**, score 30s 창 raw값, baseline
   24샘플 독립 upper-median idx12) + task(CNN 페이싱 83.3ms, 출력 이름 매칭
   yaw/pitch, 얼굴 소실 시 필터만 리셋). 테스트 8종 그린. gaze.sw 게이트 기통과.
   ~~남은 것: wasm export + studio HUD + 웹 diff 게이트~~ **전부 완료 (2026-08-14)**:
   ① wasm `gaze_task_new/free/reset/gpu`(FaceTask 478pt flat [x,y] 배선, 빈 배열
   =소실 틱) + 게이트 헬퍼 `gaze_crop_box/crop_pixels/normalize/decode_bins`
   (preprocess 순수 함수 1:1 노출), GazeTask에 `last_filtered`/`last_cnn()` 노출.
   ② studio 집중도 체크박스+HUD — FaceTask lm 1회 추론을 아이템 오버레이와 공유
   (v-ai 이중 로드 낭비의 수리 지점), 비전 틱 10fps 페이싱. 헤드리스 스모크:
   무얼굴 INITIALIZING→NO_FACE 전이 확인, 스크린샷 OK.
   ③ **게이트 `web/demo/gaze-ab.html` 전부 PASS**: box 3.6e-9 / crop 6.2e-6
   (cv2 f64 기준 vs 엔진 f32) / angle 0.000°(gaze.sw+rust decode vs ORT+JS
   decode, 같은 크롭) / e2e-resample 0.50°≤1.5°(같은 u8 양자화끼리 — cv2
   재양자화 vs drawImage) / task FOCUSED·score 100 / noface 홀드→NO_FACE.
   재현: `node tools/run_web.mjs demo/gaze-ab.html`.
   **게이트 설계 교훈 (vb-diff EMA와 같은 원리)**: |엔진 f32 − 웹 drawImage u8|
   절대각은 게이트 불가 — u8 재양자화만으로 5.6° 이동 (이 테스트 프레임은 얼굴
   ~100px→448² 4.5배 업스케일이라 softmax가 평평해 서브 LSB 노이즈가 수 °로
   증폭). 웹 경로 자체가 getImageData u8이고 우리 f32가 상위 정밀도, 웹 규약상
   절대각은 baseline 상대 일관성만 필요 (gazeModel.ts 주석). 게이트는 같은
   양자화끼리 비교해 리샘플 차만 격리한다.
   ~~잔여: 다중 모니터 투영 분류기 API / blink blendshape 절반~~ **전부 배선
   완료 (2026-08-14)** — §"집중도 마감 3종" 참조 (gaze_layout API + EAR∨bs +
   MULTIPLE_FACES).
3. **박수 인식 — 판정 로직 완료**: `features/hand/gesture.rs` —
   ClapDetector **개선판** (웹 실패모드 수리: ①접촉 순간 양손→한손 융합 미발화
   → **융합 브리지 트랙**(FUSE_D 1.0·GRACE 5·접근속도) ②handedness 오판 무발화
   → 보조 신호로 강등(팜 중심 거리 겸용) ③1프레임 드랍에 pair 리셋 → 드랍 용서.
   웹 상수 CLOSE 0.4/APART 1.2/쿨다운 350ms 유지, 테스트 4종: 느린 박수 rising
   edge·빠른 박수 융합 브리지·한손 무발화·오판 내성) + thumbsUp/handRaise 1:1
   이식.
   **HandTask 조립 + hand-ab 게이트 완료 (2026-08-14)**: `features/hand/`
   roi.rs(팜 det→ROI + lm→다음 ROI) + task.rs(2손 트래킹 — HandAssociation
   축정렬 IoU 0.5·num_hands 클립·presence 0.5 게이트·월드 lm 회전 투영) +
   wasm `hand_task_*`/`gesture_*` exports. **게이트 web/demo/hand-ab.html
   PASS**: MediaPipe HandLandmarker(wasm, 같은 .task) 대비 **21pt max 1.02px /
   mean 0.33px** (CPU=GPU 동일), handedness 라벨 일치, 2프레임째 디텍터 생략
   (트래킹 계약), gesture export 합성 박수 발화+실손 무발화. 프레임은 MediaPipe
   공식 hands.jpg (make convert-mediapipe가 다운로드).
   **MediaPipe 출하 quirk 3개 — 원본 소스로 확정, 파리티 우선 복제 (재론 금지)**:
   ①팜 det ROI target angle이 **90 라디안** (tasks가 도(°) 필드 대신 라디안
   필드에 90을 넣음 — hand_detector_graph.cc; NEXT.md의 "90° 추정"은 이거였다)
   ②lm→ROI 회전 조인트 0/4/6/8 = 12점 서브셋 시절 인덱스가 전체 21점에 그대로
   (실제로는 wrist/thumb-tip/index-PIP/index-tip) ③handedness 원시값은
   **P(Right)** (TensorsToClassification binary: index0=raw=label "Right") —
   HandResult 필드는 1−raw=P(Left)로 통일.
   **게이트가 잡은 버그**: 팜 검출 입력 범위는 **[0,1]** (face는 [-1,1]) —
   letterbox_u8_rgb에 범위 인자 추가, DetectorPost::input_range() 신설
   (face-ab 무회귀 재확인 1.10px/0.30px). 잔여: 개선 상수 실카메라 튜닝(사용자),
   HandTask studio/워커 배선은 vision 워커 조립 때.
4. ~~3D 에셋 = wgpu로~~ **완료 (2026-08-14)**: vcxrust_ai items3d 이관 —
   `ai-tasks/features/face/items3d.{rs,wgsl}` + env_room.bin (Horn 피팅·GLB
   직파싱·PBR·MSAA4·씬 광원 프로브 전부). **studio three.js 오버레이 삭제**
   (importmap·#fx 캔버스 제거) — 오버레이가 서피스에 직접 알파-오버
   (`ItemsOverlay`: 렌더→리졸브→블릿). 이관 차이 4개는 items3d.rs 헤더 주석:
   ①에셋 파일 IO → **호스트 bytes 주입**(`preload_glb` — wasm fetch/ffi fs)
   ②pollster 에러스코프 제거(naga 정적 테스트가 게이트, MSL+SPIR-V 코드젠까지)
   ③RGB 프레임 광원 프로브 추가(웹 probeSceneLight 등가 — YUV판은 모바일용 보존)
   ④ai_engine wgpu 30은 API가 다른 빌드(Option 필드들·multiview_mask 등).
   wasm exports: `studio_items/studio_item_glb/studio_items_pose/studio_items_probe`.
   검증: 유닛 11종(피팅 3·GLB 5·env·wgsl 2) 그린 + **합성 얼굴 카메라(y4m)
   헤드리스에서 hat1이 B티어·배경블러 위에 실렌더 — 스크린샷 확인**. 실카메라
   착용감(스케일·앵커) 눈검증은 사용자 몫 — 보정값은 웹 GLB_HATS와 동일.
   ⚠ 카드: wasm 1.96MB (image png/jpeg/webp 디코더 추가분) — 저사양 로드타임
   상 텍스처 디코드 외부화(createImageBitmap→RGBA 주입) 또는 feature 분리 검토.
   ⚠ 프레이밍 크롭 중 아이템 좌표 보정은 여전히 P3 이월분 (INTEGRATION.md §2).
5. ~~wasm 빌드 + 웹 테스트~~ **완료 (2026-08-14)** — 전 게이트 그린: vb-diff /
   face-ab 1.10px / hand-ab 1.02px / gaze-ab / ffi-diff / studio 헤드리스+스크린샷
   (합성 얼굴 y4m으로 아이템 실렌더까지). 네이티브 전 스위트 0 실패.
6. ~~ai-ffi 뼈대 + 네이티브 구현·테스트 → 웹과 diff~~ **완료 (2026-08-14)**:
   `crates/ai-ffi` (staticlib+cdylib+rlib) — **vcxrust_ai C ABI 표면 재현**
   (모바일이 .so 교체만으로 갈아타는 seam): `set_video_stream_info(.sw 경로)` /
   `update_effects_config(JSON — 웹 studio_config와 같은 EffectsPatch 계약)` /
   `render_mask(I420 in-place — 실추론+이펙트 스택)` / `destroy` /
   `vcx_string_free`. panic 방벽(catch_unwind), YUV는 BT.601 full-range
   (yuv.rs — ⚠ CPU 왕복은 임시, GPU 상주 변환이 모바일 필수 카드).
   내부는 GateHarness 재사용 — `frame_infer` 신설(오프스크린 실추론+리드백,
   frame()과 타깃 공유). 미배선 표면(update_video_config, face/hand/item/focus)은
   헤더 주석에 명시 — 태스크는 ai-tasks에 완비, 배선은 모바일 실연결 때(로드맵 E).
   cbindgen 헤더 생성(ffi-header 타깃)도 그때.
   **게이트**: ①C 표면 스모크(실추론 — 무인물 프레임에 #00a05a 배경이 덮여
   Y평면 평균 ~104 확인) ②**네이티브=웹 교차 증명 `web/demo/ffi-diff.html`**:
   같은 픽스처(vb-diff makeFrame/makeMask Rust 1:1 포팅)를 네이티브(Metal/naga)가
   덤프 → 브라우저(Chrome/Tint)와 채널 diff — **color max=0 (비트 일치!) /
   blur max=1 mean 0.010**. 재현: `cargo test -p ai-ffi --release &&
   node tools/run_web.mjs demo/ffi-diff.html`.
- **성능 최우선** (저사양 타깃 원칙 유지)

**야간 지시 1~6 전부 완료 (2026-08-14)** — 남은 사용자 몫: 실카메라 눈검증
(아이템 착용감·집중도·박수 상수 튜닝).

**다음 세션 계획 (2026-08-14 밤, 사용자 확정 — 아바타(P5)는 에셋 없어 보류,
vcxreact 연결 제외)**:
1. ~~FaceTask 입력 GPU화~~ **완료 (2026-08-14)** — §"FaceTask 입력 GPU화" 참조.
   게이트 전부 그린: 커널 CPU 대조 max 7.9e-6 / process_tex vs process_gpu
   **0.000px** / face-ab lm-tex 0.30px + lm-tex-vs-cpu **0.000px** / studio·vb-diff
   무회귀.
2. ~~터치업/메이크업 이관~~ **완료 (2026-08-14)** — §"터치업/메이크업 이관" 참조.
   게이트: 래스터라이저 단위 5종 + tests/face_fx.rs(립 틴트 diff 36레벨·얼굴 밖
   0·해제 시 비트 복원) + vb-diff 무회귀.
3. ~~집중도 마감 3종~~ **완료 (2026-08-14)** — §"집중도 마감 3종" 참조.
   게이트: 상태머신 3종+블렌드셰이프 2종+num_faces 계약 네이티브 그린,
   face-ab bs 스테이지 **52계수 vs MediaPipe max 0.072 / blink 0.018** PASS,
   gaze-ab·studio 무회귀 (studio 헤드리스 bs=on).
4. ~~잔잔한 마감~~ **완료 (2026-08-14)** — §"잔잔한 마감 6종" 참조.
(당초 후보였던 "이미지배경 블러 사전 베이크"는 측정 정정으로 실크기 ~8ms/프레임
확인 — 카드는 유지하되 급하지 않음, 위 항목 뒤로.)

**→ 4항목 전부 완료 (2026-08-14 심야).** 남은 사용자 몫 (실카메라 눈검증):
터치업/메이크업 룩(기본값 = vcxrust 테스트 룩 — studio 체크박스), 프레이밍 중
3D 아이템 줌 동행, 실멀티모니터 gaze_layout, MULTIPLE_FACES(두 사람), blink
bs 체감. 다음 엔진 작업 후보는 "다음 작업" D(P4 웹 연결 — **사용자 확인 필수**,
v-ai 연결 보류 중) 또는 E(아바타 에셋 대기 / mnv4-RVM 교체).

## 다음 작업 (우선순위 — 2026-08-14 정리)

**A. P1 마무리 — VideoPipeline 완성** (지금 여기)
1. ~~웹 파이프라인 픽셀 diff 게이트~~ **완료 (2026-08-14)** — 16개 검사 전부
   채널 diff **max ≤ 1/255**. §"픽셀 diff 게이트" 참조. 이미지배경 자체 블러도
   이 작업에서 같이 개통(3번의 절반).
2. ~~프레이밍(인물 중앙화)~~ **완료 (2026-08-14)** — §"프레이밍" 참조.
   게이트: 크롭 수학 diff PASS(F/color·F/image max 1) + bbox 네이티브 게이트 +
   framing.rs 단위테스트 5종. ⚠ 잔여: 프레이밍 크롭 중 **페이스 이펙트 좌표 보정**
   (INTEGRATION.md §2 계약 — studio 3D 아이템이 아직 미보정, P3 오버레이 때 같이).
3. ~~mirror/degree~~ **완료 (2026-08-14)** — §"mirror/degree" 참조. 게이트 T상
   6종(cover wide/tall·mirror·90/180·쿼터턴) 전부 **max=0 완전 일치**.
   잔여: 배경 종횡비 반전(cropFactor≥1.6)의 blur-fill 사전합성
   (fitBackgroundForCanvas)은 미이관 — 세로 프레임 대응 때 host/엔진 배치 결정.
4. ~~studio HUD 정직화~~ **완료 (2026-08-14)** — HUD = rAF 간격 p50(체감 스루풋)
   + GPU 실측(5s 논블로킹 `gpu_sync`=onSubmittedWorkDone 샘플 — 루프 무정지)
   + 제출 벽시계(허수 참고).
   **⚠ 측정 정정 (2026-08-14 밤, 사용자가 잡음)**: 당초 기록한 "이미지배경+
   blur60+조명+프레이밍 64.3ms, 강등 코앞"은 **큐 대기가 포함된 HUD 샘플의
   허수**였다 — rAF가 GPU보다 빨라 큐가 밀린 상태의 수치. **페이싱 재측정**
   (`studio.html?perf=1` — 프레임 제출→gpu_sync 대기→다음, 스테이지별 30샘플):
   기본 6.6 / blur60 8.3 / 이미지배경 6.6 / 이미지배경+blur60 14.2 /
   +조명+프레이밍 **p50 16.1ms** (p90 19.3). 조합 분해: 이미지배경 자체는
   공짜, **이미지배경×블러 조합에서만 +7.6ms** (인셰이더 배경 블러 225탭).
   교훈: **HUD gpu 샘플은 강등 판정용(큐 포함이 옳음), 프레임 실비용 주장은
   반드시 ?perf=1 페이싱으로.** P2 카드(1회 사전 블러 베이크)는 유효하되
   실크기는 프레임당 ~8ms (64ms가 아님).

**B. P2 티어 — 저사양이 제품 타깃 (사용자 최우선 강조, target-hardware-lowend)**
5. **B 티어 코어 완료 (2026-08-14)** — 완료 기준 충족: 헤드리스에서 GPU 추론을
   막고(tier=b 강제) 배경·블러·조명·프레이밍 생존 확인 (`tier=B cpu_infer=7ms`).
   - 경로: studio.js가 src를 288×160으로 축소 → `infer_frame_cpu`(R11) 로짓 →
     `studio_frame_mask`(신설 export — 캔버스 임포트+`process_gpu_mask` ch=2).
     `process_gpu_mask`에 **마스크 치수 파라미터** 추가 — CPU 마스크(288×160)가
     GPU 모델 해상도(RVM 256×144)와 달라도 ingest가 textureLoad로 리샘플.
   - 강등: auto 티어 = gpu 실측(5s 샘플) 66ms 초과 **2연속 → B, 승격 없음**.
     studio에 티어 셀렉트(auto/A/B) + 현재 티어·강등 표시 + cpu_infer HUD.
   - 잔여 정리 (2026-08-14): ①~~세션 없는 ensure~~ **완료** — GpuLeg 분리 +
     process_mask_nogpu (§"잔잔한 마감 6종") ②~~강등 창(p90)~~ **완료** —
     studio.js v-ai 규약 (엔진 이관 여부는 P4 연결 때 재론) ③이미지배경 블러
     사전 베이크 카드 유지 ④효과별 최소 티어 게이팅은 C 티어(소프트 합성) 생기면.

**C. P3 얼굴 스택 — 아바타(표정 포함 인물 교체)의 전제**
6. ~~blendshape 모델 개통~~ **완료 (2026-08-15 새벽)** — **CPU max_err 7.5e-7 /
   GPU 6.6e-7 / GPU 1.34ms**, 기존 5모델+RVM 전부 무회귀(전 스위트 59 그린).
   재현: ort_dump(오라클 2벌 — 게이트는 최종만, 이등분은 --intermediates) →
   gate_ort_models(CPU)/gate_models_gpu(GPU).
   **이번에 뚫은 것 (MLP-Mixer 일반화 — 다음 트랜스포머류 모델의 초석)**:
   ①Activation 3종(Sqrt/Neg/Recip — 전 백엔드) ②비상수 Div→Recip+Mul
   ③ReduceSum/중간축 ReduceMean → 희소 pointwise conv ④h==1 채널평균 →
   transpose+gpool (conv 출력 채널벡터 폴딩 충돌 회피) ⑤**Transpose 커널**(h=1
   W↔C 실전치, CPU+GPU) ⑥**Relayout 커널**(desc 스트림 항등 재배치 — flatten
   일반화) ⑦W-concat(AddExtraTokens) → 플랫 concat 샌드위치 ⑧외적 Mul(pvec×
   cvec상수) → transpose+1×1 conv ⑨행 브로드캐스트 → relayout 샌드위치+PvecTensor
   ⑩**SwOperand 3종 신설**: PvecTensor(픽셀벡터, grid/packed 2레이아웃)·
   TiledTensor(L|4)·cvec c=1은 스칼라/Tiled(1)로 ⑪**SwConst**(프리로드 상수
   텐서 — 학습된 토큰; 전 실행기 로드 시 주입) ⑫rank-2 [M,K] desc (1,M,K)
   (M>1 Mixer dense — 종전 M=1 가정) ⑬Reshape 일반화: desc 동일→chcopy /
   스트림 동일→relayout / 채널-major 한쪽→relayout+transpose 분해.
   **함정 기록 (재발 방지)**: ⓐReshape는 ONNX NCHW row-major 보존이지 desc(NHWC)
   스트림 보존이 아니다 — C>1·HW>1 텐서가 끼면 transpose 보정 필수
   ⓑ채널벡터(1,1,N)→그리드(N,1,1) Reshape를 chcopy로 내리면 GPU flatten 경로로
   새어 지오메트리 오염 — **desc3 비교로 갈라야** 한다 (16곳 오배선, NO_REUSE에선
   무증상 → 재사용에서만 발병; **debug 빌드로 게이트 1회 돌리면 assert가 잡는다**)
   ⓒ노드 삽입 canon은 반드시 while 루프 (for+캐시 len은 꼬리 누락 — 실제로 당함)
   ⓓ진단 시 act 융합된 텐서(sqrt/recip)는 ORT 중간값과 다르게 보이는 위양성
   ⓔ파일명 긴 tf2onnx 이름은 FNV-1a 절단 (ort_dump.py/dump_all.rs 동일 규칙).
   모델 정찰: MLP-Mixer(GhumMarkerPoser), 입력 [1,146,2] → 출력 52계수, 195노드.
   변환: `models/mediapipe/face/face_blendshapes.onnx` 생성됨(prep_mediapipe).
   **뚫은 것**: ①ReduceSum(L2 norm의 마지막 축 합) → 희소 가중치 pointwise Conv
   캐논 (rank-3 [1,a,k]는 평탄 채널벡터라 쌍합=1×1 conv, canon/reduce.rs)
   ②Activation에 **Sqrt/Neg/Recip 3종 추가** (ai-core apply + ai-cpu apply4 +
   ai-gpu act_expr/ALL 11종 — 전 스위트 그린) ③비상수 Div → Reciprocal+Mul 분해
   (fold_constants.rs — 노드 삽입은 원 위치+바로 뒤, topo 유지).
   ④W축 프리픽스 Slice — **뚫음**: h==1 & start==0이면 레이아웃 [w][cg]에서
   주소식 동일한 연속 프리픽스 = 순수 alias (canon/slice.rs).
   **다음 관문 (여기서부터)**: ⑤Transpose perm=[0,3,2,1] 19개(토큰↔채널,
   [1,64,1,97]→[1,97,1,64], h=1) — 실데이터 2D 전치라 **transpose 커널
   신설(CPU+GPU)** + lower 배선 필요. (w,c)→(c,w) 하나면 된다. GPU는 C4 레인
   재배치(flatten 커널 참조 — ai-gpu/kernels/flatten.rs가 유사 선례).
   ⑥변환 후 게이트:
   `gate_ort_models.rs`(CPU)/`gate_models_gpu.rs`(GPU)에 추가 + MediaPipe 대비
   계수 diff(face-ab.html 확장). 재현: `cargo run --release -q -p ai-convert --
   models/mediapipe/face/face_blendshapes.onnx -o /tmp/bs.sw --name face-bs`.
7. **Horn 피팅 이관 + Expression Stream API** — FaceResult를 {points 478,
   blendshapes 52, pose(quat·t·scale)}로. 웹 face-3d.ts/모바일 items3d와 같은
   FIT_PTS 15점·파워이터레이션. studio 3D 아이템이 첫 소비자(지금은 롤만 반영).
8. ~~FaceTask 입력 GPU화~~ **완료 (2026-08-14)** — 상세는 아래 §"FaceTask 입력
   GPU화" 절.
9. ~~터치업/메이크업 기하 이관~~ **완료 (2026-08-14)** — 위 §"터치업/메이크업
   이관" 절.
10. ~~OneEuroFilter VIDEO 파리티~~ **완료 (2026-08-14)** — §"잔잔한 마감 6종"
    (MediaPipe 원본 확정값 + 좌표계 수리).

**D. P4 웹 연결**
11. ~~VbEngine 3함수 ai-wasm 구현~~ **엔진 쪽 완료 (2026-08-15)** — §"배선 국면"
    참조 (Director + vb_* + web/vb-engine.js 심 + 워커 게이트 PASS).
    남은 것은 **v-ai 리포 쪽 연결만**: pipeline.worker.ts 티어 사다리에
    ai_engine 추가 (엔진 스왑/강등 래치) — v-ai 연결은 보류 중, 착수 전 확인.

**E. 이후 순서**: P5 아바타 replace 합성 모드 → hand/gaze 태스크(웹 크롭 규약
주의 — 게이즈는 비회전 bbox+종횡비 무시) → ai-ffi(vcxrust_ai 표면 재현, ncnn 제거
= 모바일 최대 성능 카드) → 오디오 #6(STFT는 vcxrust_ai에서 이관).

**대기/보류 결정 (건드리기 전 확인)**
- RVM fgr 해상도 한계(모델 해상도로 소프트) → mnv4-RVM 교체 때 입력 해상도와 함께 결정
- 3D 렌더 파리티(three.js vs items3d WGSL) — 사용자가 "나중에" 확정
- webgl2 성능 추격(1.28배) — mnv4-RVM 교체 후 재개 (기확정)
- v-ai 실연결 — 사용자가 보류, 착수 전 확인 필수

---

## P1 진행 기록 (이력·함정 — 새 세션은 위 "다음 작업"만 보면 됨)

**`INTEGRATION.md`가 통합 설계의 단일 진실** (정찰 지도는 메모리 `vai-runtime-map`).
핵심 발견: 웹·모바일은 "다른 구현"이 아니라 **같은 알고리즘의 TS판/Rust판**
(상수까지 동일)이고, 두 벌 다 추론+후처리 12스테이지+합성이 한 덩어리 →
**추론만 교체하면 이원화·모바일 왕복 문제가 안 풀린다. ai_engine이 비디오
파이프라인 전체의 코어가 되어야 한다** (ai-tasks에 Pipeline층 신설).

**P1 1차분 완료 (2026-08-13 심야)**: `ai-tasks/src/video/` — VideoPipeline
(Pipeline층 신설). 구조: params.rs(EffectsPatch JSON 머지: 없음=유지/null=해제/
값=설정) + stage.rs(스테이지 추가 절차 문서화 — 1스테이지=1rs+1wgsl+naga테스트,
dyn 트레이트 없이 고정 배선인 이유 포함) + stages/{preprocess(컴퓨트→모델 입력
버퍼 직결), mask_ingest(softmax|pha+시간EMA 핑퐁), mask_upsample(JBF),
bg_blur(1/5해상도 7탭×6), compose(배경4모드+coverage+spill+edge+lightwrap+
밝기/흑백)}. **프레임당 CPU 픽셀 0, 리드백 0** (uniform 몇십B가 전부) —
ai-gpu-runtime에 `input_storage()` 추가(전처리 컴퓨트가 모델 입력 버퍼에 직접
씀). 바인드그룹은 전부 Res 생성 시 1회(프레임 루프 재생성 0). 데모
`web/demo/studio.html`(+studio.js, wasm exports studio_attach/config/bg_image/
frame) — 헤드리스 `node tools/run_web.mjs demo/studio.html --camera --long`
PASS(원본→블러→단색배경 90프레임), 스크린샷 눈검증 완료.
**실제 제품 에셋 연결 (2026-08-14)**: `make studio-assets` — v-room 가상배경
6종(assets/bg) + v-ai GLB 13종(assets/glb) 복사(gitignore). studio에 배경 프리셋
셀렉트(이미지 배경 모드 실검증 — 스크린샷 OK) + **3D 아이템 오버레이**: three.js
투명 캔버스(#fx)가 FaceTask(face_task_gpu) 478pt를 소비, 유사변환(위치·스케일·롤)
피팅. ⚠ 데모용 임시 스탠드인 — 정밀 Horn 피팅+yaw/pitch는 P3 Expression Stream,
**최종 방향은 vcxrust_ai items3d(wgpu PBR) 이관으로 웹·모바일 렌더러 통일**
(모바일엔 three.js가 없다 — INTEGRATION.md §1.6). 함정 기록: three.js 오버레이는
`setClearColor(0,0)` 필수 — 기본 클리어가 불투명 검정이라 출력을 통째로 덮는다.
FaceTask 입력이 아직 u8 프레임(getImageData)이라 아이템 켜면 CPU 픽셀 경유 —
P3에서 GPU 크롭으로.
**P1 2차분 (2026-08-14)**: ① mask_refine 스테이지(분리 5탭 blur h/v + 엣지 인지
재혼합 — maskBlurPx/edgeBlend/edgeGamma/edgeFeather 웹 상수) — compose·bg_blur가
refined 마스크를 소비. ② **스튜디오 조명**(relight): compose.wgsl에 광원 2개
(target person/bg/all, 방사감쇠², 소프트 롤오프) + EffectsPatch studioLight
(hex 색, null=해제) + studio 조명 토글 — 스크린샷 검증(배경 냉광·인물 온광 확인).
Fullscreen::new_entry(한 wgsl 다중 fs 엔트리) 헬퍼 추가.
**마스크 전멸 사고 수리 (2026-08-14) — 필독**: 새 GPU 전처리에 f16 변형을 얹으며
①f32 파이프라인의 auto-layout으로 만든 바인드그룹을 f16 파이프라인에 재사용 —
**auto-layout은 파이프라인마다 별개 정체성이라 비호환 → 컴퓨트 패스 조용히 무효
→ 입력 버퍼에 아무것도 안 써서 마스크 전멸** ②f16 파이프라인 생성을 adapter
caps로 게이트 — 디바이스 기능(SHADER_F16) 미요청 기기에선 생성만으로 검증 에러.
수리: preprocess에 **명시적 공유 레이아웃** + device.features() 게이트.
게이트 신설 `ai-tasks/tests/vb_pipeline.rs`: 0입력→pha 0 확인 후 파이프라인이
실프레임으로 pha 0.185·합성 전경 18.5%를 만들어야 PASS — **"크래시 없음" 스모크가
마스크 전멸을 PASS시킨 구멍을 막는 진짜 게이트**. 교훈: 파이프라인 변형 추가 시
레이아웃은 반드시 명시적 공유. **studio는 RVM 고정** (사용자 확정 — R11 셀렉터
삭제, R11은 P2 CPU 폴백 티어 전용).

**RVM 매팅 합성 정식 연결 (2026-08-14, 사용자 지시 "정석대로")**: compose가
**fgr×α + bg×(1−α)** 교과서 식으로 합성 — fgr(모델이 복원한 순수 전경색)을 전면
사용, 배경색 오염(머리카락에 옛 배경이 배는 것) 원리적 제거. 스필 억제/엣지
다크닝은 매팅 모드에서도 **파라미터로 유지** (사용자 지적 — 정석은 합성식이지
노브 차단이 아니다; 잔여 스필은 실제 조명 반사·fgr 오차로도 남는다). fgr은 c==3
출력 자동 탐색(fgr_output, 256정렬 확인), 스테이징은 dtype 규약 동일,
pipe.uses_fgr() 게이트 단언 포함. ⚠ 한계+수리 (2026-08-14, 사용자 발견): fgr은
모델 해상도(256×144)라 전면 사용하면 카메라가 크면 인물이 통째로 소프트해진다
(src 1280 수리 후 5배 업스케일로 육안 확인). → **경계 한정 fgr**로 수리:
`fg = mix(fgr, color, smoothstep(cov.y, 1, raw))` — 알파 확실 내부는 카메라
원본(풀해상), 매팅의 실익(배경색 오염 제거)이 있는 경계만 fgr. 원리적 해결
(풀해상 fgr — guided filter 변형/입력 상향)은 mnv4-RVM 교체 때.

**RVM 개통 (2026-08-14)**: studio에 RVM(fp16)이 안 돌던 원인 3개 수리 — ① 마스크
출력을 outputs[0](RVM은 fgr) 하드코딩 → **c==1(pha)>c==2(로짓) 자동 선택**
(pipeline.rs mask_output) ② Logits2 모드 하드코딩 → out c로 자동(1=Alpha)
③ **fp16 모델**: 전처리가 f32 가정 → dt.vec4_bytes()로 f16 스토리지 변형
(preprocess enable f16) + 스테이징 rgba16float/bpr 분기. studio에 세그 모델
셀렉트(R11/RVM) + studio_invalidate(세션 교체 시 바인드그룹 폐기 필수 — 안 하면
이전 모델 버퍼를 문 채 조용히 오동작). ?model=rvm 헤드리스 PASS.
**P1 본편 완료 (2026-08-14)** — 남은 이월분: 프레이밍 중 페이스 이펙트 좌표 보정
(P3 오버레이와 한 묶음), 배경 종횡비 반전 blur-fill 사전합성(세로 프레임 대응 때),
이미지배경 블러 사전 베이크(P2 최적화 카드 — HUD 정직화가 발굴).
3D 렌더 파리티(three.js vs WGSL 차이)는 사용자가 보류 확정. **다음은 B(P2 티어)**.

**픽셀 diff 게이트 — 완료 (2026-08-14): WGSL 스택 = v-ai GLSL 스택 픽셀 등가 증명**
같은 결정적 프레임(640×360)+같은 마스크(256×144)를 양쪽에 주입해 최종 합성 RGBA
채널별 diff — **16개 검사(모드 7종×마스크 2 + EMA 2프레임) 전부 max ≤ 1/255**.
재현: `make vai-gate-assets && node tools/run_web.mjs demo/vb-diff.html`.
- 구조: ai-tasks `process_gpu_mask`(외부 마스크 직주입 — 추론 생략, fgr 강제 off,
  ema 노브. **P2 B티어의 마스크 인제스트 진입점이 이것**) + `gate.rs` GateHarness
  (프레임 업로드+오프스크린 타깃+리드백) + wasm `vb_gate_*` 4종. 비교 상대는
  `web/demo/vai-stack.js` — vendor 사본(video-worker-webgl2.js, gitignore)을
  **blob import로 열어 스테이지 팩토리를 스탠드얼론 조립** (전역 상태 무의존 확인,
  v-ai 셰이더가 바뀌면 사본 갱신만으로 게이트가 추종).
- **게이트가 잡아서 고친 WGSL 불일치 6개** (v-ai가 기준):
  ①JBF의 저해상 마스크 샘플링은 **NEAREST**(v-ai segmentationTexture 기본값 —
  Linear면 경계 전 구간 diff) ②refine 블러H 소스(personMask)도 NEAREST
  ③passthrough(원본 배경)엔 **spill/edge darkening이 없다** — bg_mode 0에서 끔
  ④**단색(#hex) 배경도 v-ai에선 2×2 이미지로 image 스테이지를 탄다** — Color에
  이미지급 상수(spill 0.18/edge 0.24/refine 1.2/0.4/0.58)+light wrapping 적용
  ⑤bg 블러 체인은 **V→H 순서** ×6 (마스크 가중 때문에 순서가 결과에 남는다)
  ⑥이미지배경 자체 블러 신설(compose에 radius=int(mix(1,12,s)) 가우시안,
  **texel=(coverScale/W, coverScale/H)** — applyScaleAndOffset 규약).
  +JBF step/radius는 f64로 계산(JS와 비트 정합 — 1ULP 차로 루프 탭 수가 갈린다).
- **EMA를 게이트에서 다루는 법** (정찰 결론): softmax 경로 minα=0.03은 8비트
  history 양자화로 **참값에서 0.065 떨어진 지점에 고착**(0.5/255/α) — 수렴 대기
  프로토콜은 양쪽 고착점이 달라 못 쓴다. 대신 ①공간 스택은 EMA off+1/255 격자
  사전 양자화 마스크로 1프레임 결정적 비교 ②EMA는 제로 히스토리 1프레임(diff=curr,
  0.3 경계는 격자가 비껴감) + |Δprob|≤0.18 설계 2프레임(동적 α 분기 경계 0.3에서
  마진 0.12 — history 1LSB 차로 분기가 갈리는 픽셀 원천 차단)으로 검증.
- 함정 기록: v-ai `createTexture`는 **현재 활성 유닛에 새 텍스처를 바인딩**한다 —
  하네스에서 스테이지를 프레임 바인딩 뒤에 지연 생성하면 TEXTURE0(프레임)이
  밀려나 JBF가 검정을 가이드로 읽는다 (frame1만 전멸, frame2 정상이 증상).
  v-ai readPixels는 bottom-up(배경 스테이지만 Y 플립) — 행 반전 필수.

**프레이밍(인물 중앙화) — 완료 (2026-08-14)**: v-ai `_updateFraming` 1:1 이식.
- 구조: `framing.rs`(순수 로직 — 목표 계산 headroom 0.15/zoomMax 1.7/하단 0.05/
  가로 1.15 + 데드밴드 0.045/0.055 + 2s 지속이탈 커밋 + EMA 시정수 0.5s + 슬루
  0.35/s + 소실 2s 홀드 후 복귀, 단위테스트 5종) + `stages/bbox.rs`(컴퓨트 리덕션:
  워크그룹 공유 atomic 선축약 → 전역 atomic은 워크그룹당 5개, EMA 이후 mask_lo
  소비, v>0.5 ≡ 웹 u8>127, 1% 문턱) + **리드백 링 2슬롯 20B/프레임**(map_async
  논블로킹, 쓴 버퍼는 회수 후에만 재기록 — v-ai PBO 링과 같은 규율. v-ai는 37KB
  마스크를 내려 CPU 스캔하지만 우리는 리덕션을 GPU에서 하고 20B만 내린다).
- compose 크롭 규약(v-ai 등가, 게이트 검증): **image/단색 = 인물 레이어만**
  (frame·mask·fgr을 crop 좌표로, 배경·relight는 화면 고정) / **원본·블러 = 합성
  전체**(배경·relight도 crop 좌표 — v-ai 캔버스 2D transform 등가). 전체 크롭은
  v-ai가 래스터 후 재샘플(이중 필터)인 반면 우리는 셰이더 단일 필터 — v-ai 자신도
  스테이지가 지원하면 셰이더 크롭 우선(framingInShader)이므로 의도된 개선.
  게이트 참고 수치로도 max 2/255 이내.
- 규율: **invalidate()는 프레이밍 상태를 리셋하지 않는다** (v-ai 주석 — 옵션 조작
  마다 줌이 1x로 튕겼다 재수렴하는 "띡띡" 방지. 리셋은 스트림 파기에서만).
  `set_framing_override`(게이트/디버그용 크롭 강제)·`framing_current`·`last_bbox`
  공개. studio 토글 추가(zoomMax 1.7/headroom 0.15).
- 게이트: vb-diff F상(고정 크롭 — v-ai GL updateFraming과 diff, max 1 PASS) +
  vb_pipeline.rs bbox 게이트(사각 마스크 → 정규화 bbox 오차 <0.01 + 0 마스크 →
  None) + studio 헤드리스(프레이밍 on 스모크 PASS, p50 변화 없음).
- **bbox 백엔드 이원화 (사용자 지시, 2026-08-14)**: 마스크가 있는 곳에서 스캔 —
  GPU 추론(process_gpu) = GPU 리덕션+20B 링 / CPU 마스크(process_gpu_mask, B/C
  티어) = `framing::scan_bbox_cpu`(리드백 0·지연 0). CPU 스캔은 16px 청크
  자동벡터화 형태(비교+any+count 핫루프에 데이터 의존 min/max 없음 — NEON/SSE/
  wasm+simd128). 게이트: rect 정밀(CPU) + 실추론 GPU 리덕션 vs pha CPU 스캔
  교차검증(EMA 차이 2px 허용). **프레이밍은 가상배경 없이도 독립 동작** —
  passthrough 판정에 framing 포함 (INTEGRATION.md §2에 계약 추가).
- ⚠ 실인물 눈검증(줌인 활강 UX)은 사용자 몫 — 카메라로 studio 프레이밍 토글 확인.

**mirror/degree — 완료 (2026-08-14)**: 역할 분담이 v-ai 그대로 —
- **프레임 변환은 호스트 몫** (추론 **전** 2D 캔버스 preprocess, v-ai
  `_prepareSourceElement` 등가 — translate(center)→scale(-1,1)→rotate(rad)→draw.
  좌표계 계약: 랜드마크·마스크가 화면 좌표와 일치). studio tick()에 이식,
  미러 체크박스+회전 셀렉트(0/90/180/270).
- **엔진 몫은 이미지 배경 좌표 보정**만: compose에 2×2 행렬(열우선, 쿼터턴+미러
  특례 부호반전)·aspect 보정(updateAspectComp)·contain 스케일(회전 콘텐츠가 화면
  안에 — transformScaleMultiplier가 effective scale/offset/texel에 반영) —
  updateTransform/applyScaleAndOffset 1:1, 계산은 f64(JS 정합). EffectsPatch
  mirror(bool)/degree(도, %360) 추가. 단색(#hex)도 image 스테이지 규약이라 동일
  경로(색상 불변이라 시각 영향 없음).
- 게이트 T상: cover-wide(1040×450)/cover-tall(800×630)/mirror/90/180/mirror-90 —
  전부 **max=0**. 이걸로 "배경 cover 크롭 수학 게이트" 항목도 소화 (S상은 종횡비
  동일이라 크롭 항등이었음). ⚠ cropFactor≥1.6은 v-ai가 blur-fill 사전합성
  (background-fit.js)을 타는데 미이관 — 게이트 픽스처도 1.6 미만으로 제한.
- **studio 화질 함정 (사용자 발견, 2026-08-14)**: ①src 캔버스(엔진 입력)가
  640×360으로 남아 카메라 1280×720을 줄였다 출력에서 재확대 → src를
  video.videoWidth/Height로 동기화 ②출력 캔버스 백킹=CSS 크기면 레티나(dpr 2)에서
  브라우저가 2배 업스케일 → **백킹 1280×720 + CSS 640×360** (디바이스 픽셀 1:1).

**FaceTask 입력 GPU화 — 완료 (2026-08-14)**: 마지막 프레임당 CPU 픽셀 경유
(getImageData 720p ~3.7MB 리드백 + JS RGB 재패킹 + wasm 복사) 제거.
- 구조: `ai-tasks/src/detect/gpu/` 신설 — `letterbox.rs`+`crop.rs`(커널별 1rs+
  1wgsl, `shaders/`) + `frame.rs`(FrameTex — Rgba8 상주 프레임, 웹 임포트/네이티브
  write_texture) + `mod.rs`(**GpuPre** = 커널+프레임 홀더, KernelPair 공통 뼈대).
  커널이 `input_storage()`(모델 입력 버퍼)에 **직결** — vb preprocess와 같은
  f32/f16 명시 공유 레이아웃 규약. `GpuSession::input_storage/detect_uploaded`
  신설, `FaceTask::process_tex(ctx, pre, view, det, lm, w, h, t)` 드라이버 추가
  (레터박스는 재획득 프레임만, 크롭은 매 프레임 — 남는 CPU 전송은 uniform 몇십 B
  + 소형 출력 리드백뿐).
- **샘플링은 textureLoad 수동 bilinear** (HW 샘플러 금지 — 8비트 가중치 양자화로
  ~4e-3 벌어진다): CPU 판(letterbox_u8_rgb/crop_u8_rgb)과 f32 동일 경로라
  커널 게이트가 max 7.9e-6. 두 규약이 다름을 그대로 복제 — 레터박스=픽셀 중심
  +패딩 lo, 크롭=warpPerspective 코너 정합+replicate.
- 바인드그룹은 **디스패치마다 생성** (프레임 텍스처·세션이 호출 사이 바뀔 수
  있고 낡은 바인딩은 이전 모델 버퍼에 조용히 씀 — studio_invalidate 사고 급).
  프레임당 ≤2회라 µs대.
- wasm: `face_task_tex(task, det, lm, canvas, t)`(스탠드얼론 — FrameTex 임포트) +
  `studio_face(task, det, lm, t)`(**studio 파이프라인 프레임 텍스처 공유** —
  `VideoPipeline::frame_view()` 신설, 재임포트 0). studio.js는 getImageData를
  진짜 픽셀 소비자(게이즈 CNN 비전 틱 10fps·광원 프로브 8틱)로만 축소 —
  probe는 JS가 페이싱을 소유하므로 `probe_scene_light_rgb_now`(무스로틀 판)
  신설해 이중 스로틀(8×8=64틱) 방지.
- 게이트: `tests/tex_input.rs`(커널 단독 CPU 대조, 모델 불필요 — 레터박스
  128²[-1,1]/192²[0,1]·크롭 회전+경계 걸침, max<1e-4) +
  `tests/face_task_tex.rs`(process_tex vs process_gpu **478pt 0.000px** + 트래킹
  계약 + 재검출) + face-ab.html `tex` 스테이지(MediaPipe 대비 0.30px = 기존 경로와
  동일, **lm-tex-vs-cpu 0.000px** tol 1.0) + studio 헤드리스·vb-diff 무회귀.
- 잔여: HandTask·게이즈 CNN 크롭의 GPU화(커널은 범용 — lo/hi 인자화 완료, 게이즈는
  비회전 bbox+ImageNet 정규화라 **별도 커널 필요**), items 광원 프로브 GPU화
  (마지막 getImageData 소비자 — 16×16 샘플이면 다운스케일 리드백로도 충분).

**배선 국면 완료 (2026-08-15) — ai-ffi(모바일)·ai-wasm(웹) 양쪽, 사용자 지시
"동일 재현이 아니라 우리 구현을 편하게 쓰는 표면"**. 정찰 2건(vcxrust_ai C
ABI/JNI 전체 표면·v-ai VbEngine 티어 계약 — 파일:라인 대조) 후 3층 구현:
- **`ai_tasks::Director` 신설 (src/director.rs)** — studio.js에서 검증한
  오케스트레이션의 Rust화: 세그 파이프라인 + FaceTask(process_tex) + 터치업/
  메이크업 fx + 3D 아이템(크롭 동행·프로브 8틱) + 손 제스처(detectFps 페이싱·
  FIFO 16 큐) + 집중도(비전 틱·num_faces=2) + **지연 로드**(모델 바이트 주입 →
  켜질 때 세션 생성). 단일 JSON = EffectsPatch + 태스크 키(faceItems/
  handDetection/focusDetection — vcxrust effects JSON과 같은 필드명). 판정 3종
  `needs_render/tasks_active/passthrough` + `wants_pixels(t)`(호스트 픽셀 추출
  최소화). 결과는 우리 타입 JSON(`focus_json` 7상태 전체·`poll_gesture_json`).
  A티어 `frame`/B티어 `frame_mask`(process_mask_nogpu — 세그 모델 불필요)/
  analyzer-only(target 없이 태스크만 — vcxrust 고속경로 등가). `detach`=웜(세션
  유지)/`reset`=세션 드랍(모델 바이트 유지). **바인딩에 분기 없음 규칙의 이행** —
  ffi/wasm은 접착만. 게이트 tests/director.rs: 실모델 e2e — FOCUSED 도달·합성
  검증·웜 재가동.
- **ai-ffi 완성**: 표면 = set_video_stream_info(seg .sw) + set_face/hand/gaze_
  model_info(gaze는 bs nullable) + set_item_model_dir(GLB fs 지연 로더) +
  update_effects_config(단일 JSON — VideoOptionsC 이원 채널은 **재현 안 함**) +
  set_background_image(RGBA 바이너리 — base64 왕복 제거) + set_focus_layout +
  render_mask(음수 치수 허용·passthrough면 변환도 안 함·analyzer-only 무수정) +
  get_focus_state/poll_hand_gesture(vcx_string_free) + destroy=리셋(모델 유지) +
  오디오 fe_* 5종(16k/48k `{dir}/fe16|fe48/graph.json+weights.bin`, 미지원
  레이트는 480 passthrough). **VbResult Failure=-1**(vcxrust와 동일 값 — 호스트
  `==-1` 검사 보호). **JNI 레이어**(java_api.rs, cfg android — C 표면을 그대로
  감싸는 무로직 래퍼, Kotlin 시그니처는 파일 헤더): 안드로이드는 C ABI 대신 JNI
  심볼 소비(vcxrust와 같은 분리), aarch64-linux-android 타입체크 통과. **cbindgen
  헤더** `make ffi-header` → crates/ai-ffi/include/ai_engine_ffi.h. 게이트:
  tests/ffi.rs extended_surface_smoke — C 표면으로 집중도 FOCUSED 실추론·음수
  치수·오디오·웜 destroy 재가동 (표면 테스트는 SURFACE_LOCK 직렬화 — 전역
  STATE 공유).
- **ai-wasm vb 모듈 + 심**: `vb.rs`(OffscreenCanvas 서피스 + ImageBitmap 무복사
  임포트 — web-sys 기능 추가) exports: vb_attach/config/model/glb/bg_image/
  layout/frame/frame_mask(B티어)/analyze(무합성)/focus_state/poll_gesture/
  detach(웜)/passthrough·needs_render·wants_pixels. **`web/vb-engine.js`** =
  VbEngine 3함수 심(v-ai pipeline.worker 계약 그대로): passthrough 제로카피,
  **출력 확보 후에만 입력 close**(먼저 닫고 throw하면 워커 복구 경로가 detach
  비트맵 전송 — 원본 대조로 확정한 함정), 모델 로드는 config에서
  fire-and-forget·로드 중 passthrough, VBOptions 번역(blur/brightness/grayscale
  ÷100·degree 360→0·base64 배경→vb_bg_image·faceEffects→faceItems+GLB fetch·
  메이크업 룩 프리셋→기본 룩 매핑 — v-ai MAKEUP_LOOKS 주입 자리 표시).
  **게이트 web/demo/vb-engine.html + vb-worker.js — 진짜 Worker 경계에서 4검사
  전부 PASS**: passthrough 입력 동일 객체(제로카피 증명)·render(배경 합성+크기
  보존)·focus FOCUSED(얼굴 y4m)·warm destroy→재합성.
- 정찰 확정 사실(실연결 때 필요): v-ai "티어"는 2층 — 세그 모델 사다리(gl-rvm…
  — 우리 심과 무관)와 **VbEngine 엔진 모듈**(우리가 붙는 자리, lite가 최소
  선례). 입력은 항상 ImageBitmap(VideoFrame 소유는 워커), 반환 비트맵은 즉시
  transfer라 재사용 금지. destroy는 엔진 스왑 때도 불림·웜 재사용 전제.
  v-ai 쪽 유일 수정 지점 = pipeline.worker.ts 사다리(엔진 셀렉션에 avatar/3D
  강제 webgl2 조건 추가). 이건 v-ai 연결(보류) 때.
- **모바일 아티팩트 빌드 개통 (2026-08-15)**: `make build-android`(cargo-ndk,
  arm64-v8a/armeabi-v7a/x86_64 → target/jniLibs) + `make build-ios`(3타깃 →
  lipo 심 유니버설 → AiEngineFFI.xcframework, cbindgen 헤더 동봉) —
  vcxrust_ai README 레시피의 ai-ffi판. **실측 (vs vcxrust_ai 출하본)**:
  Android .so = arm64 **8.4 vs 21.2MB**, v7a 5.7 vs 14.1, x86_64 9.1 vs 25.8
  (**ABI당 ~2.5배 작음** — ncnn C++ 스택 부재). iOS는 .a(링크 전 아카이브)가
  68.4 vs 66.0MB로 동급으로 **보이지만 착시** — 링크 실험으로 실증: 같은 조건
  (-Os -dead_strip, 동일 수출 심볼 5개 사용, 미니 바이너리)으로 디바이스 .a를
  링크하면 **ai_engine 6.8MB vs vcxrust 15.6MB(+MoltenVK 별도) = 2.3배 작음**
  (Android와 일치). .a가 안 준 이유: 아카이브는 링커가 버릴 데드코드까지 전부
  담고, Rust 모노모피제이션이 링크 전 오브젝트 볼륨을 부풀린다 — 실배포 크기는
  링크 후 증가분이 기준. 모델은 라이브러리에 **미포함**(호스트 조달 규약 —
  전 기능 .sw 합계 ~36MB, VB만이면 RVM 7.6MB; vcxrust resources는 16MB).
  추가 다이어트 여지: 모바일 프로필 lto/opt-level=z 미적용 (wasm 다이어트
  카드와 한 묶음).
- 잔여 카드: YUV↔RGB CPU 왕복(ffi)의 GPU 상주화, HandTask/게이즈 크롭 GPU화,
  워커별 feature 빌드(wasm 다이어트 — P4 실연결 때), JNI 클래스명은 앱에 맞춰
  리네임, 모바일 릴리스 프로필(lto/opt-z) 튜닝.

**잔잔한 마감 6종 — 완료 (2026-08-14)**:
- **OneEuro VIDEO 확정** (smoothing.rs): MediaPipe face_landmarks_detector_graph
  원본 대조 — one_euro **min_cutoff 0.05 / beta 80 / d_cutoff 1.0** (스트림 모드
  + num_faces==1일 때만 — 우리도 lm 1명). **좌표계 버그 동시 수리**: MediaPipe는
  denormalize(px) 후 필터인데 정규화 좌표를 그대로 미분하면 속도가 프레임 폭배
  작아져 과잉 랙 — apply가 (img_w, img_h)를 받아 축별 px 속도로 계산,
  object_scale도 MediaPipe 기본(lm bbox px (w+h)/2) 내부 계산으로 변경.
- **강등 p90 창** (studio.js): 단발 2연속 → **v-ai _recordCycle 규약** — 1s
  gpu_sync 샘플 ×10 = 창, 첫(웜업) 창 폐기, 창 p90>66ms 2연속이면 강등
  (단발 스파이크 내성). v-ai 원본은 120프레임 창 — 우리는 비동기 GPU라 매 프레임
  sync가 불가해 1s 샘플로 대체 (HUD gpu 샘플=큐 포함이 옳다는 측정 규율 유지).
- **프레이밍 중 아이템 좌표 보정** (P3 이월 해소): ItemsOverlay::set_view_crop
  (framing scale/cx/cy) — draw에서 compose 크롭의 역변환으로 화면 좌표화
  (z도 1/s — 줌만큼 아이템 스케일 동행). studio.rs overlay가 framing_current()를
  매 프레임 전달.
- **C티어 ensure 분리**: Res의 모델 종속부를 **GpuLeg(Option)**로 분리 —
  `process_mask_nogpu`/`with_frame_texture_nogpu` 신설 (**세그 세션 전혀 없이**
  외부 마스크 합성 스택 전체 가동 = B/C 티어가 RVM 7.6MB 로드+컴파일 생략).
  EMA(mask_lo) 해상도는 세션 있으면 모델, 없으면 외부 마스크 크기. 기존 경로
  (process_gpu/process_gpu_mask+세션)는 동일 동작 — vb-diff·ffi 무회귀.
  게이트 `tests/vb_nogpu.rs`: 모델 파일 0개로 절반 마스크 합성 — bg 순색/fg
  프레임 픽셀 일치 확인. 잔여: 진짜 C티어(소프트 합성 — GPU 완전 부재)는 별개.
- **세로 blur-fill 배치 결정 + 웹 이식**: 웹은 **호스트 몫** — v-ai
  background-fit.js(캔버스 2D 1회 사전합성: contain 본층 + 미러 띠 + 2패스
  선명/블러 — 정수 좌표·PAD 오버스캔 seam 함정 검증본)를 studio.js에 이식,
  uploadBgBitmap에서 cropFactor≥1.6이면 자동 적용 (결과 비율=프레임 비율이라
  엔진 cover 수학 1:1 통과 — 엔진 변경 0). 모바일은 ai-ffi 세로 대응 때 Rust
  포팅 (로드맵 E).
- **wasm 다이어트 (측정 완료 — 실행은 카드로)**: 출하 wasm 2.45MB. 크레이트
  분해(디버그 심볼 빌드 4.15MB 기준, twiggy — `CARGO_PROFILE_RELEASE_DEBUG=1
  cargo build --target wasm32-unknown-unknown + twiggy top`): **image 디코더
  (png/jpeg/webp — GLB 텍스처용) 414KB/10%**, **rustfft (오디오 STFT) 250KB/6%**,
  ai-* 크레이트 합 ~780KB/19%, std/serde/wgpu-naga ~570KB, 섹션·데이터 1.49MB
  (이 중 bindgen 커스텀 섹션 541KB·이름 432KB는 출하본에서 제거됨, .rodata
  508KB — WGSL 템플릿·앵커 포함). **결론 (우선순위 순 카드)**: ①**워커별
  feature 빌드** — video/vision 워커에 rustfft(오디오 전용)와 image 디코더
  (items3d 전용)는 사장 무게: audio·items feature 분리로 워커당 ~수백 KB 절감
  ②**GLB 텍스처 디코드 외부화** (createImageBitmap→RGBA 주입 — image 크레이트
  통째 제거) ③SwOp serde Deserialize 71KB — .sw 파서의 serde 탈피는 이득 대비
  작음(보류). 실행 시점: P4 워커 조립 때 (빌드 매트릭스가 그때 생긴다).

**자원 절약 감사 + studio 수리 4건 (2026-08-14 심야, 사용자 지적)**:
- **`studio.html?audit=1` 신설** — off 시 진짜 안 도는지를 엔진 프레임 카운터
  (model_stats_h.frames — JS 추정이 아니라 세션이 실제 추론한 횟수)로 단언하는
  실루프 감사. 스케줄: 30f 전부 off(부가 모델 **로드조차 없어야**) → on(집중도+
  아이템+터치업+메이크업+프레이밍) 70f → 전부 off 후 60f **델타 0 단언**.
  얼굴 카메라(`--video=web/models/mediapipe/face_256x144.y4m` —
  `tools/make_face_y4m.py`로 실얼굴 픽스처에서 생성, 전 세션 스크래치 유실분
  도구화)로 **PASS: on det=lm=60/gaze=bs=9 → off 전부 동결, R11 미로드**.
  ⚠ seg 카운터는 판정 제외 — frames는 finish_frame(동기화) 때만 기록되는데
  studio 세그는 제출만 하는 설계라 항상 0 (가동 증거는 렌더 출력).
- **R11 지연 로드**: 시작 시 무조건 로드하던 것을 B티어가 실제 필요할 때
  (셀렉트 b / auto 강등 확정 / 헤드리스 B스모크)만 — perf 측정에서 "R11로
  재나?" 오해의 원인이기도 했다 (**perf·기본 측정은 전부 RVM GPU**, R11은
  폴백 전용이 사실이나 init 로그가 오해를 유발). 감사의 r11-eager 단언이 지킨다.
- **?perf=1 "화면 멈춤"은 설계** (페이싱 프로토콜 종료 후 루프 정지) — 단
  결과가 콘솔에만 있어 멈춤처럼 보였다 → 상태줄에 스테이지별 p50/p90 표기 추가.
- **스크립트 스모크 게이트**: 자동 토글 시나리오(blur60·배경 전환 등)가
  인터랙티브에서도 돌던 것을 `navigator.webdriver`(자동화)에서만으로 제한.
- **ensureFaceTask/ensureGaze 이중 로드 레이스 수리** (audit이 잡음): item·focus
  동시 켜기 → face 이중 로드 → 두 번째가 face를 덮어 num_faces=2가 유령 핸들에
  적용(det=1로 발현) + 세션 누수. in-flight 가드 + 로드 완료 시 스위치 상태 반영.
- **gaze CNN 통계 수리** (엔진): GazeTask CNN 블록이 finish_frame을 안 불러
  stats.frames가 0 — 리드백이 이미 동기화라 비용 0으로 기록만 추가 (감사·강등
  판정의 입력이 되는 카운터).
- **강등 오발 사고 수리 (사용자 발견 — 필독)**: 배경+블러+메이크업을 켜면
  auto 티어가 **멀쩡한 GPU를 B로 강등**시켰다. 원인 = 판정 입력이 "제출→큐
  소진" 비동기 샘플이라 **큐에 쌓인 프레임까지 포함** — 120Hz rAF(8.3ms)에
  실비용 15ms면 큐 2~3장이 상시라 샘플이 3~6배(60~100ms)로 부풀어 66ms 문턱을
  허수로 돌파, "승격 없음" 규약 때문에 B에 갇혔다 (어제 "64.3ms 강등 코앞"
  정정과 같은 허수의 실사고화). 수리: HUD/강등 샘플을 **페이싱 방식**으로 교체 —
  1s마다 큐 배수 → 한 프레임 제출 → 완료 대기 (= ?perf=1과 같은 "한 프레임
  실비용" 정의, 샘플 프레임만 ~2프레임 멈칫·비가시). 실비용 15~16ms는 66ms를
  못 넘는다. + 수동 티어 A 선택 = 강등 상태 리셋(탈출구). 교훈: **강등 판정
  입력도 실비용이어야 한다** — 큐 포함 지연은 rAF 주사율에 비례해 부푸는
  양이라 문턱 비교에 못 쓴다 (v-ai webgl2는 readPixels 동기라 cycle≈실비용이
  성립했던 것 — WebGPU로 오면서 전제가 깨졌던 자리).

**집중도 마감 3종 — 완료 (2026-08-14)**: v-ai focus-tracker 정찰(파일:라인 대조)
후 1:1 이식.
- **다중 모니터 분류기** (`gaze/state.rs`): MonitorInfo/ScreenLayout(가상
  데스크톱 px + yawDeg 오버라이드, targetIndex=**배열 위치**·isCurrent 기준) +
  `matchMonitorByGazeDelta` 이식. 규약: ①매칭이 온타깃 박스 검사 **앞** — 박스 안
  시선도 인접 모니터가 가로챈다 ②단위 혼용이 설계(시선 도° · 모니터 px/|px| →
  proj는 도) ③인접(4px 이음새+겹침) 모니터는 정렬도 비례 완화 k→0.5 (24→12°),
  yawDeg 모니터는 ×(1−min(|yaw|,60)/60×0.4) ④OTHER_MONITOR는 LOOKING_AWAY의
  리라벨 — 히스테리시스 공유, 이미 이탈 상태면 즉시 전환 ⑤EYES_CLOSED도
  monitor_index 유지(감김 전 분류) ⑥score엔 비집중으로 쌓임 ⑦baseline은 전역
  1개, **타깃 모니터 변경 시 리셋**(GazeTask::set_layout). API: wasm
  `gaze_layout(task, json|null)` — 레이아웃 조달(getScreenDetails)은 호스트 몫.
- **MULTIPLE_FACES**: FaceTask `set_num_faces(2)` — MediaPipe FaceLandmarker의
  "tracked<num_faces면 검출 지속" 규약 그대로 **트래킹 중에도 매 프레임 디텍터**
  (count=post-NMS 검출 수 근사 — 웹은 landmarker 통과 수지만 우리 lm은 1명).
  결과에 faceCount 노출, studio는 집중도 켤 때만 2 (끄면 1 — 디텍터 비용 제거).
  분석·랜드마크는 최고점 1명 유지 (웹 analyze.ts도 faces[0]만).
- **blink 블렌드셰이프 절반** (`face/blendshapes.rs`): **입력 규약을 MediaPipe
  원본(face_blendshapes_graph.cc + landmarks_to_tensor_calculator.cc)에서 확정**
  — 478→146 서브셋(kLandmarksSubsetIdxs, 홍채 468~477 포함 = refined 메시 전제),
  좌표는 **프레임 px (x×W, y×H), 센터링·스케일 없음** (모델이 내부 L2 정규화),
  출력 52계수 중 9/10 = eyeBlinkL/R. GazeTask가 비전 틱마다 bs 세션(옵션) 추론 →
  `(bsL≥0.55 ∧ bsR≥0.55) ∨ EAR` (웹 blink.ts). wasm gaze_task_gpu에 bs 핸들
  추가(**0=없음** — EAR만), 게이트 헬퍼 bs_input_from_landmarks.
  **게이트가 규약을 실증**: face-ab bs 스테이지 — 우리 lm→우리 bs 모델 vs
  MediaPipe faceBlendshapes 52계수 diff **max 0.0717 / blink 0.0175** (tol 0.12;
  규약이 틀리면 크게 발산하는 구조). convert-mediapipe에 face_blendshapes 변환·
  웹 복사 추가됨.
- 잔여: 다중 모니터는 실멀티모니터 눈검증(사용자)과 v-ai 호스트 배선(P4)만.

**터치업/메이크업 이관 — 완료 (2026-08-14)**: vcxrust_ai face/{touchup,makeup}.rs
→ `features/face/{touchup,makeup}.rs` (래스터라이저 근사-원문 이식 + 옵션 타입
동파일화), 소비자는 vcxrust pack 셰이더 → **compose.wgsl 통합**.
- 파라미터 계약: EffectsPatch에 `touchUp {enabled, strength}` / `makeup {enabled,
  intensity, lip/blush/shadow}` (camelCase — vcxrust update_effects_config와 동일
  스키마라 ai-ffi도 공짜로 파리티).
- 데이터 흐름: FaceTask 랜드마크(정규화) → `VideoPipeline::update_face_fx` —
  CPU 래스터라이즈(128² 오버레이, 수십 µs) → write_texture(R8 터치업 + RGBA8
  mul/over) → compose uniform(tu_map/tu_par/mk_map, 전부 0=off). studio는
  studio_face가 process_tex 직후 자동 호출 (1프레임 지연 — 128² 소프트 마스크라
  비가시).
- 셰이더 수식 (vcxrust pack 1:1): 터치업 = 마스크 가중 3×3 가우스(1-2-1,
  stride=face_w×0.022 clamp[1,8]px) ×1.02 리프트, α=0.62×strength — 단 luma가
  아니라 **RGB에 적용** (웹 drawTouchUp이 RGB 블러; vcxrust만 YUV 사정으로 Y만).
  메이크업 = multiply(base×mix(1,color,α)) → source-over(mix). 오버레이 샘플은
  128² textureLoad bilinear(스트레이트 알파).
- 크롭 보정 불필요의 근거: pack은 화면좌표 블러라 blur_px/crop_scale 보정이
  필요했지만 우리는 프레임 좌표(cuv) 샘플이라 프레이밍 줌이 자동 반영.
- 게이트: 래스터라이저 단위 5종(다크 헤일로 가드 포함 — 이관 그대로) +
  **tests/face_fx.rs**(합성 랜드마크 + process_gpu_mask ema=off: 립 밴드 diff
  36레벨 / 터치업 피부 diff / 얼굴 밖 0 / update_face_fx(None) 비트 복원 —
  "크래시 없음" 스모크가 못 잡는 uniform 오배선을 픽셀로 잡는다). 함정: 립은
  even-odd **밴드**만 칠해진다 — 입 중앙을 찍으면 diff 0이 정상.
- studio: 터치업/메이크업 체크박스(기본 룩 = vcxrust 테스트 룩), 헤드리스
  마일스톤에 스모크 포함. 실카메라 룩 튜닝은 사용자 몫.

**실행 계획 (2026-08-13, INTEGRATION.md §3)** — 데모 HTML 주도
(사용자 제안): `web/demo/studio.html` = 제품 UI 미니어처, 단계마다 효과를 켜며
게이트+눈검증. P1 VideoPipeline(vcxrust_ai WGSL 이관) → P2 티어 정책(§1.5
매트릭스: 추론×합성 2축, A=GPU+GPU / B=CPU추론+GPU합성 / C=lite — "GPU 안 되면
효과 전부 끔"의 이분법 제거) → P3 얼굴 스택(blendshape+Horn+터치업/메이크업 =
**Expression Stream**, 아바타 전제) → P4 웹 연결(VbEngine) → P5 아바타
스파이크(person-replacement 모드) → P6+ Gaze/Hand/ai-ffi/오디오.
아바타(사람 전체 교체, 표정 포함)는 §1.6 — 지금부터 계약에 반영.

**GPU↔CPU 왕복에 대해 (사용자 질문, 2026-08-13)**: 소형 텐서 리드백(det 출력
~60KB → ROI 결정, lm 출력 6KB)은 **구조적으로 불가피** — ROI는 다음 디스패치를
결정하는 제어 흐름이라 MediaPipe GPU 파이프라인도 똑같이 내린다. 반면 지금
스파이크에서 **이미지가 CPU를 왕복하는 것**(레터박스·회전 크롭을 CPU에서 하고
결과를 업로드)은 단순화일 뿐: 프레임을 GPU 텍스처로 상주시키고(카메라 임포트
경로는 이미 있음 — present.rs copy_external_image_to_texture) 레터박스/회전
warp를 커널로 만들면 업로드는 프레임 1회로 줄고, 트래킹 중엔 디텍터 생략이라
사실상 크롭 커널 1개+6KB 리드백만 남는다. **vision 워커 조립 때 GPU 상주 크롭
커널로 전환** — 지금은 파리티 게이트가 우선이라 CPU 크롭을 임시 허용했지만,
**⚠ 타깃은 M1이 아니라 저사양 PC다 (사용자 강조, 메모리 target-hardware-lowend)**:
구형 iGPU·저가 노트북에선 업로드·왕복·wasm 픽셀 루프가 실비용이므로 GPU 상주
전처리 커널은 "나중에 하면 좋은 것"이 아니라 vision 워커 조립의 필수 항목.
세그(가상배경)도 같은 그림: 합성·카메라 프레임은 **이미 GPU 직행**
(infer_and_present — copy_external_image_to_texture 임포트, 마스크 GPU 상주,
Compositor가 캔버스에 합성)이고, **모델 입력(축소 f32)만 CPU 경유** — GPU
전처리 커널(리사이즈+정규화) 하나면 프레임당 CPU를 오가는 픽셀이 0이 된다.
합성 결과를 트랙으로 내보내는 건 canvas.captureStream() (CPU 리드백 없음).

**fastenhancer(#6, 오디오)는 그다음**: 선행조건(ai-cpu)은 풀렸지만 DFT·1D
conv·복소 = 새 커널 계열이라 스파이크 단위가 크고, #4는 이미 변환해둔 5모델을
기능으로 바꾸는 마지막 조각이라 레버리지가 더 크다. (#3 잔여인 tflite 직접
임포터도 #4 뒤로.)
v-ai 정찰 전체 지도(런타임·티어·플래그·file:line)는 메모리 `vai-runtime-map` 참조.

---

## 확정된 로드맵

| # | 항목 | 상태 |
|---|---|---|
| 0 | `ai-tasks` 크레이트 (공개 API 본체) | **진행 중 — 아래 참조** |
| 1 | 폴백 트리거 3종 | **엔진 쪽 완료** (2026-08-13) — v-ai 연결만 보류, §1 참조 |
| 2 | `ai-cpu` (SIMD128/NEON + 스레드) + R11 vs tflite-simd A/B | **완료** (2026-08-13) — 브라우저 A/B + 커널 스프린트로 **tflite-simd 동률**, §2·§2.5 |
| 3 | `Reshape` canon + `PRELU` + `MAX_POOL_2D` | **완료** (2026-08-13) — 5모델 전부 개통+ORT 게이트+벤치, §3.5 |
| 4 | 랜드마크 스파이크 (face_detector → 좌표 diff) | **face 완료** (2026-08-13) — FaceTask+게이트 lm 0.30px PASS, §4. hand 복제·게이즈 체인 남음 |
| 5 | 파이프라인 로직 (ROI 트래킹 / 회전 정규화 / OneEuroFilter / Horn 피팅) | **대부분 #4에서 선반영** — 남은 것: OneEuroFilter VIDEO 파리티, Horn 피팅, 게이즈 페이싱 |
| 6 | 오디오 (DFT / 1D conv / 복소) | **완료** (2026-08-14) — fastenhancer 16k/48k, SNR 125dB+, **wasm 0.85 / native 0.64ms — ORT wasm 1.39배 우위**, §6 |

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
- `GpuSession` (`gpu_session.rs`, 구 Segmenter — §4에서 리네임) — 모델 수명 + 프레임 루프 +
  **프레임타임 링버퍼(p50/p90)**. `ai-wasm`의 `MODEL` thread_local이 이걸 담는다.
  전 export 경로가 이걸 통과한다. 다중 상주는 `Pool<GpuSession>`(핸들 API, §4).
- `TaskError` — `NoGpu` / `DeviceLost` / `Runtime` / `Gpu` 구조화 (폴백 판정 근거).
- `model_stats()` wasm export 추가 (p50/p90/last/frames).

**남은 것**
- `GpuSession::process()` 하나로 묶기 — 지금은 `upload → infer → (호스트 합성) → finish_frame`을
  바인딩이 순서대로 부른다. 합성이 플랫폼 서피스에 걸려 있어 아직 안 묶었다.
  `ai-ffi` 만들 때 같은 순서를 또 쓰게 되면 그때 묶는다.
- `ai-ffi` 뼈대 (C ABI, `repr(C)`, opaque handle). 모바일 실연결은 나중.

**지켜야 할 규칙 (ARCHITECTURE.md에도 박아둠)**
> `ai-wasm` / `ai-ffi`에는 분기(`if`)가 없다. 분기가 생겼다면 로직이고 `ai-tasks`로 내려간다.
> 플랫폼마다 진짜 다른 것만 바인딩에 남긴다: 서피스 획득, 프레임 임포트, 스레드 모델, 모델 바이트 조달.

**이름 규칙 (2026-08-13, 사용자 확정)**: `GpuSession`/`CpuSession` = 백엔드에 로드된
**모델 1개 인스턴스**(수명+프레임 루프+통계) — ORT `InferenceSession` 관례이자
실행기(`ai_cpu::Model`·`ai_gpu_runtime::Model`)와의 위상 구분. `~Task` 이름은 **파이프라인 레벨
타입에 예약** — 모델 여러 개+전·후처리+트래킹을 묶은 도메인 기능 단위
(예정: FaceTask = det+lm+ROI트래킹+OneEuroFilter, SegmentTask 등, 로드맵 #4-3/#5).
ai-tasks 구조 = 세션(GpuSession/CpuSession/Pool, 재료) + 태스크 로직(detect/,
앞으로 roi·tracking·filter) + 태스크 타입(예정).

---

## 1. 폴백 트리거 3종 — 엔진 쪽 끝 (2026-08-13)

1. ✅ **device lost 구독** — `GpuContext::new()`가 `set_device_lost_callback` 등록
   (웹은 `device.lost` 프라미스 경로라 `Error::from_js` panic 이슈와 무관 — 소스로 확인).
   `ctx.lost_reason()` 폴링, `GpuSession` 프레임 경로 3곳에서 체크 → `TaskError::DeviceLost`,
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
  가중치 재패킹 + last_use 슬롯 재사용 계획), `model.rs`(구 exec.rs — 프레임 중 분석·할당 없음),
  `kernels/`(conv=브로드캐스트 GEMM 4px×8cout, dw=채널 벡터화, 나머지 콜드 스칼라).
- alias/concat 융합은 채널 스트라이드 뷰로 무복사. 상태 ping-pong은 프레임 시작 swap.
- 스레드: rayon 행 밴드(네이티브 전용, `set_threads`). wasm 스레드는 COOP/COEP 필요 — 미착수.
- `ai-tasks::CpuSession`(구 CpuSegmenter — §4에서 리네임) + wasm export:
  `load_model_cpu/infer_frame_cpu/model_stats_cpu/model_io_cpu`.
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
이론하한, `ai_cpu::Model::infer_profiled`). ② **wasm — `node tools/profile_web.mjs
'demo/cpu-ab.html?only=ours' --ops`** (`ai_cpu::Model::bench_steps` = 스텝당 N회 합산으로
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
   오버리드를 crate 전체에서 안전화 — model.rs 참조).
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

## 4. 랜드마크 스파이크 1·2단계 — 완료 (2026-08-13)

**결과**: 다중모델 핸들 API + face_detector 후처리 전체가 CPU·GPU 양 백엔드에서
돌고, **MediaPipe FaceDetector(wasm, 같은 tflite 가중치) 대비 박스 diff 1.10px /
키포인트 1.27px / score 0.007** (256×144 프레임, 게이트 tol 3px PASS — CPU와 GPU가
동일 수치). 3모델 동시 상주(det+lm+게이즈, WebGPU) 스모크 포함.

**구현**:
- `ai-tasks/src/detect/` — MediaPipe calculator 등가 순수 Rust:
  `anchors.rs`(SsdAnchors, face 896/palm 2016 프리셋), `decode.rs`(TensorsToDetections,
  reverse_output_order 고정), `nms.rs`(**가중** NMS — 후보 점수가중 평균, 일반 NMS로
  바꾸면 박스가 떤다), `letterbox.rs`(keep_aspect 패딩 계산+역투영+`letterbox_u8_rgb`
  픽셀 헬퍼). 박스/점수 텐서는 이름이 아니라 **원소 수로** 식별 (tf2onnx가 이름을 바꿈).
- `ai-tasks/src/pool.rs` — `Pool<T>` 핸들 저장소 (핸들 단조증가·재사용 없음,
  take/put은 wasm async 경계용).
- wasm exports: `load_model_h`/`unload_model_h`/`model_io_h`/`model_stats_h`/
  `infer_frame_h` + CPU 짝(`*_cpu_h`) + `detect_gpu`/`detect_cpu(handle, preset,
  letterboxed_rgb, src_w, src_h)`. 단일 슬롯 exports는 기존 데모 호환으로 유지.
- **리네임 2건 (사용자 지적)**: ① `ai_tasks::Segmenter`→**`GpuSession`**,
  `CpuSegmenter`→**`CpuSession`** (gpu_session.rs/cpu_session.rs) — 디텍터·랜드마크·
  게이즈까지 담는 범용 모델 인스턴스라 세그 이름은 오해였다. ② 실행기 대칭:
  `ai_cpu::CpuModel`→**`ai_cpu::Model`** (exec.rs→model.rs) — `ai_gpu_runtime::Model`
  (model.rs)과 크레이트 경로만 다른 같은 이름·같은 파일명이 되도록. 크레이트가
  네임스페이스다 — 타입 이름에 백엔드 접두사 금지.
- 게이트 2단: 네이티브 `ai-tasks/tests/face_detect.rs`(구조 게이트, 모델 없으면
  스킵) + 브라우저 `web/demo/face-ab.html`(좌표 diff, `AI_ENGINE_RESULT` 규약).
  MediaPipe 비교 상대 자산은 `face_detector.tflite`(Makefile convert-mediapipe가 복사).

**전처리 규약 (파리티의 절반)**: 디텍터 입력은 **[-1,1] + keep_aspect 레터박스
(검정 패딩=-1)** — MediaPipe ImageToTensor(BORDER_ZERO) 등가. 웹은 검정 캔버스에
drawImage, 캔버스 없는 호스트는 `letterbox_u8_rgb`. 검출 좌표는 엔진이 레터박스
역투영까지 해서 원본 프레임 정규화 좌표로 돌려준다. **랜드마크 입력은 [0,1] 회전
ROI 크롭, replicate 경계** — 샘플링은 OpenCV warpPerspective 규약(dst 정수픽셀
코너정합, +0.5 없음; GL 경로와 반픽셀 다르지만 기준인 CPU delegate가 이쪽).

**3·4단계 — FaceTask + 랜드마크 게이트 완료 (2026-08-13 심야)**:
- `detect/roi.rs` — DetectionsToRects(kp 2점 회전) + RectTransformation(1.5
  square_long) + ROI warp 크롭(`crop_u8_rgb`) + LandmarkProjection 역투영.
  전부 절대 px `Roi` 하나로 (MediaPipe의 NormalizedRect 왕복 실수 방지).
- `face.rs` — **FaceTask** (첫 `~Task` 타입): prev ROI 트래킹(검출은 놓쳤을 때만),
  presence sigmoid 게이트(<0.5 → reset), 랜드마크 기반 다음 ROI(kp 33/263 회전),
  process_cpu/process_gpu 드라이버 (수학 공유, 세션은 주입 — 백엔드 선택은 호스트).
  lm 출력 식별: 최대 길이=랜드마크(0~256 px 좌표), 순서상 첫 1원소=presence 로짓
  (실측: 얼굴 +19.7/빈화면 -14.1; Identity_2 [1,1]은 상수 0 — 미사용).
- `filter.rs` — OneEuroFilter+LandmarkSmoother (객체크기 정규화, MediaPipe 구조).
  **파라미터 미확정** — VIDEO 게이트 검증 전까지 smoothing=false 권장.
- wasm: `face_task_new/free/reset`, `face_task_cpu/gpu(task, det, lm, frame_u8,
  w, h, t_ms)` — 픽셀 처리 전부 엔진 안 (JS는 u8 RGB만 넘긴다).
- 게이트: `ai-tasks/tests/face_task.rs`(트래킹 계약: 2프레임째 디텍터 생략,
  presence 미달 시 검출 복귀 — det.stats().frames로 검증) + face-ab.html 랜드마크
  스테이지 — **MediaPipe FaceLandmarker(IMAGE) 대비 478점 max 0.30px/mean 0.13px**
  (CPU/GPU 동일). 디텍터(1.10px)보다 좋은 이유: lm 게이트는 MediaPipe도 같은
  검출→ROI를 거치므로 크롭 보간 차만 남는다.

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

## 6. 오디오 — fastenhancer 개통 완료 (2026-08-14)

**결과**: 16k/48k 양 레이트 개통, **wav2wav ONNX 대비 SNR 126.4dB**(사실상 완전
일치 — vcx-noise ncnn판은 49dB), 48k 실측 **native 1.83ms / wasm 2.79ms per
hop**(예산 10.67ms의 17%/26%). 게이트: `tests/audio.rs`(그래프 오라클 1e-4 +
e2e SNR≥45dB + hop<5ms) + `web/demo/audio-ab.html`(브라우저 SNR·벽시계 —
**GPU 없이(init_engine 없이) 도는지도 검증** = audio 워커 계약).
재현: `make convert-fastenhancer && cargo test -p ai-tasks --test audio`.

**구조 (경로 결정의 근거 포함)**:
- 공개 wav2wav ONNX는 이미 스트리밍 1-hop(wav_in+캐시 5)이고 DFT가 모델 안.
  `tools/prep_fastenhancer.py`가 **DFT 양끝을 수술**해 spec2spec 서브그래프만
  추출(경계: 압축스펙 mul_1 → 마스크 convolution) — **--verify가 numpy로
  전/후처리를 재현해 원본과 1e-9 대조** (이 수식이 Rust 이식의 단일 기준:
  α=0.3 압축·clip 1e-5·복소 마스크 곱·β=10/3 역압축·Nyquist드랍·conj-irfft).
- **`.sw`가 아니라 전용 미니 실행기** (`features/audio/graph.rs+ops.rs` —
  rank≤4 f32, JSON 그래프+가중치 blob 394KB): 서브그래프가 rank-4 attention·
  동적 MatMul·ConvTranspose1d 등 이미지 IR과 이질적이고 텐서가 미소(최대
  48×128)해서. vcx-noise가 ncnn 전용 재작성으로 간 것과 같은 판단 —
  ai-convert 오디오 계열 일반화는 다음 오디오 모델 때.
- `features/audio/stft.rs`(vcx-noise 이식, rustfft) + `enhancer.rs`(스트리밍
  상태+전/후처리). wasm exports `enhancer_new/free/reset/frame_len/process`.
- **오디오는 CPU 고정** (AudioWorklet에 WebGPU 없음 — 워커 토폴로지 규약).

**ORT 추격전 완료 (2026-08-14 밤, 사용자 지시 — wasm 1차 + native 2차)**:
**wasm 2.79 → 0.85ms (ORT wasm 1.18의 1.39배 우위) / native 0.90 → 0.637ms
(ORT MLAS 0.58과 동급, 1.10배)**. SNR 125dB+ 유지. 수단 (전부 wasm 프로파일 근거 — enhancer_profile export 신설):
① conv1d **레지스터 블로킹**: ox 16블록(F32x4×4)을 레지스터에 유지하고
(ic,kk)를 안쪽으로 + **oc 4행 페어링**(x 로드 재사용 4배, acc 16+x4+w4=24
vreg — wasm에서도 스필 없이 이득, 꼬리는 2행 경로) ② MatMul j 16블록 동일
블로킹 ③ Gemm(W^T) 2행×4열 8누산 ④ **Cephes 다항 exp**(fast_exp/exp4 —
MLAS MlasComputeLogistic 방식, sigmoid/tanh/softmax 벡터화 + F32x4::div 신설;
오라클 1e-6 유지) ⑤ ew fast path(같은 shape/스칼라/마지막축 bias flat) +
odometer 인덱싱 ⑥ **Slice/Split 마지막축 memcpy 특례** (GRU 게이트·헤드
슬라이스 전부 이 모양) ⑦ Transpose 마지막축-유지 특례(행 memcpy)
⑧ Reshape/Squeeze/Unsqueeze **zero-copy**(소비 카운트 기반 마지막 소비자
move) ⑨ ConvTranspose 프리팩 gather(w [kk][oc][ic]·x [xi][ic] 재배치로 ic
내적 연속화 — 원소 gather는 캐시미스로 퇴행했었다).
**실패 실험 (재시도 금지)**: 첫 명시 F32x4 시도(kk 바깥, orow를
load-modify-store로 훑는 구조)는 native 퇴행+wasm 무반응 — orow 메모리
트래픽이 지배해서다. **블로킹 없는 벡터화는 이득이 없다**가 교훈 (acc를
레지스터에 유지하는 구조로 바꾸자 2.5배). 잔여 카드: 텐서 플랜(할당 제거 —
잡 op 0.3ms), ConvTranspose 블로킹, Gemm 프리팩.
**함정 기록**: ①ONNX Slice의 ends=INT64_MAX가 wasm(32비트 isize) `as isize`
캐스트에서 −1로 뭉개져 마지막 원소가 잘림 — **네이티브에선 무증상, audio-ab
게이트가 잡음**. slice는 i64로 계산할 것 (ops.rs). ②run_profiled는 실행마다
정렬 순서가 달라진다 — 프로파일 합산은 이름 기준으로.

**데모 (2026-08-14, 사용자 지시로 2회 개편)**: `web/demo/audio-live.html` —
**10초 녹음 → 모델 선택(ai_engine/onnxruntime/원음) → 출력** 워크플로.
같은 녹음을 모델만 바꿔 반복 청취(엔진별 캐시), WAV 3트랙 다운로드.
마이크는 기본 설정 그대로(제약 강제 없음 — 사용자 지시). 라이브 루프백·
동시비교 SNR 화면은 이 개편으로 대체됨 (조사 결론은 아래 항목에 보존).
실사용: `node tools/run_web.mjs demo/audio-live.html --headed` (--camera는
가짜 장치용이라 실마이크 시엔 제외). 스모크: ?smoke=1 — 녹음 2초 단축,
양 엔진 처리+재생 자동 (rec 2s: ai_engine 0.27s / ORT 0.35s 처리).

**"ORT가 약간 좋게 들림" 조사 (2026-08-14 밤, 사용자 청감)**: 라이브 데모에
**동시 비교 모드**(같은 입력 두 엔진 동시 처리 + 프레임번호 큐 정렬 + 실시간
SNR)와 **3트랙 A/B 녹음**(원음/우리/ORT WAV 저장) 추가. 조사 결론:
- **이 모델은 GRU 재귀 발산 시스템** — 미세 수치 차이가 시간에 따라 증폭돼
  "정상 구현끼리도" 궤적이 갈라진다. 실측: 무음+비프 8초에서 numpy 청사진
  (f64+ORT 서브그래프) vs full ONNX(ORT native) = **84dB**, 우리 Rust도
  **84.3dB(청사진과 동급 = 구현 정합)**, wasm쌍(우리 vs ORT wasm)은 ~34dB.
  랜덤 노이즈 3hop 검증(1e-9)이나 실음성 픽스처(126dB)에선 안 보이던 특성 —
  **긴 무음 구간이 발산을 키운다**.
- fast_exp/exp4는 원인 아님 (정확 exp 실험으로 반증 — 무변화).
- 절대각 정합은 게이트 불가 → 라이브 diff 판정은 경로 정상성(>10dB — 정렬
  깨짐/신호 불일치 감지)만. 실음질 판단은 **A/B 녹음**으로 (같은 발화 3트랙).
- 함정: diff 측정은 ①무음 프레임 제외(−60dBFS 게이트 — 아니면 바닥 잔차
  비율만 잰다) ②ORT busy 스킵 금지(입력 큐 — 프레임 유실 = 캐시 스트림
  영구 어긋남) ③전환 시 양쪽 동시 백지 리셋.

**잔여**: ①v-ai audio 워커 실연결(AudioWorklet 배선 — P4와 한 묶음)
②16k 브라우저 게이트(네이티브 오라클은 통과, 픽스처는 48k만) ③텐서 플랜
최적화 카드 ④ai-ffi 오디오 표면(진짜 vcx-noise 교체 — 모바일 실연결 때).

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
