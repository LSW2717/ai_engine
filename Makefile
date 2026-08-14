# ai_engine 태스크 러너 — 절대경로 금지, 모든 경로는 저장소 루트 기준
.PHONY: build-native test bench build-wasm build-wasm-dev serve-web web clean setup-wasm ffi-header build-android build-ios convert-rvm-web convert-r11-web convert-fastenhancer

build-native:
	cargo build --release -p ai-gpu

# ai-ffi C 헤더 (모바일 브리지 소비) — cargo install cbindgen 필요
ffi-header:
	cbindgen --config crates/ai-ffi/cbindgen.toml --crate ai-ffi \
	  --output crates/ai-ffi/include/ai_engine_ffi.h crates/ai-ffi
	@echo "→ crates/ai-ffi/include/ai_engine_ffi.h"

# ── 모바일 아티팩트 (vcxrust_ai README 레시피의 ai-ffi판) ──
# Android .so 3종 (arm64-v8a/armeabi-v7a/x86_64) — cargo install cargo-ndk +
# ANDROID_NDK_HOME 필요. 출력: target/jniLibs/<abi>/libai_ffi.so
# (앱 배치 시 -o ./android/app/src/main/jniLibs 로 교체)
build-android:
	cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 \
	  -o target/jniLibs build --release -p ai-ffi

# iOS xcframework: 디바이스(arm64) + 시뮬레이터 유니버설(arm64+x86_64, lipo)
# → target/xcframework/AiEngineFFI.xcframework (헤더 = cbindgen 생성물 동봉)
build-ios: ffi-header
	cargo build --release --target aarch64-apple-ios -p ai-ffi
	cargo build --release --target aarch64-apple-ios-sim -p ai-ffi
	cargo build --release --target x86_64-apple-ios -p ai-ffi
	mkdir -p target/ios-sim-universal target/xcframework
	lipo -create -output target/ios-sim-universal/libai_ffi.a \
	  target/aarch64-apple-ios-sim/release/libai_ffi.a \
	  target/x86_64-apple-ios/release/libai_ffi.a
	lipo -info target/ios-sim-universal/libai_ffi.a
	rm -rf target/xcframework/AiEngineFFI.xcframework
	xcodebuild -create-xcframework \
	  -library target/aarch64-apple-ios/release/libai_ffi.a \
	  -headers crates/ai-ffi/include \
	  -library target/ios-sim-universal/libai_ffi.a \
	  -headers crates/ai-ffi/include \
	  -output target/xcframework/AiEngineFFI.xcframework

test:
	cargo test --workspace

bench:
	cargo run --release -p ai-bench

# +simd128: ai-cpu 폴백 커널의 SIMD128 경로 (없으면 스칼라로 조용히 떨어진다).
# 모든 모던 브라우저가 지원 (2021+).
# pkg-relaxed: +relaxed-simd 추가 빌드 — fma가 1명령(FMLA)이 되어 CPU 경로가
# ~25% 빨라진다. Chrome 114+/Firefox 128+ 지원, Safari 미지원 → 로더가
# relaxed를 먼저 시도하고 CompileError면 pkg로 폴백한다 (cpu-ab.js 참조).
build-wasm:
	RUSTFLAGS="-C target-feature=+simd128" \
	wasm-pack build crates/ai-wasm --target web --release --out-dir ../../web/pkg --out-name ai_engine
	RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" \
	wasm-pack build crates/ai-wasm --target web --release --out-dir ../../web/pkg-relaxed --out-name ai_engine

build-wasm-dev:
	RUSTFLAGS="-C target-feature=+simd128" \
	wasm-pack build crates/ai-wasm --target web --dev --out-dir ../../web/pkg --out-name ai_engine

# 기본은 VS Code Live Server 사용 (web/index.html 우클릭 → Open with Live Server).
# 아래는 폴백.
serve-web:
	python3 -m http.server 8080 -d web

web: build-wasm serve-web

convert-rvm-web:
	cargo run --release -p ai-convert -- models/rvm_fp32.onnx -o web/models/rvm_256x144.sw \
	  --size 256x144 --set-input downsample_ratio=1.0 \
	  --state r1i=r1o --state r2i=r2o --state r3i=r3o --state r4i=r4o --name rvm --fp16

# R11(CPU 폴백) — .sw 변환 + cpu-ab.html 비교 상대(tflite f16 모델·onnx) 복사.
# tflite 자산은 v-ai 리포에서 온다 (경로가 다르면 V_AI=... 로 넘길 것).
V_AI ?= ../../vcxreact/packages/v-ai
convert-r11-web:
	cargo run --release -p ai-convert -- models/segm_mnv4s050_s2_160x288_nhwc.onnx \
	  -o web/models/segm_r11_160x288.sw --name segm-r11
	cp models/segm_mnv4s050_s2_160x288_nhwc.onnx web/models/
	@test -f $(V_AI)/assets/models/segm_mnv4s050_s2_160x288_float16.tflite \
	  && cp $(V_AI)/assets/models/segm_mnv4s050_s2_160x288_float16.tflite web/models/ \
	  && cp $(V_AI)/assets/tflite/tflite-simd.js $(V_AI)/assets/tflite/tflite-simd.wasm web/demo/tflite/ \
	  || echo "경고: $(V_AI) 없음 — tflite 비교 상대는 건너뜀 (ai-cpu/ORT만 동작)"

# MediaPipe 계열 5모델: .task→onnx(tf2onnx)→.sw + 벤치 자산 복사 (NEXT.md §3.5)
convert-mediapipe:
	/usr/bin/python3 tools/prep_mediapipe.py \
	  $(V_AI)/assets/models/face_landmarker.task $(V_AI)/assets/models/hand_landmarker.task \
	  -o models/mediapipe
	cp $(V_AI)/assets/models/mobileone_s0_gaze.onnx models/ 2>/dev/null || true
	cargo run --release -q -p ai-convert -- models/mobileone_s0_gaze.onnx -o models/gaze.sw --name gaze
	cargo run --release -q -p ai-convert -- models/mediapipe/face/face_detector.onnx -o models/mediapipe/face/face_detector.sw --name face-det
	cargo run --release -q -p ai-convert -- models/mediapipe/face/face_landmarks_detector.onnx -o models/mediapipe/face/face_landmarks.sw --name face-lm
	cargo run --release -q -p ai-convert -- models/mediapipe/face/face_blendshapes.onnx -o models/mediapipe/face/face_blendshapes.sw --name face-bs
	cargo run --release -q -p ai-convert -- models/mediapipe/hand/hand_detector.onnx -o models/mediapipe/hand/hand_detector.sw --name hand-det
	cargo run --release -q -p ai-convert -- models/mediapipe/hand/hand_landmarks_detector.onnx -o models/mediapipe/hand/hand_landmarks.sw --name hand-lm
	mkdir -p web/models/mediapipe
	cp models/gaze.sw models/mobileone_s0_gaze.onnx \
	  models/mediapipe/face/face_detector.sw models/mediapipe/face/face_detector.onnx \
	  models/mediapipe/face/face_landmarks.sw models/mediapipe/face/face_landmarks_detector.onnx \
	  models/mediapipe/face/face_blendshapes.sw models/mediapipe/face/face_blendshapes.onnx \
	  models/mediapipe/hand/hand_detector.sw models/mediapipe/hand/hand_detector.onnx \
	  models/mediapipe/hand/hand_landmarks.sw models/mediapipe/hand/hand_landmarks_detector.onnx \
	  web/models/mediapipe/
	cp $(V_AI)/assets/models/face_landmarker.task $(V_AI)/assets/models/hand_landmarker.task web/models/mediapipe/
	cp tests/data/frame_256x144.rgb web/models/mediapipe/
	# face-ab 좌표 diff 게이트의 MediaPipe 비교 상대 (같은 가중치의 원본 tflite)
	cp models/mediapipe/face/face_detector.tflite web/models/mediapipe/
	# hand-ab 게이트의 손 테스트 프레임 (MediaPipe 공식 샘플 — 얼굴 프레임엔 손이 없다)
	test -f web/models/mediapipe/hands.jpg || \
	  curl -sL -o web/models/mediapipe/hands.jpg \
	  "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"

setup-wasm:
	rustup target add wasm32-unknown-unknown
	@command -v wasm-pack >/dev/null || cargo install wasm-pack --locked

clean:
	cargo clean
	rm -rf web/pkg

# vb-diff 픽셀 diff 게이트 비교 상대 — v-ai 비디오 워커 원본 사본 (바이트 동일,
# gitignore). vai-stack.js가 blob import로 스테이지를 꺼내 스탠드얼론 구동한다.
vai-gate-assets:
	mkdir -p web/demo/vendor
	cp $(V_AI)/src/virtual-background/video-worker-webgl2.js \
	   $(V_AI)/src/virtual-background/background-fit.js \
	   $(V_AI)/src/virtual-background/webgl2-engine-span.js web/demo/vendor/

# fastenhancer (오디오, 로드맵 #6): 공개 wav2wav ONNX → spec2spec 수술+검증+
# 미니 실행기 포맷 + 브라우저 게이트 자산. 픽스처는 vcxrust_ai vcx-noise에서.
VCX_RUST ?= ../vcxrust_ai
convert-fastenhancer:
	/usr/bin/python3 tools/prep_fastenhancer.py --src $(V_AI)/assets/models/fastenhancer_b_48k.onnx \
	  -o models/fastenhancer/fe48_spec2spec.onnx --verify --export models/fastenhancer/fe48
	/usr/bin/python3 tools/prep_fastenhancer.py --src $(V_AI)/assets/models/fastenhancer_b_16k.onnx \
	  -o models/fastenhancer/fe16_spec2spec.onnx --verify --export models/fastenhancer/fe16
	cp $(VCX_RUST)/crates/vcx-noise/fast-enhancer/tests/fixtures/in_48k.f32 \
	  $(VCX_RUST)/crates/vcx-noise/fast-enhancer/tests/fixtures/ref_48k_wav2wav.f32 models/fastenhancer/
	mkdir -p web/models/fastenhancer/fe48 web/models/fastenhancer/fe16
	cp models/fastenhancer/fe48/graph.json models/fastenhancer/fe48/weights.bin web/models/fastenhancer/fe48/
	cp models/fastenhancer/fe16/graph.json models/fastenhancer/fe16/weights.bin web/models/fastenhancer/fe16/
	cp models/fastenhancer/in_48k.f32 models/fastenhancer/ref_48k_wav2wav.f32 web/models/fastenhancer/


# studio 데모 실제 제품 에셋 (배경 이미지 + 3D GLB) — v-ai/v-room에서 복사 (gitignore)
V_ROOM ?= ../../vcxreact/packages/v-room
studio-assets:
	mkdir -p web/demo/assets/bg web/demo/assets/glb
	for i in 1 2 3 4 5 6; do cp $(V_ROOM)/assets/images/room/filters/study/$$i.jpg web/demo/assets/bg/ 2>/dev/null || true; done
	cp $(V_AI)/assets/models/*.glb web/demo/assets/glb/
