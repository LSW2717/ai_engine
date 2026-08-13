# ai_engine 태스크 러너 — 절대경로 금지, 모든 경로는 저장소 루트 기준
.PHONY: build-native test bench build-wasm build-wasm-dev serve-web web clean setup-wasm ffi-header convert-rvm-web convert-r11-web

build-native:
	cargo build --release -p ai-gpu

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
	cargo run --release -q -p ai-convert -- models/mediapipe/hand/hand_detector.onnx -o models/mediapipe/hand/hand_detector.sw --name hand-det
	cargo run --release -q -p ai-convert -- models/mediapipe/hand/hand_landmarks_detector.onnx -o models/mediapipe/hand/hand_landmarks.sw --name hand-lm
	mkdir -p web/models/mediapipe
	cp models/gaze.sw models/mobileone_s0_gaze.onnx \
	  models/mediapipe/face/face_detector.sw models/mediapipe/face/face_detector.onnx \
	  models/mediapipe/face/face_landmarks.sw models/mediapipe/face/face_landmarks_detector.onnx \
	  models/mediapipe/hand/hand_detector.sw models/mediapipe/hand/hand_detector.onnx \
	  models/mediapipe/hand/hand_landmarks.sw models/mediapipe/hand/hand_landmarks_detector.onnx \
	  web/models/mediapipe/
	cp $(V_AI)/assets/models/face_landmarker.task $(V_AI)/assets/models/hand_landmarker.task web/models/mediapipe/
	cp tests/data/frame_256x144.rgb web/models/mediapipe/
	# face-ab 좌표 diff 게이트의 MediaPipe 비교 상대 (같은 가중치의 원본 tflite)
	cp models/mediapipe/face/face_detector.tflite web/models/mediapipe/

setup-wasm:
	rustup target add wasm32-unknown-unknown
	@command -v wasm-pack >/dev/null || cargo install wasm-pack --locked

clean:
	cargo clean
	rm -rf web/pkg

# studio 데모 실제 제품 에셋 (배경 이미지 + 3D GLB) — v-ai/v-room에서 복사 (gitignore)
V_ROOM ?= ../../vcxreact/packages/v-room
studio-assets:
	mkdir -p web/demo/assets/bg web/demo/assets/glb
	for i in 1 2 3 4 5 6; do cp $(V_ROOM)/assets/images/room/filters/study/$$i.jpg web/demo/assets/bg/ 2>/dev/null || true; done
	cp $(V_AI)/assets/models/*.glb web/demo/assets/glb/
