# ai_engine 태스크 러너 — 절대경로 금지, 모든 경로는 저장소 루트 기준
.PHONY: build-native test bench build-wasm build-wasm-dev serve-web web clean setup-wasm ffi-header convert-rvm-web

build-native:
	cargo build --release -p ai-gpu

test:
	cargo test --workspace

bench:
	cargo run --release -p ai-bench

# +simd128: ai-cpu 폴백 커널의 SIMD128 경로 (없으면 스칼라로 조용히 떨어진다).
# 모든 모던 브라우저가 지원 (2021+).
build-wasm:
	RUSTFLAGS="-C target-feature=+simd128" \
	wasm-pack build crates/ai-wasm --target web --release --out-dir ../../web/pkg --out-name ai_engine

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

setup-wasm:
	rustup target add wasm32-unknown-unknown
	@command -v wasm-pack >/dev/null || cargo install wasm-pack --locked

clean:
	cargo clean
	rm -rf web/pkg
