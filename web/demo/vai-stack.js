// vai-stack — v-ai GLSL 후처리 스택 스탠드얼론 하네스 (vb-diff.html의 비교 상대).
//
// vendor/video-worker-webgl2.js(= v-ai 원본의 바이트 동일 사본, make vai-gate-assets)
// 를 소스 변환(blob import)으로 열어 스테이지 팩토리를 꺼내 직접 조립한다.
// 팩토리들은 모듈 전역 상태에 의존하지 않는 top-level 함수라 이 조립이 가능하다
// (정찰 확인). v-ai 셰이더가 바뀌면 사본만 갱신하면 게이트가 자동 추종한다.
//
// 마스크 주입 지점 2개:
//   ch=1: segmentationTexture(RGBA8 알파)에 직접 — softmax/EMA 우회 (공간 스택 게이트)
//   ch=2: buildSoftmaxStage + tflite 스텁 힙(RG f32 로짓) — softmax+EMA 포함 게이트
//
// 주의(정찰 결과): 출력 readPixels는 bottom-up(배경 스테이지가 Y 플립) → 여기서
// 행 반전해 돌려준다. 프레임/마스크는 raw 바이트 업로드(컬러 매니지먼트 변수 제거).

let modP = null;
function loadVaiModule() {
  if (modP) return modP;
  modP = (async () => {
    const SRC = new URL('vendor/video-worker-webgl2.js', location.href).href;
    const BASE = SRC.slice(0, SRC.lastIndexOf('/') + 1);
    let src = await (await fetch(SRC)).text();
    if (src.startsWith('<')) throw new Error('vendor 사본 없음 — make vai-gate-assets 먼저');
    src =
      src
        .replaceAll('import.meta.url', JSON.stringify(SRC))
        .replace(`'./background-fit.js'`, JSON.stringify(BASE + 'background-fit.js'))
        .replace(`'./webgl2-engine-span.js'`, JSON.stringify(BASE + 'webgl2-engine-span.js')) +
      `\nexport { buildJointBilateralFilterStage, buildMaskPostProcessStage, buildSoftmaxStage,
        buildBackgroundBlurStage, buildBackgroundImageStage, buildPassthroughStage,
        createTexture, compileShader, createPiplelineStageProgram };\n`;
    return import(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
  })();
  return modP;
}

// v-ai _computePostProcessingConfig 등가 (blur ∈ [0,1], imageBackground = !!background)
function computePpc(blur, imageBackground) {
  const b = Math.max(0, Math.min(1, blur));
  return {
    sigmaSpace: 2.0 + b * 3.2,
    sigmaColor: 0.1 + b * 0.36,
    coverage: [Math.max(0.01, 0.3 - b * 0.05), Math.min(0.99, 0.7 + b * 0.05)],
    lightWrapping: 0.05 + b * 0.1,
    blurStrength: b,
    maskRefine: imageBackground
      ? { maskBlurPx: 1.2, edgeBlend: 0.4, edgeGamma: 0.98, edgeFeather: 0.58 }
      : { maskBlurPx: 1.1, edgeBlend: 0.36, edgeGamma: 0.98, edgeFeather: 0.54 },
    compositeEdge: imageBackground
      ? { spillSuppression: 0.18, edgeDarkening: 0.24 }
      : { spillSuppression: 0.14, edgeDarkening: 0.2 },
  };
}

export async function createVaiStack(canvas, W, H, segW, segH) {
  const mod = await loadVaiModule();
  canvas.width = W;
  canvas.height = H;
  // 컨텍스트 선점 — buildXXX가 다시 getContext해도 이 속성이 유지된다 (readPixels 안정)
  const gl = canvas.getContext('webgl2', {
    preserveDrawingBuffer: true,
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('WebGL2 미지원');

  // ── 공유 리소스 (buildWebGL2Pipeline 등가) ──
  const vs = mod.compileShader(
    gl,
    gl.VERTEX_SHADER,
    `#version 300 es
in vec2 a_position; in vec2 a_texCoord; out vec2 v_texCoord;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); v_texCoord = a_texCoord; }`
  );
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const uvBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);

  // 프레임 텍스처 — LINEAR/CLAMP (v-ai inputFrameTexture 등가)
  const frameTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, frameTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

  // 마스크(seg) NEAREST · 인물 마스크 NEAREST — v-ai 기본 (파리티 핵심)
  const segTex = mod.createTexture(gl, gl.RGBA8, segW, segH);
  const personMask = mod.createTexture(gl, gl.RGBA8, W, H);

  const segCfg = { inputResolution: '256x144', deferInputResizing: true, model: 'meet' };
  const jbf = mod.buildJointBilateralFilterStage(
    gl, vs, posBuf, uvBuf, segTex, segCfg, personMask, canvas
  );
  const mpp = mod.buildMaskPostProcessStage(gl, vs, posBuf, uvBuf, personMask, canvas);
  const refined = mpp.getOutputTexture();

  let bgStage = null;
  let bgType = null;
  let softmax = null; // ch=2 경로 전용 (지연 생성 — history 상태 소유)
  const fakeHeap = new Float32Array(segW * segH * 2);
  const fakeTflite = { HEAPF32: fakeHeap, _getOutputMemoryOffset: () => 0 };
  const maskBytes = new Uint8Array(segW * segH * 4);

  function makeBgStage(type, bgCanvas) {
    if (bgStage) bgStage.cleanUp();
    if (type === 'blur') {
      bgStage = mod.buildBackgroundBlurStage(gl, vs, posBuf, uvBuf, refined, canvas);
    } else if (type === 'image') {
      bgStage = mod.buildBackgroundImageStage(gl, posBuf, uvBuf, refined, bgCanvas, canvas);
    } else {
      bgStage = mod.buildPassthroughStage(gl, posBuf, uvBuf, refined, canvas);
    }
    bgType = type;
  }

  return {
    // cfg = vb-diff의 EffectsPatch 모양 ({background, blur, brightness, grayscale, studioLight})
    // bgImage = { data: Uint8Array RGBA, w, h } (background==='image'일 때)
    configure(cfg, bgImage) {
      const type = cfg.background ? 'image' : cfg.blur > 0 ? 'blur' : 'passthrough';
      let bgCanvas = null;
      if (type === 'image') {
        if (cfg.background === 'image') {
          bgCanvas = document.createElement('canvas');
          bgCanvas.width = bgImage.w;
          bgCanvas.height = bgImage.h;
          bgCanvas
            .getContext('2d')
            .putImageData(
              new ImageData(new Uint8ClampedArray(bgImage.data), bgImage.w, bgImage.h), 0, 0
            );
        } else {
          // 단색(#hex) — v-ai _createColorImage 등가 (2×2 단색 캔버스 → image 스테이지)
          bgCanvas = document.createElement('canvas');
          bgCanvas.width = 2;
          bgCanvas.height = 2;
          const cx = bgCanvas.getContext('2d');
          cx.fillStyle = cfg.background;
          cx.fillRect(0, 0, 2, 2);
        }
      }
      makeBgStage(type, bgCanvas);
      const ppc = computePpc(cfg.blur, !!cfg.background);
      jbf.updateSigmaSpace(ppc.sigmaSpace);
      jbf.updateSigmaColor(ppc.sigmaColor);
      mpp.updateMaskRefineConfig(ppc.maskRefine);
      bgStage.updateCoverage?.(ppc.coverage);
      bgStage.updateLightWrapping?.(ppc.lightWrapping);
      bgStage.updateBlendMode?.('screen');
      bgStage.updateBlurAmount?.(ppc.blurStrength);
      bgStage.updateEdgeConfig?.(ppc.compositeEdge);
      bgStage.updateOutputAdjustments?.(cfg.brightness, cfg.grayscale);
      bgStage.updateRelight?.(cfg.studioLight || { enabled: false });
    },

    // mirror/degree 배경 보정 — image 스테이지만 (v-ai updateTransform 그대로)
    setTransform(mirror, rotationRad) {
      bgStage.updateTransform?.({ mirror, rotation: rotationRad });
    },

    // 인물 프레이밍 크롭 강제 — image 스테이지(image/단색 배경)만 지원.
    // blur/passthrough의 전체 크롭은 v-ai에선 캔버스 2D transform — 게이트 페이지가
    // 출력에 2D 크롭을 걸어 재현한다 (_compositeFrame 등가).
    setFraming(scale, cx, cy) {
      bgStage.updateFraming?.(scale, cx, cy);
    },

    // 시간 상태 리셋 — softmax history 재생성 (직주입 경로엔 시간 상태 없음)
    reset() {
      if (softmax) {
        softmax.cleanUp();
        softmax = null;
      }
    },

    // frameRgba: W×H×4 u8 (top-down) / mask: ch=1이면 segW×segH f32 알파,
    // ch=2면 segW×segH×2 f32 로짓 [bg, person]. 반환: top-down RGBA.
    frame(frameRgba, mask, ch = 1) {
      // ⚠ 스테이지 생성은 프레임 바인딩 **앞**에 — createTexture가 현재 활성
      // 유닛(TEXTURE0)에 새 텍스처를 바인딩해 프레임 텍스처를 밀어낸다 (실제로 당함)
      if (ch === 2 && !softmax) {
        softmax = mod.buildSoftmaxStage(gl, vs, posBuf, uvBuf, segCfg, fakeTflite, segTex);
      }
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, frameTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, frameRgba);
      if (ch === 2) {
        fakeHeap.set(mask);
        softmax.render(); // 로짓 업로드 + softmax + EMA(0.3/0.03/0.9) → segTex + history
      } else {
        for (let i = 0; i < segW * segH; i++) {
          maskBytes[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, mask[i])) * 255);
        }
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, segTex);
        gl.texSubImage2D(
          gl.TEXTURE_2D, 0, 0, 0, segW, segH, gl.RGBA, gl.UNSIGNED_BYTE, maskBytes
        );
      }
      jbf.render();
      mpp.render();
      bgStage.render();
      gl.activeTexture(gl.TEXTURE0); // blurPass가 TEXTURE2에 남긴다 — 관례 복구
      const raw = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, raw);
      // bottom-up → top-down
      const out = new Uint8Array(W * H * 4);
      const row = W * 4;
      for (let y = 0; y < H; y++) {
        out.set(raw.subarray((H - 1 - y) * row, (H - y) * row), y * row);
      }
      return out;
    },
  };
}
