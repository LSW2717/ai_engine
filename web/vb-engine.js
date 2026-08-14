// vb-engine.js — **VbEngine 3함수 심** (v-ai pipeline.worker 티어 계약):
//   configCustomVideoStream(Partial<VBOptions>) — 동기, 미준비면 버퍼링
//   destroyCustomVideoStream()                  — 웜 리셋 (모델·세션 유지)
//   processWorkerFrame(ImageBitmap, timeSec) → {bitmap, passthrough}
//
// 계약의 함정(원본 대조 확정):
//   - passthrough면 **입력 비트맵 그대로**(제로카피, close 금지)
//   - 처리 프레임은 출력 비트맵을 **확보한 뒤에만** 입력을 close
//     (먼저 닫고 throw하면 워커 복구 경로가 detach 비트맵을 전송해 프레임 유실)
//   - 모델 로드는 config에서 fire-and-forget — 로드 중엔 passthrough
//
// 번역(VBOptions → 우리 EffectsPatch): blur/brightness/grayscale ÷100,
// degree 360→0, background base64 → RGBA 디코드 → vb_bg_image + "image".
// faceEffects.{hat,eyewear,beard} → faceItems + GLB fetch. 우리 확장 키
// (focusDetection/handDetection/framing/touchUp/studioLight)는 그대로 통과.
// 엔진 쪽 절반은 crates/ai-wasm/src/vb.rs (Director 접착).

let ai = null;
let ready = false;
let initPromise = null;
let surfaceCanvas = null;
let pendingConfigs = [];
const fetched = new Set(); // 모델/GLB 중복 fetch 방지
let segReady = false;

const url = (p) => new URL(p, import.meta.url).href;
const MODEL = {
  seg: url('models/rvm_256x144.sw'),
  // B티어 폴백 — R11 CPU 세그 (지연 로드: 강등 확정 때만)
  seg_cpu: url('models/segm_r11_160x288.sw'),
  face_det: url('models/mediapipe/face_detector.sw'),
  face_lm: url('models/mediapipe/face_landmarks.sw'),
  gaze: url('models/mediapipe/gaze.sw'),
  gaze_bs: url('models/mediapipe/face_blendshapes.sw'),
  hand_det: url('models/mediapipe/hand_detector.sw'),
  hand_lm: url('models/mediapipe/hand_landmarks.sw'),
};
const GLB_BASE = url('assets/glb/');

// 메이크업 프리셋 룩 — 엔진은 명시 색상만 받는다. v-ai 룩 테이블은 v-ai 몫이라
// 여기선 기본 룩 하나로 매핑 (실연결 때 v-ai MAKEUP_LOOKS를 주입하는 자리).
const DEFAULT_LOOK = {
  lip: { color: '#d98f95', alpha: 0.45 },
  blush: { color: '#edaab2', alpha: 0.18, size: 0.23 },
  shadow: { color: '#b98d84', alpha: 0.16 },
};

// ── mirror/degree 호스트 전처리 (webgl2 _prepareSourceElement 파리티) ──
// 엔진 계약: mirror/degree 프레임 변환은 **호스트 몫** (엔진은 이미지 배경 좌표
// 보정에만 쓴다 — any_active 에도 안 들어간다). 추론 전 적용 = 좌표계가 화면
// 좌표계가 되는 계약(face-effects lm)도 이걸로 지켜진다.
let cfgMirror = false;
let cfgDegree = 0;
let txCanvas = null;

function hostTransformActive() {
  return cfgMirror || ((cfgDegree % 360) + 360) % 360 !== 0;
}

/** 원본 비트맵 → mirror/degree 적용본 (같은 크기 캔버스 — mirror 먼저, rotate 다음) */
async function hostTransform(bitmap) {
  const w = bitmap.width;
  const h = bitmap.height;
  if (!txCanvas || txCanvas.width !== w || txCanvas.height !== h) {
    txCanvas = new OffscreenCanvas(w, h);
  }
  const cx = txCanvas.getContext('2d');
  cx.save();
  cx.setTransform(1, 0, 0, 1, 0, 0);
  cx.clearRect(0, 0, w, h);
  cx.translate(w / 2, h / 2);
  if (cfgMirror) cx.scale(-1, 1);
  const d = ((cfgDegree % 360) + 360) % 360;
  if (d !== 0) cx.rotate((d * Math.PI) / 180);
  cx.drawImage(bitmap, -w / 2, -h / 2, w, h);
  cx.restore();
  return createImageBitmap(txCanvas);
}

async function ensureInit() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      let m;
      try {
        m = await import(url('pkg-relaxed/ai_engine.js'));
        await m.default();
      } catch {
        m = await import(url('pkg/ai_engine.js'));
        await m.default();
      }
      ai = m;
      await ai.init_engine();
      surfaceCanvas = new OffscreenCanvas(2, 2);
      ai.vb_attach(surfaceCanvas);
      ready = true;
      for (const c of pendingConfigs.splice(0)) applyConfig(c);
    } catch (e) {
      console.warn('[vb-engine] init 실패:', e);
      initPromise = null; // 다음 config에서 재시도
      throw e;
    }
  })();
  return initPromise;
}

function fetchModel(kind) {
  if (fetched.has(kind)) return;
  fetched.add(kind);
  fetch(MODEL[kind])
    .then(async (r) => {
      if (!r.ok) throw new Error(`${MODEL[kind]}: ${r.status}`);
      ai.vb_model(kind, new Uint8Array(await r.arrayBuffer()));
      if (kind === 'seg') segReady = true;
    })
    .catch((e) => console.warn('[vb-engine] 모델', kind, e));
}

function fetchGlb(kind) {
  if (kind === 'none' || fetched.has('glb:' + kind)) return;
  fetched.add('glb:' + kind);
  fetch(GLB_BASE + kind + '.glb')
    .then(async (r) => {
      if (!r.ok) throw new Error(`${kind}.glb: ${r.status}`);
      ai.vb_glb(kind, new Uint8Array(await r.arrayBuffer()));
    })
    .catch((e) => console.warn('[vb-engine] GLB', kind, e));
}

/** base64 배경 → RGBA 업로드 (비동기 — 완료 전 프레임은 이전 배경) */
async function uploadBase64Bg(b64) {
  const blob = await (await fetch('data:image/*;base64,' + b64)).blob();
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const cx = c.getContext('2d');
  cx.drawImage(bmp, 0, 0);
  const d = cx.getImageData(0, 0, c.width, c.height);
  ai.vb_bg_image(new Uint8Array(d.data.buffer), c.width, c.height);
  bmp.close();
}

/** VBOptions(Partial) → 우리 단일 JSON — 도착한 키만 번역 (머지 규약 보존) */
function translate(cfg) {
  const out = {};
  if ('blur' in cfg) out.blur = (cfg.blur ?? 0) / 100;
  if ('brightness' in cfg) out.brightness = (cfg.brightness ?? 100) / 100;
  if ('grayscale' in cfg) out.grayscale = (cfg.grayscale ?? 0) / 100;
  if ('mirror' in cfg) {
    out.mirror = !!cfg.mirror;
    cfgMirror = !!cfg.mirror; // 호스트 전처리용
  }
  if ('degree' in cfg) {
    out.degree = cfg.degree === 360 ? 0 : cfg.degree ?? 0;
    cfgDegree = out.degree; // 호스트 전처리용
  }
  if ('background' in cfg) {
    const bg = cfg.background;
    if (bg == null || bg === '') out.background = null;
    else if (bg.startsWith('#')) out.background = bg;
    else {
      out.background = 'image';
      void uploadBase64Bg(bg).catch((e) => console.warn('[vb-engine] bg:', e));
    }
  }
  for (const k of ['studioLight', 'framing', 'touchUp', 'focusDetection', 'handDetection']) {
    if (k in cfg) out[k] = cfg[k];
  }
  if ('makeup' in cfg) {
    // v-ai 평면형 {enabled, look, intensity} → 우리 명시 색상형
    const mk = cfg.makeup;
    out.makeup = mk?.enabled
      ? { enabled: true, intensity: mk.intensity ?? 1.0, ...DEFAULT_LOOK }
      : null;
  }
  if ('faceEffects' in cfg) {
    const fe = cfg.faceEffects ?? {};
    if (fe.makeup !== undefined) {
      out.makeup = fe.makeup?.enabled
        ? { enabled: true, intensity: fe.makeup.intensity ?? 1.0, ...DEFAULT_LOOK }
        : null;
    }
    const items = {
      enabled: !!fe.enabled,
      hat: fe.hat ?? 'none',
      eyewear: fe.eyewear ?? 'none',
      beard: fe.beard ?? 'none',
    };
    out.faceItems =
      items.enabled && (items.hat !== 'none' || items.eyewear !== 'none' || items.beard !== 'none')
        ? items
        : null;
  }
  return out;
}

// 번역된 우리 키 기준 누적 config — **fetch 판정 전용**. 엔진 판정(vb_needs_render)은
// 프레임 비행 중 캐시(stale)일 수 있어 모델 fetch 결정에 쓰면 로드가 누락된다.
const applied = {};
function noteApplied(j) {
  for (const k of Object.keys(j)) applied[k] = j[k];
}
// Director.needs_render 등가 (any_active + faceItems)
function needsRenderLocal() {
  return (
    applied.background != null ||
    (applied.blur ?? 0) > 0 ||
    Math.abs((applied.brightness ?? 1) - 1) > 1e-3 ||
    (applied.grayscale ?? 0) > 0 ||
    !!applied.studioLight?.enabled ||
    !!applied.framing?.enabled ||
    !!(applied.touchUp?.enabled && (applied.touchUp.strength ?? 0) > 0) ||
    !!applied.makeup?.enabled ||
    !!applied.faceItems
  );
}

function applyConfig(cfg) {
  const j = translate(cfg);
  try {
    ai.vb_config(JSON.stringify(j));
  } catch (e) {
    console.warn('[vb-engine] config:', e);
    return;
  }
  noteApplied(j);
  // 필요한 모델 fire-and-forget (v-ai _loadTFLiteModel 규약)
  if (needsRenderLocal()) fetchModel('seg');
  const faceNeeded =
    applied.faceItems ||
    applied.touchUp?.enabled ||
    applied.makeup?.enabled ||
    applied.focusDetection?.enabled;
  if (faceNeeded) {
    fetchModel('face_det');
    fetchModel('face_lm');
  }
  if (applied.focusDetection?.enabled) {
    fetchModel('gaze');
    fetchModel('gaze_bs');
  }
  if (applied.handDetection?.enabled) {
    fetchModel('hand_det');
    fetchModel('hand_lm');
  }
  if (applied.faceItems) {
    for (const k of [applied.faceItems.hat, applied.faceItems.eyewear, applied.faceItems.beard]) {
      fetchGlb(k);
    }
  }
}

// ── B티어: CPU 추론(R11, ai-cpu) + GPU 합성(vb_frame_mask) — studio.js 검증
// 로직 이식. 강등 판정 = 1s 페이싱 샘플(gpu_sync 배수 → 한 프레임 실비용) 10개
// 창 p90>66ms 2연속 (첫 창 웜업 폐기, 승격 없음 — v-ai 규약).
let cpuSeg = null; // { io, canvas, ctx2d, rgb }
let cpuSegLoading = null;
let demoted = false;
let gpuWin = [];
let winCount = 0;
let badWindows = 0;
let lastGpuSample = 0;

function ensureCpuSeg() {
  if (cpuSeg) return Promise.resolve();
  if (!cpuSegLoading) {
    cpuSegLoading = (async () => {
      const r = await fetch(MODEL.seg_cpu);
      if (!r.ok) throw new Error(`${MODEL.seg_cpu}: ${r.status}`);
      ai.load_model_cpu(new Uint8Array(await r.arrayBuffer()));
      const io = ai.model_io_cpu();
      const canvas = new OffscreenCanvas(io.w, io.h);
      cpuSeg = {
        io,
        canvas,
        ctx2d: canvas.getContext('2d', { willReadFrequently: true }),
        rgb: new Float32Array(io.w * io.h * 3),
      };
      console.info(`[vb-engine] B티어 준비 (R11 ${io.w}x${io.h} CPU)`);
    })().catch((e) => {
      cpuSegLoading = null;
      console.warn('[vb-engine] B티어 로드 실패:', e);
    });
  }
  return cpuSegLoading;
}

/** B티어 한 프레임: R11 CPU 로짓 → 엔진 주입 → GPU 합성 (효과 전부 생존) */
function cpuInferMask(src) {
  const { io, ctx2d, rgb: crgb } = cpuSeg;
  ctx2d.drawImage(src, 0, 0, io.w, io.h);
  const d = ctx2d.getImageData(0, 0, io.w, io.h).data;
  for (let p = 0, q = 0; p < d.length; p += 4, q += 3) {
    crgb[q] = d[p] / 255;
    crgb[q + 1] = d[p + 1] / 255;
    crgb[q + 2] = d[p + 2] / 255;
  }
  return ai.infer_frame_cpu(crgb, io.outputs[0]); // view 아닌 복사 (규약 함정 회피)
}

function recordGpuSample(ms) {
  gpuWin.push(ms);
  if (gpuWin.length < 10) return;
  const s = gpuWin.slice().sort((a, b) => a - b);
  const p90 = s[Math.floor(s.length * 0.9)];
  gpuWin.length = 0;
  winCount++;
  if (winCount <= 1) return; // 웜업 창 폐기
  if (p90 > 66) {
    badWindows++;
    if (badWindows >= 2 && !demoted) {
      demoted = true;
      console.warn(`[vb-engine] B티어 강등 (실비용 창 p90 ${p90.toFixed(1)}ms × ${badWindows})`);
      void ensureCpuSeg();
    }
  } else {
    badWindows = 0;
  }
}

/** u8 RGB 추출 (손/집중도 CNN·광원 프로브 소비 틱에만 — vb_wants_pixels) */
let rgbCanvas = null;
function extractRgb(bitmap) {
  if (!rgbCanvas || rgbCanvas.width !== bitmap.width || rgbCanvas.height !== bitmap.height) {
    rgbCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  }
  const cx = rgbCanvas.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bitmap, 0, 0);
  const d = cx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const rgb = new Uint8Array(bitmap.width * bitmap.height * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
    rgb[j] = d[i];
    rgb[j + 1] = d[i + 1];
    rgb[j + 2] = d[i + 2];
  }
  return rgb;
}

// ───────── VbEngine 3함수 ─────────

export function configCustomVideoStream(config) {
  if (!ready) {
    pendingConfigs.push(config);
    void ensureInit().catch(() => {});
    return;
  }
  applyConfig(config);
}

export function destroyCustomVideoStream() {
  if (ready) {
    try {
      ai.vb_detach();
    } catch (e) {
      console.warn('[vb-engine] detach:', e);
    }
  }
}

export async function processWorkerFrame(bitmap, timeSec) {
  const t = timeSec * 1000;
  if (!ready) {
    void ensureInit().catch(() => {});
    return { bitmap, passthrough: true };
  }
  const tx = hostTransformActive();
  if (ai.vb_passthrough()) {
    if (!tx) return { bitmap, passthrough: true };
    // mirror/degree만 켠 조합 — 프레임 변환은 호스트 몫 (엔진 any_active 미포함).
    // 세그/GPU 없이 2D 캔버스만으로 동작한다.
    const out = await hostTransform(bitmap);
    bitmap.close();
    return { bitmap: out, passthrough: false };
  }
  // 추론 전 변환 적용 = 좌표계가 화면 좌표계가 되는 계약 (webgl2 파리티)
  const src = tx ? await hostTransform(bitmap) : bitmap;
  let rgb;
  try {
    if (ai.vb_wants_pixels(t)) rgb = extractRgb(src);
  } catch (e) {
    console.warn('[vb-engine] rgb 추출:', e);
  }
  try {
    if (ai.vb_needs_render()) {
      const useCpu = demoted && !!cpuSeg; // 강등 확정 + R11 준비된 뒤에만 B
      if (!segReady && !useCpu) {
        // 세그 로드 중 — (변환본) 그대로 (검은 화면 방지, v-ai 규약)
        if (src !== bitmap) bitmap.close();
        return { bitmap: src, passthrough: src === bitmap };
      }
      if (useCpu) {
        // B티어: R11 CPU 추론 → GPU 합성
        const logits = cpuInferMask(src);
        await ai.vb_frame_mask(src, logits, 2, cpuSeg.io.w, cpuSeg.io.h, rgb, t);
      } else if (performance.now() - lastGpuSample > 1000) {
        // A티어 페이싱 샘플 — 큐 배수 후 이 프레임만 완료 대기 (실비용)
        lastGpuSample = performance.now();
        await ai.gpu_sync();
        const t0 = performance.now();
        await ai.vb_frame(src, rgb, t);
        await ai.gpu_sync();
        recordGpuSample(performance.now() - t0);
      } else {
        await ai.vb_frame(src, rgb, t);
      }
      const out = surfaceCanvas.transferToImageBitmap();
      if (src !== bitmap) src.close();
      bitmap.close(); // 출력 확보 후에만 (복구 경로 함정)
      return { bitmap: out, passthrough: false };
    }
    // analyzer-only — 태스크만 돌고 (변환본) 반환
    await ai.vb_analyze(src, rgb, t);
    if (src !== bitmap) {
      bitmap.close();
      return { bitmap: src, passthrough: false };
    }
    return { bitmap, passthrough: true };
  } catch (e) {
    // 원본 bitmap 은 워커 복구 경로가 전송한다 — 변환본만 정리하고 재throw
    if (src !== bitmap) src.close();
    throw e;
  }
}

// ───────── 진단/게이트 보조 ─────────

export function getFocusState() {
  return ready ? ai.vb_focus_state() : null;
}

export function pollGesture() {
  return ready ? ai.vb_poll_gesture() : null;
}

export function isReady() {
  return ready;
}

export function initVbEngine() {
  return ensureInit();
}
