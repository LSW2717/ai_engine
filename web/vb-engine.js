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
  if ('mirror' in cfg) out.mirror = !!cfg.mirror;
  if ('degree' in cfg) out.degree = cfg.degree === 360 ? 0 : cfg.degree ?? 0;
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

function applyConfig(cfg) {
  const j = translate(cfg);
  try {
    ai.vb_config(JSON.stringify(j));
  } catch (e) {
    console.warn('[vb-engine] config:', e);
    return;
  }
  // 필요한 모델 fire-and-forget (v-ai _loadTFLiteModel 규약)
  if (ai.vb_needs_render()) fetchModel('seg');
  const faceNeeded =
    j.faceItems || j.touchUp?.enabled || j.makeup?.enabled || j.focusDetection?.enabled;
  if (faceNeeded) {
    fetchModel('face_det');
    fetchModel('face_lm');
  }
  if (j.focusDetection?.enabled) {
    fetchModel('gaze');
    fetchModel('gaze_bs');
  }
  if (j.handDetection?.enabled) {
    fetchModel('hand_det');
    fetchModel('hand_lm');
  }
  if (j.faceItems) for (const k of [j.faceItems.hat, j.faceItems.eyewear, j.faceItems.beard]) fetchGlb(k);
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
  if (ai.vb_passthrough()) return { bitmap, passthrough: true };
  let rgb;
  try {
    if (ai.vb_wants_pixels(t)) rgb = extractRgb(bitmap);
  } catch (e) {
    console.warn('[vb-engine] rgb 추출:', e);
  }
  if (ai.vb_needs_render()) {
    if (!segReady && !hasRenderWithoutSeg()) {
      // 모델 로드 중 — 원본 그대로 (검은 화면 방지, v-ai 규약)
      return { bitmap, passthrough: true };
    }
    await ai.vb_frame(bitmap, rgb, t);
    const out = surfaceCanvas.transferToImageBitmap();
    bitmap.close(); // 출력 확보 후에만 (복구 경로 함정)
    return { bitmap: out, passthrough: false };
  }
  // analyzer-only — 태스크만 돌고 원본 그대로
  await ai.vb_analyze(bitmap, rgb, t);
  return { bitmap, passthrough: true };
}

// 아이템만 켜진 경우(비디오 효과 없음)는 seg 없이도 렌더 가능 여부 — 현재
// Director는 렌더 경로에 seg를 요구하지 않는 조합이 없어 false 고정 (아이템도
// 파이프라인 위 오버레이라 seg 필요). 조합이 생기면 여기서 판별.
function hasRenderWithoutSeg() {
  return false;
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
