// MediaPipe 계열 5모델 wasm CPU 벤치 — ai-cpu(.sw) vs ORT wasm 1T (+ MediaPipe e2e 참고선).
// 시드 난수 입력, 모델당 순차 로드 (CPU_MODEL은 단일 슬롯).
// 헤드리스: node tools/run_web.mjs demo/models-bench.html

import * as ort from '../compare/ort/ort.bundle.min.mjs';

const MODELS = [
  { tag: 'gaze_448', sw: 'gaze.sw', onnx: 'mobileone_s0_gaze.onnx' },
  { tag: 'face_det_128', sw: 'face_detector.sw', onnx: 'face_detector.onnx' },
  { tag: 'face_lm_256', sw: 'face_landmarks.sw', onnx: 'face_landmarks_detector.onnx' },
  { tag: 'hand_det_192', sw: 'hand_detector.sw', onnx: 'hand_detector.onnx' },
  { tag: 'hand_lm_224', sw: 'hand_landmarks.sw', onnx: 'hand_landmarks_detector.onnx' },
];
const BASE = '../models/mediapipe/';
const WARMUP = 3;
const FRAMES = 20;

const say = (t) => (document.getElementById('status').textContent = t);
const fmt = (v) => (v > 0 ? v.toFixed(2) + ' ms' : '-');
const row = (name, a, b, g) => {
  const tr = document.createElement('tr');
  const ratio = a > 0 && b > 0 ? (b / a).toFixed(2) + 'x' : '-';
  tr.innerHTML = `<td>${name}</td><td class="num">${fmt(a)}</td>` +
    `<td class="num">${fmt(g)}</td><td class="num">${fmt(b)}</td><td class="num">${ratio}</td>`;
  document.getElementById('rows').appendChild(tr);
};

function seeded(n) {
  let s = 12345 >>> 0;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    a[i] = (s >>> 8) / 16777216;
  }
  return a;
}

const p50 = (v) => [...v].sort((x, y) => x - y)[v.length >> 1];

let aiMod = null;
async function loadAiWasm() {
  try {
    const m = await import('../pkg-relaxed/ai_engine.js');
    await m.default();
    aiMod = m;
  } catch {
    const m = await import('../pkg/ai_engine.js');
    await m.default();
    aiMod = m;
  }
}

async function benchOurs(swUrl) {
  const bytes = new Uint8Array(await (await fetch(BASE + swUrl)).arrayBuffer());
  aiMod.load_model_cpu(bytes);
  const io = aiMod.model_io_cpu();
  const input = seeded(io.h * io.w * io.c);
  const out = io.outputs[0];
  for (let i = 0; i < WARMUP; i++) aiMod.infer_frame_cpu_view(input, out);
  const t = [];
  for (let i = 0; i < FRAMES; i++) {
    const t0 = performance.now();
    aiMod.infer_frame_cpu_view(input, out);
    t.push(performance.now() - t0);
  }
  return p50(t);
}

// WebGPU (GPU 백엔드) — model_bench는 내부에서 시드 입력·워밍업·동기화까지 한다
let gpuReady = false;
async function benchOursGpu(swUrl) {
  if (!gpuReady) {
    if (!navigator.gpu) throw new Error('WebGPU 미지원');
    await aiMod.init_engine();
    gpuReady = true;
  }
  const bytes = new Uint8Array(await (await fetch(BASE + swUrl)).arrayBuffer());
  await aiMod.load_model(bytes);
  const r = await aiMod.model_bench(FRAMES);
  return r.ms_per_frame;
}

async function benchOrt(onnxUrl) {
  ort.env.wasm.numThreads = 1;
  const sess = await ort.InferenceSession.create(BASE + onnxUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const meta = sess.inputMetadata?.[0];
  const dims = (meta?.shape ?? []).map((d) => (typeof d === 'number' && d > 0 ? d : 1));
  const n = dims.reduce((a, b) => a * b, 1);
  const tensor = new ort.Tensor('float32', seeded(n), dims);
  const feeds = { [sess.inputNames[0]]: tensor };
  for (let i = 0; i < WARMUP; i++) await sess.run(feeds);
  const t = [];
  for (let i = 0; i < FRAMES; i++) {
    const t0 = performance.now();
    await sess.run(feeds);
    t.push(performance.now() - t0);
  }
  await sess.release();
  return p50(t);
}

// MediaPipe tasks-vision e2e 참고선 (CDN — 오프라인이면 스킵)
async function benchMediapipe() {
  try {
    const vision = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );
    const fileset = await vision.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    // 실제 사람 프레임 (RVM 게이트 프레임) — 빈 캔버스면 디텍터만 돌고
    // 랜드마크 모델이 아예 안 돈다
    const raw = new Uint8Array(
      await (await fetch(BASE + 'frame_256x144.rgb')).arrayBuffer()
    );
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 144;
    const cx = canvas.getContext('2d');
    const img = cx.createImageData(256, 144);
    for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
      img.data[j] = raw[i];
      img.data[j + 1] = raw[i + 1];
      img.data[j + 2] = raw[i + 2];
      img.data[j + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    for (const [name, cls, task] of [
      ['face_landmarker(e2e)', vision.FaceLandmarker, 'face_landmarker.task'],
      ['hand_landmarker(e2e)', vision.HandLandmarker, 'hand_landmarker.task'],
    ]) {
      try {
        const lm = await cls.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: BASE + task, delegate: 'CPU' },
          runningMode: 'VIDEO',
          ...(name.startsWith('hand') ? { numHands: 1 } : { numFaces: 1 }),
        });
        const t = [];
        for (let i = 0; i < WARMUP + FRAMES; i++) {
          const t0 = performance.now();
          lm.detectForVideo(canvas, performance.now());
          if (i >= WARMUP) t.push(performance.now() - t0);
        }
        lm.close();
        console.log(`AI_ENGINE_RESULT: mbench ${name} ${p50(t).toFixed(2)}ms`);
        row(name, p50(t), -1);
      } catch (e) {
        console.log(`AI_ENGINE_RESULT: mbench ${name} ERR ${String(e).slice(0, 120)}`);
      }
    }
  } catch (e) {
    console.log(`AI_ENGINE_RESULT: mbench mediapipe-skip ${String(e).slice(0, 120)}`);
  }
}

async function main() {
  say('ai_engine wasm 로드 중…');
  await loadAiWasm();
  for (const m of MODELS) {
    say(`${m.tag} 측정 중…`);
    let a = -1;
    let b = -1;
    let g = -1;
    try {
      a = await benchOurs(m.sw);
    } catch (e) {
      console.log(`AI_ENGINE_RESULT: mbench ${m.tag} ours ERR ${String(e).slice(0, 150)}`);
    }
    try {
      g = await benchOursGpu(m.sw);
    } catch (e) {
      console.log(`AI_ENGINE_RESULT: mbench ${m.tag} gpu ERR ${String(e).slice(0, 150)}`);
    }
    try {
      b = await benchOrt(m.onnx);
    } catch (e) {
      console.log(`AI_ENGINE_RESULT: mbench ${m.tag} ort ERR ${String(e).slice(0, 150)}`);
    }
    row(m.tag, a, b, g);
    console.log(
      `AI_ENGINE_RESULT: mbench ${m.tag} ai-cpu ${a.toFixed(2)}ms gpu ${g.toFixed(2)}ms ` +
        `ort ${b.toFixed(2)}ms (${(b / a).toFixed(2)}x)`
    );
  }
  await benchMediapipe();
  say('완료');
  console.log('AI_ENGINE_RESULT: mbench-done');
}

main();
