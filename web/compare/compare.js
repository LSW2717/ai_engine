// RVM 삼자 비교 — ORT-web WebGPU(fp32/fp16) vs webgl2-engine-span vs ai_engine.
// 모델 추론만 측정 (전처리/합성 없음). 각 엔진의 실사용 API 그대로 사용.

import * as ort from './ort/ort.all.bundle.min.mjs';
import { Webgl2Engine } from './webgl2-engine-span.js';
import initWasm, { init_engine, load_model, model_bench } from '../pkg/ai_engine.js';

const H = 144;
const W = 256;
const WARMUP = 5;
const FRAMES = 30;
const STATE_SHAPES = {
  r1: [1, 16, 72, 128],
  r2: [1, 20, 36, 64],
  r3: [1, 40, 18, 32],
  r4: [1, 64, 9, 16],
};

const rows = document.getElementById('rows');
const status = document.getElementById('status');
const say = (t) => (status.textContent = t);
const addRow = (name, ms, load, note = '') => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${name}</td><td class="num">${
    typeof ms === 'number' ? ms.toFixed(2) : ms
  }</td><td class="num">${typeof load === 'number' ? load.toFixed(0) : '-'}</td><td>${note}</td>`;
  rows.appendChild(tr);
  if (typeof ms === 'number') {
    console.log(`AI_ENGINE_RESULT: compare ${name} ${ms.toFixed(3)}ms load ${load.toFixed(0)}ms`);
  } else {
    console.log(`AI_ENGINE_RESULT: compare ${name} ERROR ${note}`);
  }
};

function seededInput() {
  let s = 12345 >>> 0;
  const n = 3 * H * W;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    a[i] = (s >>> 8) / 16777216; // [0,1)
  }
  return a;
}

// ---- A) onnxruntime-web WebGPU ----
function toF16(f32) {
  // Chrome 135+: Float16Array 내장 (round-to-nearest)
  const h = new Float16Array(f32.length);
  h.set(f32);
  return new Uint16Array(h.buffer);
}

async function benchOrt(modelUrl, tag, fp16) {
  // wasmPaths 기본값 = 번들 모듈 옆 (./ort/로 주면 번들 기준으로 이중 해석됨)
  ort.env.wasm.numThreads = 1; // SAB/COEP 불필요
  const t0 = performance.now();
  const sess = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
  });
  const load = performance.now() - t0;

  // 입력별 dtype을 세션 메타데이터에서 (fp16 export는 혼합 타입)
  const meta = {};
  try {
    for (const m of sess.inputMetadata ?? []) meta[m.name] = m.type;
  } catch (_) { /* 구버전 폴백 */ }
  const mk = (name, f32, dims) => {
    const t = meta[name] ?? (fp16 ? 'float16' : 'float32');
    return t === 'float16'
      ? new ort.Tensor('float16', toF16(f32), dims)
      : new ort.Tensor('float32', f32, dims);
  };
  const feeds = {
    src: mk('src', seededInput(), [1, 3, H, W]),
    downsample_ratio: mk('downsample_ratio', new Float32Array([1]), [1]),
  };
  for (const [k, shape] of Object.entries(STATE_SHAPES)) {
    feeds[k + 'i'] = mk(k + 'i', new Float32Array(shape.reduce((a, b) => a * b)), shape);
  }
  const step = async () => {
    const out = await sess.run(feeds);
    for (const k of Object.keys(STATE_SHAPES)) feeds[k + 'i'] = out[k + 'o'];
  };
  for (let i = 0; i < WARMUP; i++) await step();
  const t1 = performance.now();
  for (let i = 0; i < FRAMES; i++) await step();
  const ms = (performance.now() - t1) / FRAMES;
  await sess.release();
  return { ms, load, tag };
}

// ---- B) webgl2-engine-span ----
async function benchWebgl2() {
  const canvas = document.createElement('canvas');
  canvas.width = 8;
  canvas.height = 8;
  const gl = canvas.getContext('webgl2', {
    antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error('WebGL2 컨텍스트 생성 실패');
  const engine = new Webgl2Engine(gl, {});
  const t0 = performance.now();
  await engine.load('./assets/rvm_144x256.plan.json', './assets/rvm_144x256.weights.bin');

  engine.upload('input_1', seededInput());
  // 순환쌍 수집 + 0 초기화 (video-worker-webgl2.js 레시피 그대로)
  const outNames = new Set(engine.plan.outputs.map((o) => o.name));
  const recurrent = [];
  for (const inp of engine.plan.inputs) {
    if (inp.name === 'input_1') continue;
    const outName = outNames.has(inp.name + 'o')
      ? inp.name + 'o'
      : outNames.has(inp.name + '_out')
        ? inp.name + '_out'
        : null;
    if (outName) {
      const [, c, h, w] = inp.shape;
      engine.upload(inp.name, new Float32Array(c * h * w));
      recurrent.push([inp.name, outName]);
    }
  }
  const step = () => {
    engine.run();
    for (const [a, b] of recurrent) {
      const ta = engine.tex.get(a);
      const tb = engine.tex.get(b);
      engine.tex.set(a, tb);
      engine.tex.set(b, ta);
    }
  };
  // 워밍업(지연 셰이더 컴파일 포함)
  for (let i = 0; i < WARMUP; i++) step();
  gl.finish();
  const load = performance.now() - t0;
  const t1 = performance.now();
  for (let i = 0; i < FRAMES; i++) step();
  gl.finish();
  const ms = (performance.now() - t1) / FRAMES;
  return { ms, load, draws: engine.stats.draws };
}

// ---- B2) TF.js (공식 RVM tfjs export, 공식 데모 사용법 그대로) ----
function toNHWC(nchw) {
  const out = new Float32Array(nchw.length);
  const hw = H * W;
  for (let c = 0; c < 3; c++) {
    for (let p = 0; p < hw; p++) {
      out[p * 3 + c] = nchw[c * hw + p];
    }
  }
  return out;
}

async function benchTfjs(backend) {
  const tf = window.tf;
  if (!tf) throw new Error('tf 전역 없음');
  await tf.setBackend(backend);
  await tf.ready();
  const t0 = performance.now();
  const model = await tf.loadGraphModel('../models/rvm_tfjs/model.json');
  const src = tf.tensor(toNHWC(seededInput()), [1, H, W, 3]);
  const downsample_ratio = tf.tensor(1.0);
  let states = [tf.tensor(0), tf.tensor(0), tf.tensor(0), tf.tensor(0)];
  let lastPha = null;

  const step = async () => {
    const [fgr, pha, r1o, r2o, r3o, r4o] = await model.executeAsync(
      { src, r1i: states[0], r2i: states[1], r3i: states[2], r4i: states[3], downsample_ratio },
      ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']
    );
    states.forEach((s) => s.dispose());
    states = [r1o, r2o, r3o, r4o];
    fgr.dispose();
    if (lastPha) lastPha.dispose();
    lastPha = pha;
  };

  for (let i = 0; i < WARMUP; i++) await step();
  await lastPha.data(); // 워밍업 동기화
  const load = performance.now() - t0;
  const t1 = performance.now();
  for (let i = 0; i < FRAMES; i++) await step();
  await lastPha.data(); // 최종 1회 동기화 (webgl2 방식과 동일)
  const ms = (performance.now() - t1) / FRAMES;
  states.forEach((s) => s.dispose());
  lastPha.dispose();
  src.dispose();
  downsample_ratio.dispose();
  model.dispose();
  return { ms, load };
}

// ---- C) ai_engine (.sw) ----
async function benchOurs() {
  await initWasm();
  await init_engine();
  const t0 = performance.now();
  const resp = await fetch('../models/rvm_256x144.sw');
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const report = await load_model(bytes);
  const load = performance.now() - t0;
  const bench = await model_bench(FRAMES); // 내부: 시드 입력, 워밍업 3, 상태 ping-pong
  return { ms: bench.ms_per_frame, load, report };
}

async function main() {
  if (!navigator.gpu) {
    say('WebGPU 미지원 — ORT/ours 측정 불가');
  }
  // 서멀 편향을 우리 쪽에 불리하게: 우리 엔진을 마지막에
  try {
    say('1/4 ORT-web WebGPU (fp32) 실행 중…');
    const r = await benchOrt('../models/rvm_fp32_ortfix.onnx', 'fp32', false);
    addRow('onnxruntime-web WebGPU (fp32)', r.ms, r.load, 'ceil_mode 패치 필요했음');
  } catch (e) {
    addRow('onnxruntime-web WebGPU (fp32)', 'ERR', '-', String(e).slice(0, 300));
  }
  try {
    say('2/4 ORT-web WebGPU (fp16 export) 실행 중…');
    const r = await benchOrt('../models/rvm_fp16_webgpu.onnx', 'fp16', true);
    addRow('onnxruntime-web WebGPU (fp16 export)', r.ms, r.load, '프로덕션용 export');
  } catch (e) {
    addRow('onnxruntime-web WebGPU (fp16 export)', 'ERR', '-', String(e).slice(0, 300));
  }
  try {
    say('3/6 TF.js WebGL (공식 RVM export) 실행 중…');
    const r = await benchTfjs('webgl');
    addRow('TF.js (webgl, 공식 int8 export)', r.ms, r.load, '공식 데모 방식');
  } catch (e) {
    addRow('TF.js (webgl)', 'ERR', '-', String(e).slice(0, 300));
  }
  try {
    say('4/6 TF.js WebGPU 실행 중…');
    const r = await benchTfjs('webgpu');
    addRow('TF.js (webgpu)', r.ms, r.load, '');
  } catch (e) {
    addRow('TF.js (webgpu)', 'ERR', '-', String(e).slice(0, 300));
  }
  try {
    say('5/6 webgl2-engine-span 실행 중…');
    const r = await benchWebgl2();
    addRow('webgl2-engine-span.js', r.ms, r.load, `${r.draws} draws (로드=지연컴파일 포함)`);
  } catch (e) {
    addRow('webgl2-engine-span.js', 'ERR', '-', String(e).slice(0, 300));
  }
  try {
    say('4/4 ai_engine (.sw) 실행 중…');
    const r = await benchOurs();
    addRow('ai_engine (wgpu, fp32)', r.ms, r.load, `${r.report.ops} ops`);
  } catch (e) {
    addRow('ai_engine (wgpu, fp32)', 'ERR', '-', String(e).slice(0, 300));
  }
  say('완료');
  console.log('AI_ENGINE_RESULT: compare-done');
}

main();
