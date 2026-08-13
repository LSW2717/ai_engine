// R11 세그멘테이션 CPU 폴백 3자 A/B — ai-cpu(wasm SIMD128) vs tflite-simd vs ORT wasm.
//
// 같은 카메라 프레임을 288×160 인터리브 RGB [0,1] 한 벌로 만들어 세 런타임에 그대로
// 먹인다 (셋 다 논리 NHWC라 ab.js와 달리 레이아웃 분기가 없다). 출력은 셋 다
// 픽셀당 [배경, 사람] 2채널 로짓 — v-ai softmax 스테이지와 같은 식으로
// 사람 확률 = sigmoid(사람 - 배경) 을 만들어 그레이스케일로 그린다.
//
// 공정성: COOP/COEP 없는 서버(python -m http.server)에서 열리므로
// crossOriginIsolated=false → 세 런타임 모두 1스레드. 이게 v-ai 배포 기본 조건이다.
// ms는 추론 호출을 감싼 벽시계 — ai-cpu/tflite는 동기 호출이라 순수하고, ORT는
// await 한 번이 끼어 이벤트루프 홉이 섞인다(그 오차까지가 실사용 비용이다).
//
// 모델 주의: ai-cpu(.sw)와 ORT(onnx)는 같은 fp32 가중치 계보라 마스크가 수치급으로
// 일치해야 하고(diff ~1e-5), tflite는 float16 가중치 export라 미세한 차이가 정상이다.

import * as ort from '../compare/ort/ort.bundle.min.mjs';

// ai_engine wasm은 두 빌드가 있다: pkg-relaxed(+relaxed-simd, fma 1명령) / pkg(기본).
// relaxed를 먼저 시도하고 컴파일이 거부되면(Safari) 기본으로 폴백 — 매직바이트
// 감지 대신 실제 컴파일을 신뢰한다.
let aiMod = null; // { load_model_cpu, infer_frame_cpu, model_stats_cpu, model_io_cpu }
let aiBuild = '';

async function loadAiWasm() {
  try {
    const m = await import('../pkg-relaxed/ai_engine.js');
    await m.default();
    aiMod = m;
    aiBuild = 'relaxed-simd';
  } catch (e) {
    console.warn('[cpu-ab] relaxed-simd 빌드 거부 → 기본 simd128로 폴백:', e);
    const m = await import('../pkg/ai_engine.js');
    await m.default();
    aiMod = m;
    aiBuild = 'simd128';
  }
  window.__aiMod = aiMod; // 진단 통로 (tools/profile_web.mjs --ops)
}

const SW_URL = '../models/segm_r11_160x288.sw';
const TFLITE_MODEL_URL = '../models/segm_mnv4s050_s2_160x288_float16.tflite';
const ONNX_URL = '../models/segm_mnv4s050_s2_160x288_nhwc.onnx';
const TFLITE_WASM_URL = new URL('./tflite/tflite-simd.wasm', import.meta.url).href;

const TARGET_FPS = 30;
const FRAME_MS = 1000 / TARGET_FPS;
const DONE_AFTER_TICKS = 8; // 헤드리스(run_web.mjs)용 완료 센티널 시점

const $ = (id) => document.getElementById(id);
const say = (t, err = false) => {
  $('status').textContent = t;
  $('status').classList.toggle('err', err);
};

let stream = null;
let video = null;
let running = false;
let io = null; // model_io_cpu() — {h, w, c, input, outputs}
let px = 0;
let rgb = null; // 288×160 인터리브 RGB [0,1] — 세 런타임 공용
let srcCtx = null;

// 엔진별 상태 — mask는 softmax 결과(픽셀당 사람 확률), times는 벽시계 링버퍼
const engines = {
  ours: { box: 'en-ours', metric: 'm-ours', canvas: 'outOurs', mask: null, times: [], ok: false },
  tfl: { box: 'en-tfl', metric: 'm-tfl', canvas: 'outTfl', mask: null, times: [], ok: false },
  ort: { box: 'en-ort', metric: 'm-ort', canvas: 'outOrt', mask: null, times: [], ok: false },
};
let tflite = null, tflInOff = 0, tflOutOff = 0;
let sess = null, ortInName = '', ortOutName = '';

// ?only=ours|tfl|ort — 한 런타임만 돌린다 (프로파일링용: 남의 샘플이 안 섞이게)
const ONLY = new URLSearchParams(location.search).get('only');

const median = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
const p90 = (a) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.9)] : NaN);
const push = (arr, v) => { arr.push(v); if (arr.length > 60) arr.shift(); };
const onlyKey = { ours: 'ours', tfl: 'tfl', ort: 'ort' }[ONLY] ?? null;
const enabled = (e) =>
  e.ok && $(e.box).checked && (!onlyKey || engines[onlyKey] === e);

// 픽셀당 [배경, 사람] 로짓 → 사람 확률 (v-ai softmax 스테이지와 동일한 식)
function softmax2(logits, mask) {
  for (let i = 0; i < mask.length; i++) {
    mask[i] = 1 / (1 + Math.exp(logits[2 * i] - logits[2 * i + 1]));
  }
}

function drawMask(engine) {
  const c = $(engine.canvas);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(io.w, io.h);
  for (let i = 0; i < px; i++) {
    const v = Math.max(0, Math.min(255, engine.mask[i] * 255)) | 0;
    img.data[4 * i] = v; img.data[4 * i + 1] = v; img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function meanAbsDiff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

// ── 로드 (엔진 하나가 죽어도 나머지는 돈다 — 체크박스만 꺼진다) ──────────

async function loadOurs() {
  await loadAiWasm(); // GPU 초기화(init_engine) 불필요 — CPU 경로는 wasm만 있으면 된다
  const resp = await fetch(SW_URL);
  if (!resp.ok) throw new Error(`${SW_URL} 없음 (${resp.status}) — make convert-r11-web`);
  const report = aiMod.load_model_cpu(new Uint8Array(await resp.arrayBuffer()));
  io = aiMod.model_io_cpu();
  console.log(
    `[ai-cpu-ab] 로드(${aiBuild}): ${report.name} ops ${report.ops} 입력 ${io.w}x${io.h}`
  );
}

async function loadTflite() {
  const mod = await import('./tflite/tflite-simd.js');
  tflite = await mod.default({
    locateFile: (f) => (f === 'tflite-simd.wasm' ? TFLITE_WASM_URL : f),
  });
  const buf = await (await fetch(TFLITE_MODEL_URL)).arrayBuffer();
  tflite.HEAPU8.set(new Uint8Array(buf), tflite._getModelBufferMemoryOffset());
  tflite._loadModel(buf.byteLength);
  tflInOff = tflite._getInputMemoryOffset() / 4;
  tflOutOff = tflite._getOutputMemoryOffset() / 4;
}

async function loadOrt() {
  ort.env.wasm.numThreads = 1; // COI 없어도 1이 명시가 안전
  sess = await ort.InferenceSession.create(ONNX_URL, { executionProviders: ['wasm'] });
  ortInName = sess.inputNames[0];
  ortOutName = sess.outputNames[0];
}

// ── 프레임 루프 ──────────────────────────────────────────────────────────

let fps = 0, fpsT0 = performance.now(), ticks = 0, doneSent = false;

async function frame() {
  if (!running) return;
  const t0 = performance.now();

  srcCtx.drawImage(video, 0, 0, io.w, io.h);
  const data = srcCtx.getImageData(0, 0, io.w, io.h).data;
  for (let p = 0, q = 0; p < data.length; p += 4, q += 3) {
    rgb[q] = data[p] / 255; rgb[q + 1] = data[p + 1] / 255; rgb[q + 2] = data[p + 2] / 255;
  }

  const e = engines;
  if (enabled(e.ours)) {
    const t = performance.now();
    // 뷰 반환(복사 1회) — tflite HEAPF32 직독과 같은 규약. 즉시 소비한다.
    const logits = aiMod.infer_frame_cpu_view(rgb, io.outputs[0]);
    push(e.ours.times, performance.now() - t);
    softmax2(logits, e.ours.mask);
    drawMask(e.ours);
  }
  if (enabled(e.tfl)) {
    const t = performance.now();
    tflite.HEAPF32.set(rgb, tflInOff); // 입력 복사도 tflite 실사용 비용 — 시간에 포함
    tflite._runInference();
    push(e.tfl.times, performance.now() - t);
    softmax2(tflite.HEAPF32.subarray(tflOutOff, tflOutOff + px * 2), e.tfl.mask);
    drawMask(e.tfl);
  }
  if (enabled(e.ort)) {
    const t = performance.now();
    const out = await sess.run({ [ortInName]: new ort.Tensor('float32', rgb, [1, io.h, io.w, 3]) });
    push(e.ort.times, performance.now() - t);
    softmax2(out[ortOutName].data, e.ort.mask);
    drawMask(e.ort);
  }

  fps++;
  const now = performance.now();
  if (now - fpsT0 >= 1000) {
    const f = fps / ((now - fpsT0) / 1000);
    $('m-fps').textContent = `${f.toFixed(1)} fps (루프 전체)`;

    const fmt = (eng) => `p50 ${median(eng.times).toFixed(2)} · p90 ${p90(eng.times).toFixed(2)} ms`;
    let line = `AI_ENGINE_RESULT: cpu-ab fps ${f.toFixed(1)}`;
    if (enabled(e.ours)) {
      const s = aiMod.model_stats_cpu(); // wasm 내부 순추론 (입력 복사·JS 왕복 제외)
      $('m-ours').textContent = `벽시계 ${fmt(e.ours)}\n순추론 p50 ${s.p50_ms.toFixed(2)} · p90 ${s.p90_ms.toFixed(2)} ms`;
      line += ` | ai-cpu ${median(e.ours.times).toFixed(2)}ms (순추론 ${s.p50_ms.toFixed(2)})`;
    }
    if (enabled(e.tfl)) {
      let d = '';
      if (enabled(e.ours)) d = ` · vs ai-cpu ${meanAbsDiff(e.tfl.mask, e.ours.mask).toFixed(4)}`;
      $('m-tfl').textContent = `벽시계 ${fmt(e.tfl)}${d}`;
      line += ` | tflite ${median(e.tfl.times).toFixed(2)}ms${d.replace(' · vs ai-cpu', ' diff')}`;
    }
    if (enabled(e.ort)) {
      let d = '';
      if (enabled(e.ours)) d = ` · vs ai-cpu ${meanAbsDiff(e.ort.mask, e.ours.mask).toFixed(4)}`;
      $('m-ort').textContent = `벽시계 ${fmt(e.ort)}${d}`;
      line += ` | ort ${median(e.ort.times).toFixed(2)}ms${d.replace(' · vs ai-cpu', ' diff')}`;
    }
    console.log(line);

    ticks++;
    if (ticks >= DONE_AFTER_TICKS && !doneSent) {
      doneSent = true;
      console.log('AI_ENGINE_RESULT: cpu-ab-done');
    }
    fps = 0;
    fpsT0 = now;
  }

  const wait = Math.max(0, FRAME_MS - (performance.now() - t0));
  setTimeout(() => requestAnimationFrame(frame), wait);
}

// ── 시작/정지 ────────────────────────────────────────────────────────────

async function start() {
  $('start').disabled = true;
  try {
    say('카메라 권한 요청 중…');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;
    await video.play();

    // 세 런타임 각각 시도 — 실패한 것만 빠지고 페이지는 돈다
    const tries = [
      ['ours', 'ai-cpu', loadOurs],
      ['tfl', 'tflite-simd', loadTflite],
      ['ort', 'ORT wasm', loadOrt],
    ];
    for (const [key, name, load] of tries) {
      say(`${name} 로드 중…`);
      try {
        const t = performance.now();
        await load();
        engines[key].ok = true;
        console.log(`[cpu-ab] ${name} 로드 ${(performance.now() - t).toFixed(0)}ms`);
      } catch (err) {
        console.error(`[cpu-ab] ${name} 로드 실패:`, err);
        $(engines[key].metric).textContent = `로드 실패: ${String(err).slice(0, 120)}`;
        $(engines[key].box).checked = false;
      }
    }
    if (!engines.ours.ok) throw new Error('ai-cpu 로드 실패 — 아래 메트릭 참조');

    px = io.w * io.h;
    rgb = new Float32Array(px * 3);
    for (const k of Object.keys(engines)) engines[k].mask = new Float32Array(px);
    $('src').width = io.w; $('src').height = io.h;
    srcCtx = $('src').getContext('2d', { willReadFrequently: true });
    for (const k of Object.keys(engines)) {
      $(engines[k].canvas).width = io.w;
      $(engines[k].canvas).height = io.h;
    }

    say(`실행 중 — ${io.w}×${io.h}, 1스레드 3런타임 순차 실행 (crossOriginIsolated=${self.crossOriginIsolated})`);
    running = true;
    $('stop').disabled = false;
    requestAnimationFrame(frame);
  } catch (err) {
    say(`시작 실패: ${err}`, true);
    $('start').disabled = false;
    if (stream) { for (const t of stream.getTracks()) t.stop(); stream = null; }
  }
}

function stop() {
  running = false;
  if (stream) { for (const t of stream.getTracks()) t.stop(); stream = null; }
  $('start').disabled = false;
  $('stop').disabled = true;
  say('정지됨');
}

$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);
