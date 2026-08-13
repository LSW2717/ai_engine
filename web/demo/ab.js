// 같은 카메라 프레임을 두 엔진에 동일하게 먹여 좌우로 비교한다.
//
// 입력 레이아웃이 서로 다르다는 게 이 파일의 유일한 까다로운 지점:
//   ai_engine  — 논리 NHWC(인터리브 RGB)  … upload_input이 pack_nhwc로 감쌈
//   webgl2     — 논리 NCHW(플래너)        … upload()가 data[ch*W*H + y*W + x]로 읽음
// 둘 다 [0,1] 범위. 같은 픽셀에서 두 배열을 만들어 각자에게 준다.
//
// 두 엔진 모두 알파를 CPU로 꺼내므로(readPixels / map_async) 표시되는 추론 시간에는
// 리드백 대기가 포함된다. 순추론 처리량 비교는 web/compare/ 쪽 벤치를 볼 것.

import initWasm, {
  init_engine, load_model, model_io, infer_frame, attach_canvas, infer_and_present,
  model_bench, bench_current,
} from '../pkg/ai_engine.js';
import { Webgl2Engine } from '../compare/webgl2-engine-span.js';

const TARGET_FPS = 30;
const FRAME_MS = 1000 / TARGET_FPS;
const PLAN = '../compare/assets/rvm_144x256.plan.json';
const WEIGHTS = '../compare/assets/rvm_144x256.weights.bin';

const $ = (id) => document.getElementById(id);
const say = (t, err = false) => {
  $('status').textContent = t;
  $('status').classList.toggle('err', err);
};

const outA = $('outA');
const outB = $('outB');
const ctxA = outA.getContext('2d');
const ctxB = outB.getContext('2d');

let stream = null;
let video = null;
let running = false;
let io = null;
let small = null, smallCtx = null;
let maskA = null, maskACtx = null, maskB = null, maskBCtx = null;
let fgA = null, fgACtx = null, fgB = null, fgBCtx = null;
// 프레임 스냅샷 — 마스크를 계산한 그 순간의 픽셀로 합성해야 한다.
// 합성 때 live video를 다시 그리면 추론 시간만큼 색이 앞서가 마스크가 밀려 보인다.
let snap = null, snapCtx = null;
let inv = null, invCtx = null; // (1-m) 임시 캔버스
let rgbNhwc = null, rgbNchw = null;
let wgl = null, wglRecurrent = [];

const tA = [], tB = [], tBup = [], tBrun = [], tBdown = [];
const median = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
const push = (arr, v) => { arr.push(v); if (arr.length > 60) arr.shift(); };

async function loadOurs() {
  const url = $('model').value;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} 없음 (${resp.status})`);
  await load_model(new Uint8Array(await resp.arrayBuffer()));
  io = model_io();
}

async function loadWebgl2() {
  const canvas = document.createElement('canvas');
  canvas.width = 8;
  canvas.height = 8;
  const gl = canvas.getContext('webgl2', {
    antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error('WebGL2 컨텍스트 생성 실패');
  wgl = new Webgl2Engine(gl, {});
  await wgl.load(PLAN, WEIGHTS);
  // 순환쌍 자동 감지 + 0 초기화 (compare.js와 같은 레시피)
  const outNames = new Set(wgl.plan.outputs.map((o) => o.name));
  wglRecurrent = [];
  for (const inp of wgl.plan.inputs) {
    if (inp.name === 'input_1') continue;
    const on = outNames.has(inp.name + 'o')
      ? inp.name + 'o'
      : outNames.has(inp.name + '_out') ? inp.name + '_out' : null;
    if (on) {
      const [, c, h, w] = inp.shape;
      wgl.upload(inp.name, new Float32Array(c * h * w));
      wglRecurrent.push([inp.name, on]);
    }
  }
}

// webgl2 엔진의 pha는 SPAN4 레이아웃(위상 텍스처 4장)이다. CPU로 꺼내지 않고
// 그 4장을 샘플링해 알파 캔버스에 바로 그린다 — readPixels(동기 스톨 4회) 제거.
// 픽셀 x → 위상 x%4, 열 x/4 (엔진의 위상 규약 그대로).
const WGL_VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const WGL_FS = `#version 300 es
precision highp float;
uniform sampler2D T0, T1, T2, T3;
uniform int uH, uWQ, uOff;
out vec4 o;
void main() {
  ivec2 xy = ivec2(gl_FragCoord.xy);
  // 엔진 규약: 픽셀 x → 위상 p = x % SPAN, 열 xq = x / SPAN
  // 텍스처 x 인덱스 = (레이어 + off) * WQ + xq   (pha는 C=1이라 레이어 0)
  int phase = xy.x & 3;
  int col = xy.x >> 2;
  // gl_FragCoord.y는 아래에서 위로 센다 — 텐서 행과 맞추려면 뒤집는다
  int py = uH - 1 - xy.y;
  ivec2 t = ivec2(uOff * uWQ + col, py);
  vec4 v;
  if (phase == 0) v = texelFetch(T0, t, 0);
  else if (phase == 1) v = texelFetch(T1, t, 0);
  else if (phase == 2) v = texelFetch(T2, t, 0);
  else v = texelFetch(T3, t, 0);
  float a = clamp(v.r, 0.0, 1.0);   // pha는 C=1 → 레인 0
  o = vec4(a, a, a, 1.0);           // 불투명 그레이스케일 (양쪽 엔진 동일 규약)
}`;

let wglPresent = null;

function makeWglPresent(gl, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  // 엔진과 **같은** GL 컨텍스트여야 텍스처를 공유한다 → 엔진 캔버스를 키운다
  gl.canvas.width = w;
  gl.canvas.height = h;
  const sh = (type, src) => {
    const s2 = gl.createShader(type);
    gl.shaderSource(s2, src);
    gl.compileShader(s2);
    if (!gl.getShaderParameter(s2, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s2));
    return s2;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, WGL_VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, WGL_FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const vao = gl.createVertexArray();
  return { prog, vao, gl };
}

function wglDrawAlpha(rec) {
  const { gl, prog, vao } = wglPresent;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, rec.W, rec.H);
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  for (let p = 0; p < 4; p++) {
    gl.activeTexture(gl.TEXTURE0 + p);
    gl.bindTexture(gl.TEXTURE_2D, rec.texs[p]);
    gl.uniform1i(gl.getUniformLocation(prog, `T${p}`), p);
  }
  gl.uniform1i(gl.getUniformLocation(prog, 'uH'), rec.H);
  gl.uniform1i(gl.getUniformLocation(prog, 'uWQ'), rec.WQ);
  gl.uniform1i(gl.getUniformLocation(prog, 'uOff'), rec.off | 0);
  gl.disable(gl.BLEND);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  wgl.cur.fbo = null; // 엔진의 FBO 캐시를 무효화 (우리가 바인딩을 건드렸다)
}

function drawBackground(ctx, w, h) {
  const mode = $('bg').value;
  if (mode === 'black') {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
  } else if (mode === 'gradient') {
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#1e3a5f');
    g.addColorStop(1, '#7b2d5e');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  } else {
    // filter:blur()는 이미지 밖을 투명으로 보고 샘플링해 테두리가 빠진다 —
    // 블러 반경만큼 확대해 그려 그 띠를 화면 밖으로 밀어낸다.
    const r = Math.max(8, Math.round(Math.min(w, h) * 0.02));
    const over = r * 3;
    ctx.filter = `blur(${r}px)`;
    ctx.drawImage(snap, -over, -over, w + over * 2, h + over * 2);
    ctx.filter = 'none';
  }
}

function alphaToMask(pha, mctx, mcanvas) {
  const img = mctx.createImageData(io.w, io.h);
  const d = img.data;
  const mask = $('showMask').checked;
  for (let i = 0, j = 0; i < pha.length; i++, j += 4) {
    const a = Math.max(0, Math.min(1, pha[i]));
    if (mask) {
      d[j] = d[j + 1] = d[j + 2] = a * 255;
      d[j + 3] = 255;
    } else {
      d[j] = d[j + 1] = d[j + 2] = 255;
      d[j + 3] = a * 255;
    }
  }
  mctx.putImageData(img, 0, 0);
  return mcanvas;
}

// 마스크 업스케일 품질. 알파는 모델 해상도(256×144)라 표시 해상도로 5배 확대된다.
// 캔버스 기본 보간은 'low'라 계단이 그대로 남는다 — 'high'로 올리고 확대 배율에
// 비례한 약한 블러로 경계를 페더링한다. (프로덕션 webgl2는 GPU 텍스처 필터링 +
// 시간 EMA로 같은 효과를 낸다.)
function drawMaskScaled(ctx, mcanvas, w, h) {
  const scale = w / mcanvas.width;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const blur = Math.max(0, scale * 0.5);
  ctx.filter = blur > 0.5 ? `blur(${blur.toFixed(1)}px)` : 'none';
  ctx.drawImage(mcanvas, 0, 0, w, h);
  ctx.filter = 'none';
}

// 마스크가 **불투명 그레이스케일**이라 알파 마스킹(destination-in)을 쓸 수 없다.
// 대신 휘도로 정확히 같은 식을 만든다:  out = bg·(1-m) + fg·m
//   fg층 = snap × m          (multiply)
//   bg층 = 배경 × (1-m)      ((1-m)은 흰색과 difference로 만든다)
//   합    = fg층 + bg층      (lighter)
// 전부 캔버스 GPU 합성이라 리드백이 없다.
function composite(ctx, mcanvas, fgCanvas, fgCtx, w, h) {
  if ($('showMask').checked) {
    ctx.clearRect(0, 0, w, h);
    drawMaskScaled(ctx, mcanvas, w, h);
    return;
  }
  // 1-m
  invCtx.globalCompositeOperation = 'source-over';
  invCtx.fillStyle = '#fff';
  invCtx.fillRect(0, 0, w, h);
  invCtx.globalCompositeOperation = 'difference';
  drawMaskScaled(invCtx, mcanvas, w, h);
  invCtx.globalCompositeOperation = 'source-over';

  // fg층 = snap × m
  fgCtx.globalCompositeOperation = 'source-over';
  fgCtx.drawImage(snap, 0, 0, w, h);
  fgCtx.globalCompositeOperation = 'multiply';
  drawMaskScaled(fgCtx, mcanvas, w, h);
  fgCtx.globalCompositeOperation = 'source-over';

  // bg층 = 배경 × (1-m)  — 배경은 최종 캔버스에 그린 뒤 곱한다
  drawBackground(ctx, w, h);
  ctx.globalCompositeOperation = 'multiply';
  ctx.drawImage(inv, 0, 0);
  // 합
  ctx.globalCompositeOperation = 'lighter';
  ctx.drawImage(fgCanvas, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
}

let fps = 0, fpsT0 = performance.now(), benchPause = 0;

async function frame() {
  if (!running) return;
  const t0 = performance.now();
  const w = outA.width, h = outA.height;

  // 1) 이 순간의 프레임을 고정하고(스냅샷) 그것만 쓴다 — 마스크와 색의 시점 일치
  snapCtx.drawImage(video, 0, 0, w, h);
  smallCtx.drawImage(snap, 0, 0, io.w, io.h);
  const px = smallCtx.getImageData(0, 0, io.w, io.h).data;
  const plane = io.w * io.h;
  for (let p = 0, q = 0; p < px.length; p += 4, q += 3) {
    const r = px[p] / 255, g = px[p + 1] / 255, b = px[p + 2] / 255;
    rgbNhwc[q] = r; rgbNhwc[q + 1] = g; rgbNhwc[q + 2] = b;   // 인터리브
    const i = q / 3;
    rgbNchw[i] = r; rgbNchw[plane + i] = g; rgbNchw[2 * plane + i] = b; // 플래너
  }

  // 2) ai_engine — 마스크를 GPU에 둔 채 캔버스에 바로 그린다 (리드백 없음)
  const ta0 = performance.now();
  try {
    await infer_and_present(rgbNhwc, 'pha', 0);
  } catch (e) {
    running = false;
    say(`ai_engine 추론 실패: ${e}`, true);
    return;
  }
  push(tA, performance.now() - ta0);

  // 3) webgl2 — 단계별로 쪼개 잰다. "느리다"를 주장하기 전에 어디가 느린지 봐야 한다.
  const tb0 = performance.now();
  wgl.upload('input_1', rgbNchw);
  const tUp = performance.now();
  wgl.run();
  swapWglState();
  wglDrawAlpha(wgl.tex.get('pha')); // GPU에서 바로 알파 캔버스로 (리드백 없음)
  const tRun = performance.now();
  push(tB, tRun - tb0);
  push(tBup, tUp - tb0);
  push(tBrun, tRun - tUp);

  // 4) 합성 — 마스크 소스는 각 엔진이 GPU에서 그린 캔버스 그대로
  composite(ctxA, maskA, fgA, fgACtx, w, h);
  composite(ctxB, wgl.gl.canvas, fgB, fgBCtx, w, h);

  fps++;
  const now = performance.now();
  if (now - fpsT0 >= 1000) {
    const f = fps / ((now - fpsT0 - benchPause) / 1000);
    benchPause = 0;
    $('a-fps').textContent = `${f.toFixed(1)} fps`;
    $('b-fps').textContent = `${f.toFixed(1)} fps`;
    $('a-infer').textContent =
      `추론 ${Number.isFinite(gpuMsA) ? gpuMsA.toFixed(2) : '–'} ms · 제출 ${median(tA).toFixed(2)} ms`;
    $('b-infer').textContent = `${median(tB).toFixed(2)} ms`;
    console.log(
      `AI_ENGINE_RESULT: ab fps ${f.toFixed(1)} 추론 ours ${gpuMsA.toFixed(2)}ms ` +
        `webgl2 ${gpuMsB.toFixed(2)}ms | 제출 ours ${median(tA).toFixed(2)}ms webgl2 ${median(tB).toFixed(2)}ms`
    );
    $('b-infer').textContent =
      `추론 ${Number.isFinite(gpuMsB) ? gpuMsB.toFixed(2) : '–'} ms · 제출 ${median(tB).toFixed(2)} ms`;
    fps = 0;
    fpsT0 = now;
  }

  if (performance.now() - lastAuto > AUTO_EVERY_MS) {
    lastAuto = performance.now();
    const tb = performance.now();
    await autoBench();
    // 측정에 쓴 시간은 fps 창에서 제외한다 (측정이 fps를 깎아먹지 않게)
    benchPause += performance.now() - tb;
  }

  const wait = Math.max(0, FRAME_MS - (performance.now() - t0));
  setTimeout(() => requestAnimationFrame(frame), wait);
}

async function start() {
  $('start').disabled = true;
  try {
    if (!navigator.gpu) throw new Error('WebGPU 미지원 브라우저');
    say('엔진 초기화 중…');
    await initWasm();
    await init_engine();

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

    const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
    for (const c of [outA, outB]) { c.width = vw; c.height = vh; }
    snap = new OffscreenCanvas(vw, vh); snapCtx = snap.getContext('2d');
    inv = new OffscreenCanvas(vw, vh); invCtx = inv.getContext('2d');
    fgA = new OffscreenCanvas(vw, vh); fgACtx = fgA.getContext('2d');
    fgB = new OffscreenCanvas(vw, vh); fgBCtx = fgB.getContext('2d');

    say('ai_engine 모델 로드 중…');
    await loadOurs();
    say('webgl2 엔진 로드 중…');
    await loadWebgl2();

    small = new OffscreenCanvas(io.w, io.h);
    smallCtx = small.getContext('2d', { willReadFrequently: true });
    // 우리 마스크는 WebGPU 서피스 캔버스에 직접 그려진다 (OffscreenCanvas 아님)
    maskA = document.createElement('canvas');
    maskA.width = io.w; maskA.height = io.h;
    attach_canvas(maskA);
    wglPresent = makeWglPresent(wgl.gl, io.w, io.h);
    rgbNhwc = new Float32Array(io.h * io.w * 3);
    rgbNchw = new Float32Array(io.h * io.w * 3);

    say(`실행 중 — ${io.w}×${io.h}, 두 엔진 동일 입력`);
    running = true;
    $('stop').disabled = false;
    requestAnimationFrame(frame);
  } catch (e) {
    say(`시작 실패: ${e.message ?? e}`, true);
    $('start').disabled = false;
  }
}

function stop() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $('start').disabled = false;
  $('stop').disabled = true;
  say('정지됨');
}

// ---- 등가 GPU 시간 측정 ----
// 프레임 루프의 ms는 "CPU가 명령을 다 낸 시점"이라 두 엔진을 비교할 수 없다.
// 여기서는 **양쪽 동일 방법론**으로 잰다: 입력 1회 업로드 → N프레임 연속 실행 →
// 마지막에 한 번만 동기화(ours: wait_idle / webgl2: gl.finish) → 총시간/N.
// 상태 순환이 프레임 간 의존을 만들어 실제로 직렬 실행된다.
const BENCH_FRAMES = 50;
// 자동 측정은 렌더 루프를 **멈춘다** (동기화가 필요하니 불가피). 그래서
// ① 프레임 수를 최소로, ② 간격을 넓게, ③ 한 번에 한 엔진씩 번갈아 재서
// 한 번의 정지를 절반으로 줄인다. 실측: 15프레임×2엔진 = 80~95ms 정지 →
// 그 1초 창의 fps가 28 → 25로 떨어졌다. 측정이 측정 대상을 망가뜨리면 안 된다.
const AUTO_FRAMES = 8;
const AUTO_EVERY_MS = 6000;
let lastAuto = 0;
let autoTurn = 0;            // 0 = ai_engine, 1 = webgl2 번갈아
let gpuMsA = NaN, gpuMsB = NaN;

// 라이브 중 주기적으로 **각 엔진을 단독으로** 재서 진짜 GPU 추론 시간을 갱신한다.
// 프레임 루프의 ms는 "CPU가 명령을 낸 시점"이라 추론 시간이 아니다 — 두 엔진을
// 비교하려면 둘 다 마지막에 한 번 동기화하는 같은 방법론으로 재야 한다.
async function autoBench() {
  try {
    if (autoTurn === 0) {
      gpuMsA = await bench_current(AUTO_FRAMES);
    } else {
      wgl.gl.finish();
      const t0 = performance.now();
      for (let i = 0; i < AUTO_FRAMES; i++) { wgl.run(); swapWglState(); }
      wgl.gl.finish();
      gpuMsB = (performance.now() - t0) / AUTO_FRAMES;
    }
    autoTurn ^= 1;
  } catch (_) { /* 측정 실패는 무시 — 표시만 안 갱신 */ }
}

async function runBench() {
  if (!running) {
    say('먼저 시작하세요', true);
    return;
  }
  const wasRunning = running;
  running = false;                       // 프레임 루프 정지 (측정 오염 방지)
  await new Promise((r) => setTimeout(r, 120));
  $('benchout').textContent = '측정 중…';
  try {
    // ai_engine — 내부에서 워밍업 3 + N프레임 + wait_idle 1회
    const a = await model_bench(BENCH_FRAMES);

    // webgl2 — 같은 방법론
    for (let i = 0; i < 3; i++) { wgl.run(); swapWglState(); }
    wgl.gl.finish();
    const t0 = performance.now();
    for (let i = 0; i < BENCH_FRAMES; i++) { wgl.run(); swapWglState(); }
    wgl.gl.finish();
    const b = (performance.now() - t0) / BENCH_FRAMES;

    const line =
      `등가 GPU 시간 (${BENCH_FRAMES}프레임, 마지막 1회만 동기화): ` +
      `ai_engine ${a.ms_per_frame.toFixed(2)}ms / webgl2 ${b.toFixed(2)}ms ` +
      `→ ${(a.ms_per_frame / b).toFixed(2)}배`;
    $('benchout').textContent = line;
    console.log(`AI_ENGINE_RESULT: gpubench ours ${a.ms_per_frame.toFixed(2)}ms webgl2 ${b.toFixed(2)}ms`);
  } catch (e) {
    $('benchout').textContent = `측정 실패: ${e}`;
  }
  running = wasRunning;
  if (running) requestAnimationFrame(frame);
}

function swapWglState() {
  for (const [a, b] of wglRecurrent) {
    const ta = wgl.tex.get(a), tb = wgl.tex.get(b);
    wgl.tex.set(a, tb);
    wgl.tex.set(b, ta);
  }
}

$('bench').addEventListener('click', runBench);
$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);
