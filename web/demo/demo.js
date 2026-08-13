// 카메라 → RVM 매팅 → 가상배경 합성. 30fps 목표.
//
// 경로: video → 256×144 캔버스 → ImageData → f32 NHWC(RGB, [0,1]) → infer_frame()
//      → pha(144×256 f32) → 알파 이미지로 만들어 원본 해상도로 업스케일 합성.
// 추론만 GPU이고 프레임 입출력은 CPU 왕복이다. 프로덕션은 카메라 텍스처를 GPU에
// 그대로 물려 왕복을 없애야 하지만(webgl2 엔진이 그렇게 한다), 여기서는 엔진 자체를
// 눈으로 확인하는 게 목적이라 가장 단순한 경로를 쓴다.

import initWasm, { init_engine, load_model, model_io, infer_frame } from '../pkg/ai_engine.js';

const TARGET_FPS = 30;
const FRAME_MS = 1000 / TARGET_FPS;

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const say = (t, err = false) => {
  statusEl.textContent = t;
  statusEl.classList.toggle('err', err);
};

const out = $('out');
const ctx2d = out.getContext('2d');

let stream = null;
let video = null;
let running = false;
let io = null; // {h, w, c, outputs}
let small = null; // 모델 입력 크기 캔버스
let smallCtx = null;
let maskCanvas = null; // 알파를 담을 캔버스 (모델 해상도)
let maskCtx = null;
let bgCanvas = null;
let bgCtx = null;
let rgbBuf = null; // 재사용 Float32Array — 프레임마다 할당하면 GC가 튄다
// 프레임 스냅샷 — 마스크를 계산한 그 순간의 픽셀로 합성해야 한다.
// 합성 때 live video를 다시 그리면 추론 시간만큼 색이 앞서가 마스크가 밀려 보인다.
let snap = null, snapCtx = null;

const inferTimes = [];
const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
};

async function loadSelectedModel() {
  const url = $('model').value;
  say(`모델 로드 중… ${url.split('/').pop()}`);
  const t0 = performance.now();
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} 없음 (${resp.status}) — make convert 로 .sw를 만들어야 합니다`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const report = await load_model(bytes);
  const ms = performance.now() - t0;
  io = model_io();
  $('s-load').textContent = ms.toFixed(0);
  $('s-ops').textContent = `${report.ops} / ${report.unique_pipelines}`;

  // 모델 해상도에 맞춘 작업 캔버스들
  small = new OffscreenCanvas(io.w, io.h);
  smallCtx = small.getContext('2d', { willReadFrequently: true });
  maskCanvas = new OffscreenCanvas(io.w, io.h);
  maskCtx = maskCanvas.getContext('2d');
  rgbBuf = new Float32Array(io.h * io.w * io.c);
  inferTimes.length = 0;
  say(`준비됨 — ${io.w}×${io.h}, 출력 [${io.outputs.join(', ')}], 로드 ${ms.toFixed(0)}ms`);
  return report;
}

function pickAlphaOutput() {
  // RVM은 pha(알파)와 fgr(전경)을 낸다. 알파만 쓴다.
  return io.outputs.includes('pha') ? 'pha' : io.outputs[0];
}

function drawBackground(w, h) {
  const mode = $('bg').value;
  if (mode === 'none') {
    ctx2d.clearRect(0, 0, w, h);
    return;
  }
  if (mode === 'black') {
    ctx2d.fillStyle = '#000';
    ctx2d.fillRect(0, 0, w, h);
    return;
  }
  if (mode === 'gradient') {
    const g = ctx2d.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#1e3a5f');
    g.addColorStop(1, '#7b2d5e');
    ctx2d.fillStyle = g;
    ctx2d.fillRect(0, 0, w, h);
    return;
  }
  // 원본 블러 — 배경 이미지 없이도 가상배경 느낌을 확인할 수 있다.
  // canvas의 filter:blur()는 이미지 **바깥**을 투명으로 보고 샘플링해서 가장자리가
  // 빠진다(테두리 비네팅). 블러 반경만큼 확대해 그려 그 빠지는 띠를 화면 밖으로 밀어낸다.
  const r = Math.max(8, Math.round(Math.min(w, h) * 0.02)); // 해상도 비례 반경
  const over = r * 3; // 여유 있게 오버스캔
  ctx2d.filter = `blur(${r}px)`;
  ctx2d.drawImage(snap, -over, -over, w + over * 2, h + over * 2);
  ctx2d.filter = 'none';
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

async function frame() {
  if (!running) return;
  const tFrame = performance.now();

  const w = out.width;
  const h = out.height;

  // 1) 이 순간의 프레임을 고정하고(스냅샷) 그것만 쓴다 — 마스크와 색의 시점 일치
  snapCtx.drawImage(video, 0, 0, w, h);
  smallCtx.drawImage(snap, 0, 0, io.w, io.h);
  const img = smallCtx.getImageData(0, 0, io.w, io.h).data;
  for (let p = 0, q = 0; p < img.length; p += 4, q += 3) {
    rgbBuf[q] = img[p] / 255;
    rgbBuf[q + 1] = img[p + 1] / 255;
    rgbBuf[q + 2] = img[p + 2] / 255;
  }

  // 2) 추론
  const t0 = performance.now();
  let pha;
  try {
    pha = await infer_frame(rgbBuf, pickAlphaOutput());
  } catch (e) {
    running = false;
    say(`추론 실패: ${e}`, true);
    return;
  }
  const inferMs = performance.now() - t0;
  inferTimes.push(inferMs);
  if (inferTimes.length > 60) inferTimes.shift();

  // 3) 알파를 이미지로 (모델 해상도) — 업스케일은 drawImage의 바이리니어에 맡긴다
  const maskImg = maskCtx.createImageData(io.w, io.h);
  const md = maskImg.data;
  if ($('showMask').checked) {
    for (let i = 0, j = 0; i < pha.length; i++, j += 4) {
      const v = Math.max(0, Math.min(1, pha[i])) * 255;
      md[j] = md[j + 1] = md[j + 2] = v;
      md[j + 3] = 255;
    }
  } else {
    // 전경 = 축소 프레임 픽셀 + 알파. 원본 대신 축소본을 올려 쓰면 화질이 떨어지므로
    // 알파만 담고, 색은 원본 video를 destination-in으로 마스킹해 얻는다.
    for (let i = 0, j = 0; i < pha.length; i++, j += 4) {
      md[j] = md[j + 1] = md[j + 2] = 255;
      md[j + 3] = Math.max(0, Math.min(1, pha[i])) * 255;
    }
  }
  maskCtx.putImageData(maskImg, 0, 0);

  // 4) 합성
  if ($('showMask').checked) {
    ctx2d.clearRect(0, 0, w, h);
    drawMaskScaled(ctx2d, maskCanvas, w, h);
  } else {
    drawBackground(w, h);
    // 전경: 원본 해상도 프레임을 알파로 오려낸다 (오프스크린에서 마스킹 후 합성)
    bgCtx.clearRect(0, 0, w, h);
    bgCtx.drawImage(snap, 0, 0, w, h);
    bgCtx.globalCompositeOperation = 'destination-in';
    drawMaskScaled(bgCtx, maskCanvas, w, h);
    bgCtx.globalCompositeOperation = 'source-over';
    ctx2d.drawImage(bgCanvas, 0, 0);
  }

  fpsCount++;
  const totalMs = performance.now() - tFrame;
  $('s-infer').textContent = median(inferTimes).toFixed(2);
  $('s-total').textContent = totalMs.toFixed(1);

  // 30fps 페이싱 — 남는 시간은 쉬어서 브라우저에 양보한다
  const wait = Math.max(0, FRAME_MS - (performance.now() - tFrame));
  setTimeout(() => requestAnimationFrame(frame), wait);
}

// 표시 FPS는 실제 그려진 프레임을 1초 창으로 센다
let fpsCount = 0;
let fpsT0 = performance.now();
setInterval(() => {
  const now = performance.now();
  const fps = fpsCount / ((now - fpsT0) / 1000);
  $('s-fps').textContent = running ? fps.toFixed(1) : '–';
  if (running) {
    // 자동 실행기(tools/run_web.mjs --camera)가 읽는 줄
    console.log(
      `AI_ENGINE_RESULT: demo fps ${fps.toFixed(1)} infer ${median(inferTimes).toFixed(2)}ms ` +
        `model ${$('model').value.split('/').pop()}`
    );
  }
  fpsCount = 0;
  fpsT0 = now;
}, 1000);

async function start() {
  $('start').disabled = true;
  try {
    if (!navigator.gpu) throw new Error('WebGPU 미지원 브라우저');
    say('WebGPU 초기화 중…');
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

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    out.width = vw;
    out.height = vh;
    bgCanvas = new OffscreenCanvas(vw, vh);
    bgCtx = bgCanvas.getContext('2d');
    snap = new OffscreenCanvas(vw, vh);
    snapCtx = snap.getContext('2d');

    await loadSelectedModel();
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

$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);
$('model').addEventListener('change', async () => {
  if (!running) return;
  running = false;
  try {
    await loadSelectedModel();
    running = true;
    requestAnimationFrame(frame);
  } catch (e) {
    say(`모델 전환 실패: ${e.message ?? e}`, true);
  }
});
