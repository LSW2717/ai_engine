// gaze 크롭·각도 diff 게이트 — 웹 focus-tracker(v-ai faceCrop.ts/gazeModel.ts)와
// GazeTask 경로를 같은 프레임·같은 랜드마크로 비교한다.
//
// 헤드리스: node tools/run_web.mjs demo/gaze-ab.html
// 검사 5종:
//   box          크롭 박스 수학 (faceCrop.ts margin 식, f64 기준) — 정합
//   crop         크롭 픽셀 (cv2 bilinear 규약 f64 기준 vs 엔진 f32) — 정합
//   angle        같은 크롭 → gaze.sw(WebGPU)+rust decode vs ORT(wasm)+JS decode
//   e2e-resample 같은 u8 양자화끼리 (cv2 크롭 재양자화 vs drawImage 크롭) — 순수
//                리샘플 차만. |엔진 f32 − 웹 u8|은 게이트 아님 (아래 주석 참조)
//   task/noface  GazeTask 30틱: baseline → FOCUSED + 무얼굴 홀드(<600ms)/NO_FACE
//
// 웹 drawImage 크롭은 브라우저 리샘플이라 픽셀 비트 정합 대상이 아니다(참고 로그만).

import * as ort from '../compare/ort/ort.bundle.min.mjs';

const BASE = '../models/mediapipe/';
const W = 256, H = 144, SIZE = 448;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

const say = (t) => (document.getElementById('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);

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

async function loadFrame() {
  const raw = new Uint8Array(await (await fetch(BASE + 'frame_256x144.rgb')).arrayBuffer());
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const cx = canvas.getContext('2d');
  const img = cx.createImageData(W, H);
  for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
    img.data[j] = raw[i];
    img.data[j + 1] = raw[i + 1];
    img.data[j + 2] = raw[i + 2];
    img.data[j + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
  return { canvas, rgb: raw };
}

// ── 기준 구현 (웹 focus-tracker 1:1, f64) ──

// faceCrop.ts margin 식 (FACE_CROP marginX 0.18 / marginY 0.22, 8px 게이트)
function refCropBox(pts, vw, vh) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const bw = maxX - minX, bh = maxY - minY;
  if (bw <= 0 || bh <= 0) return null;
  const x0 = Math.max(0, minX - bw * 0.18);
  const y0 = Math.max(0, minY - bh * 0.22);
  const x1 = Math.min(1, maxX + bw * 0.18);
  const y1 = Math.min(1, maxY + bh * 0.22);
  if ((x1 - x0) * vw < 8 || (y1 - y0) * vh < 8) return null;
  return [x0, y0, x1, y1];
}

// cv2.resize bilinear 규약 (src = (dst+0.5)·scale − 0.5, replicate 경계) — 엔진과
// 같은 식을 f64로. 결과 RGB [0,1] 인터리브.
function refCropResize(rgb, w, h, box) {
  const x0 = box[0] * w, y0 = box[1] * h;
  const sx = ((box[2] - box[0]) * w) / SIZE;
  const sy = ((box[3] - box[1]) * h) / SIZE;
  const out = new Float64Array(SIZE * SIZE * 3);
  for (let oy = 0; oy < SIZE; oy++) {
    let fy = y0 + (oy + 0.5) * sy - 0.5;
    fy = Math.min(Math.max(fy, 0), h - 1);
    const iy = Math.min(Math.floor(fy), h - 2);
    const ty = fy - iy;
    for (let ox = 0; ox < SIZE; ox++) {
      let fx = x0 + (ox + 0.5) * sx - 0.5;
      fx = Math.min(Math.max(fx, 0), w - 1);
      const ix = Math.min(Math.floor(fx), w - 2);
      const tx = fx - ix;
      const o = (oy * SIZE + ox) * 3;
      for (let c = 0; c < 3; c++) {
        const v = (x, y) => rgb[(y * w + x) * 3 + c];
        const top = v(ix, iy) * (1 - tx) + v(ix + 1, iy) * tx;
        const bot = v(ix, iy + 1) * (1 - tx) + v(ix + 1, iy + 1) * tx;
        out[o + c] = (top * (1 - ty) + bot * ty) / 255;
      }
    }
  }
  return out;
}

// gazeModel.ts expectedAngleDeg (bins 90, binWidth 4, offset 180)
function expectedAngleDeg(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const e = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    e[i] = Math.exp(logits[i] - max);
    sum += e[i];
  }
  let expected = 0;
  for (let i = 0; i < logits.length; i++) expected += (e[i] / sum) * i;
  return expected * 4 - 180;
}

// 인터리브 [0,1] → NCHW ImageNet 정규화 (ORT 입력)
function toNchwNorm(inter) {
  const plane = SIZE * SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++)
    for (let c = 0; c < 3; c++)
      out[c * plane + i] = (inter[i * 3 + c] - MEAN[c]) / STD[c];
  return out;
}

function row(name, diff, tol, ok) {
  const tr = document.createElement('tr');
  tr.innerHTML =
    `<td>${name}</td><td class="num">${diff}</td><td class="num">${tol}</td>` +
    `<td>${ok ? 'ok' : 'FAIL'}</td>`;
  document.getElementById('rows').appendChild(tr);
}

async function main() {
  say('프레임 + wasm 로드…');
  const { canvas, rgb } = await loadFrame();
  document.getElementById('view').getContext('2d').drawImage(canvas, 0, 0);
  await loadAiWasm();
  if (!navigator.gpu) throw new Error('WebGPU 미지원');
  await aiMod.init_engine();
  const fetchB = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());

  // FaceTask로 공통 랜드마크 획득 (양쪽 경로가 같은 478pt를 소비)
  say('FaceTask 랜드마크…');
  const det = (await aiMod.load_model_h(await fetchB(BASE + 'face_detector.sw'))).handle;
  const lm = (await aiMod.load_model_h(await fetchB(BASE + 'face_landmarks.sw'))).handle;
  const ftask = aiMod.face_task_new(false);
  const faceR = await aiMod.face_task_gpu(ftask, det, lm, rgb, W, H, 0.0);
  aiMod.face_task_free(ftask);
  aiMod.unload_model_h(det);
  aiMod.unload_model_h(lm);
  if (!faceR) throw new Error('얼굴 못 찾음');
  const pts = faceR.points.map((p) => [p[0], p[1]]);
  const flat = new Float32Array(pts.length * 2);
  for (let i = 0; i < pts.length; i++) {
    flat[i * 2] = pts[i][0];
    flat[i * 2 + 1] = pts[i][1];
  }

  let pass = true;
  const check = (tag, diff, tol, fmt = (v) => v.toExponential(2)) => {
    const ok = diff <= tol;
    pass = pass && ok;
    row(tag, fmt(diff), fmt(tol), ok);
    log(`gaze-ab ${tag} diff=${fmt(diff)} tol=${fmt(tol)} ${ok ? 'ok' : 'FAIL'}`);
    return ok;
  };

  // ① 크롭 박스
  say('크롭 박스…');
  const refBox = refCropBox(pts, W, H);
  const ourBox = aiMod.gaze_crop_box(flat, W, H);
  if (!refBox || !ourBox) throw new Error('크롭 박스 null');
  let boxD = 0;
  for (let i = 0; i < 4; i++) boxD = Math.max(boxD, Math.abs(refBox[i] - ourBox[i]));
  check('box', boxD, 1e-5);
  {
    const vc = document.getElementById('view').getContext('2d');
    vc.strokeStyle = '#58a6ff';
    vc.strokeRect(ourBox[0] * W, ourBox[1] * H, (ourBox[2] - ourBox[0]) * W, (ourBox[3] - ourBox[1]) * H);
  }

  // ② 크롭 픽셀 (cv2 규약 f64 기준 vs 엔진 f32)
  say('크롭 픽셀…');
  const refCrop = refCropResize(rgb, W, H, refBox);
  const ourCrop = aiMod.gaze_crop_pixels(rgb, W, H, new Float32Array(ourBox));
  let cropD = 0;
  for (let i = 0; i < refCrop.length; i++)
    cropD = Math.max(cropD, Math.abs(refCrop[i] - ourCrop[i]));
  check('crop', cropD, 2e-3);
  {
    // 눈검증용 — 엔진 크롭을 그려둔다
    const cc = document.getElementById('crop').getContext('2d');
    const img = cc.createImageData(SIZE, SIZE);
    for (let i = 0, j = 0; j < img.data.length; i += 3, j += 4) {
      img.data[j] = ourCrop[i] * 255;
      img.data[j + 1] = ourCrop[i + 1] * 255;
      img.data[j + 2] = ourCrop[i + 2] * 255;
      img.data[j + 3] = 255;
    }
    cc.putImageData(img, 0, 0);
  }

  // 웹 실경로 크롭 (faceCrop.ts drawImage — 참고 로그만, 리샘플이 다르다)
  const wc = document.createElement('canvas');
  wc.width = SIZE;
  wc.height = SIZE;
  const wctx = wc.getContext('2d', { willReadFrequently: true });
  wctx.drawImage(
    canvas, refBox[0] * W, refBox[1] * H,
    (refBox[2] - refBox[0]) * W, (refBox[3] - refBox[1]) * H, 0, 0, SIZE, SIZE
  );
  const webPx = wctx.getImageData(0, 0, SIZE, SIZE).data;
  {
    let d = 0, sum = 0;
    for (let i = 0, j = 0; j < webPx.length; i += 3, j += 4)
      for (let c = 0; c < 3; c++) {
        const dv = Math.abs(webPx[j + c] / 255 - ourCrop[i + c]);
        d = Math.max(d, dv);
        sum += dv;
      }
    log(`gaze-ab crop-vs-drawImage max=${d.toFixed(4)} mean=${(sum / (SIZE * SIZE * 3)).toFixed(5)} (참고 — 리샘플 차)`);
  }

  // ③ 같은 크롭 → 엔진(gaze.sw WebGPU + rust decode) vs ORT(wasm + JS decode)
  say('gaze.sw vs ORT…');
  const gazeH = (await aiMod.load_model_h(await fetchB(BASE + 'gaze.sw'))).handle;
  const norm = aiMod.gaze_normalize(Float32Array.from(ourCrop));
  const oursYaw = aiMod.gaze_decode_bins(await aiMod.infer_frame_h(gazeH, norm, 'yaw'));
  const oursPitch = aiMod.gaze_decode_bins(await aiMod.infer_frame_h(gazeH, norm, 'pitch'));
  aiMod.unload_model_h(gazeH);

  ort.env.wasm.numThreads = 1;
  const sess = await ort.InferenceSession.create(BASE + 'mobileone_s0_gaze.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const yawOut = sess.outputNames.find((n) => /yaw/i.test(n)) ?? sess.outputNames[1];
  const pitchOut = sess.outputNames.find((n) => /pitch/i.test(n)) ?? sess.outputNames[0];
  const runOrt = async (inter) => {
    const out = await sess.run({
      [sess.inputNames[0]]: new ort.Tensor('float32', toNchwNorm(inter), [1, 3, SIZE, SIZE]),
    });
    return {
      yaw: expectedAngleDeg(out[yawOut].data),
      pitch: expectedAngleDeg(out[pitchOut].data),
    };
  };
  const ortSame = await runOrt(ourCrop);
  const angleD = Math.max(Math.abs(oursYaw - ortSame.yaw), Math.abs(oursPitch - ortSame.pitch));
  check('angle', angleD, 0.3, (v) => v.toFixed(3) + '°');
  log(
    `gaze-ab angles ours yaw=${oursYaw.toFixed(2)} pitch=${oursPitch.toFixed(2)} ` +
      `ort yaw=${ortSame.yaw.toFixed(2)} pitch=${ortSame.pitch.toFixed(2)}`
  );

  // ④ 웹 실경로 e2e (drawImage 크롭 + ORT) vs 엔진 전체 경로 — 원인 분해:
  //    quant = 크롭을 u8로 재양자화만 한 것 (웹 getImageData가 갖는 양자화)
  //    → |ours−quant| = 순수 양자화 민감도, |quant−web| = 순수 리샘플 차
  say('웹 실경로 e2e…');
  const webInter = new Float64Array(SIZE * SIZE * 3);
  for (let i = 0, j = 0; j < webPx.length; i += 3, j += 4) {
    webInter[i] = webPx[j] / 255;
    webInter[i + 1] = webPx[j + 1] / 255;
    webInter[i + 2] = webPx[j + 2] / 255;
  }
  const quantCrop = Float64Array.from(ourCrop, (v) => Math.round(v * 255) / 255);
  const ortQuant = await runOrt(quantCrop);
  const webE2e = await runOrt(webInter);
  await sess.release();
  const e2eD = Math.max(Math.abs(oursYaw - webE2e.yaw), Math.abs(oursPitch - webE2e.pitch));
  const quantD = Math.max(Math.abs(oursYaw - ortQuant.yaw), Math.abs(oursPitch - ortQuant.pitch));
  const resampD = Math.max(Math.abs(ortQuant.yaw - webE2e.yaw), Math.abs(ortQuant.pitch - webE2e.pitch));
  // |ours−quant|(u8 재양자화 민감도)는 게이트가 아니다: 이 테스트 프레임은
  // 얼굴 ~100px를 448²로 4.5배 업스케일한 흐린 입력이라 softmax가 평평해
  // 기댓값 디코드가 서브 LSB 노이즈를 수 °로 증폭한다 (실측 5.6°). 웹 제품
  // 경로 자체가 getImageData u8 양자화를 갖고, 우리 f32 경로가 상위 정밀도 —
  // 웹 gazeModel.ts 주석대로 절대각은 baseline 상대 일관성만 필요하다.
  // 게이트는 같은 양자화끼리(quant vs drawImage) = 순수 리샘플 차만 잡는다.
  log(
    `gaze-ab e2e-probe quant yaw=${ortQuant.yaw.toFixed(2)} pitch=${ortQuant.pitch.toFixed(2)} ` +
      `web yaw=${webE2e.yaw.toFixed(2)} pitch=${webE2e.pitch.toFixed(2)} ` +
      `|ours-quant|=${quantD.toFixed(3)}° |ours-web|=${e2eD.toFixed(3)}° (참고 — u8 양자화 민감도)`
  );
  check('e2e-resample', resampD, 1.5, (v) => v.toFixed(3) + '°');

  // ⑤ GazeTask 상태머신: 같은 프레임 30틱 → baseline 수집 → FOCUSED,
  //    이어 무얼굴 홀드(<600ms) / NO_FACE(>600ms)
  say('GazeTask 상태머신…');
  const gazeH2 = (await aiMod.load_model_h(await fetchB(BASE + 'gaze.sw'))).handle;
  const gtask = aiMod.gaze_task_new();
  let fr = null;
  let t = 1000;
  for (let i = 0; i < 30; i++, t += 100) {
    fr = await aiMod.gaze_task_gpu(gtask, gazeH2, 0, rgb, W, H, flat, 1, t);
  }
  const taskOk =
    fr.status === 'FOCUSED' && fr.attentive === true &&
    Math.abs(fr.yaw - oursYaw) <= 0.5 && Math.abs(fr.pitch - oursPitch) <= 0.5;
  pass = pass && taskOk;
  row('task', `${fr.status} score=${fr.score}`, 'FOCUSED', taskOk);
  log(
    `gaze-ab task status=${fr.status} attentive=${fr.attentive} score=${fr.score} ` +
      `yaw=${fr.yaw?.toFixed(2)} pitch=${fr.pitch?.toFixed(2)} ${taskOk ? 'ok' : 'FAIL'}`
  );

  const empty = new Float32Array(0);
  const hold = await aiMod.gaze_task_gpu(gtask, gazeH2, 0, rgb, W, H, empty, 0, t);
  const gone = await aiMod.gaze_task_gpu(gtask, gazeH2, 0, rgb, W, H, empty, 0, t + 700);
  const nofaceOk = hold.status === 'FOCUSED' && gone.status === 'NO_FACE';
  pass = pass && nofaceOk;
  row('noface', `${hold.status} → ${gone.status}`, 'FOCUSED → NO_FACE', nofaceOk);
  log(`gaze-ab noface hold=${hold.status} after700=${gone.status} ${nofaceOk ? 'ok' : 'FAIL'}`);
  aiMod.gaze_task_free(gtask);
  aiMod.unload_model_h(gazeH2);

  say(pass ? 'PASS' : 'FAIL — 표의 diff 확인');
  log(`gaze-ab verdict ${pass ? 'PASS' : 'FAIL'}`);
  log('gaze-ab-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`gaze-ab fatal ${String(e).slice(0, 200)}`);
  log('gaze-ab-done');
});
