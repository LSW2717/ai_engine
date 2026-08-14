// vb-engine-gate.js — VbEngine 3함수 계약 게이트 (진짜 Worker 경계):
//   A passthrough: 기본 config → 반환 = **입력과 동일 객체** (제로카피)
//   B render: 단색 배경 → passthrough:false + 출력 픽셀 = 배경색 + 크기 보존
//   C focus: focusDetection on (얼굴 y4m 카메라) → FOCUSED 도달
//   D warm destroy: destroy → 같은 config 재적용 → 즉시 렌더 (재로드 없이)
// 재현: node tools/run_web.mjs demo/vb-engine.html --camera --long \
//        --video=web/models/mediapipe/face_256x144.y4m
const $ = (id) => document.getElementById(id);
const say = (t) => ($('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);

let pass = true;
const verdicts = [];
const check = (name, ok, detail = '') => {
  pass = pass && ok;
  verdicts.push(`${name}=${ok ? 'ok' : 'FAIL'}`);
  log(`vbengine ${name} ${ok ? 'ok' : 'FAIL'} ${detail}`);
};

async function main() {
  say('카메라…');
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  await video.play();
  const W = video.videoWidth || 256;
  const H = video.videoHeight || 144;
  const src = new OffscreenCanvas(W, H);
  const sctx = src.getContext('2d');

  say('워커 기동…');
  const worker = new Worker('./vb-worker.js', { type: 'module' });
  const waitMsg = (type) =>
    new Promise((res) => {
      const h = (e) => {
        if (e.data.type === type) {
          worker.removeEventListener('message', h);
          res(e.data);
        }
      };
      worker.addEventListener('message', h);
    });
  await waitMsg('ready');

  let seq = 0;
  const sendFrame = async () => {
    sctx.drawImage(video, 0, 0, W, H);
    const bitmap = await createImageBitmap(src);
    const p = waitMsg('frame');
    worker.postMessage({ type: 'frame', bitmap, time: performance.now() / 1000, seq: seq++ }, [bitmap]);
    return p;
  };
  const config = (c) => {
    const p = waitMsg('configured');
    worker.postMessage({ type: 'config', config: c });
    return p;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ── A: passthrough 제로카피 (엔진 로드 전에도, 로드 후에도 동일해야) ──
  say('A passthrough…');
  await config({ blur: 0, brightness: 100, grayscale: 0, background: null });
  const a = await sendFrame();
  check('passthrough', a.passthrough === true && a.same === true, `same=${a.same}`);
  a.bitmap.close();

  // ── B: 렌더 — 단색 배경 (VBOptions 단위: blur 0..100 등) ──
  say('B render…');
  await config({ background: '#00a05a' });
  let b = null;
  for (let i = 0; i < 300; i++) {
    b = await sendFrame();
    if (!b.passthrough) break;
    b.bitmap.close();
    b = null;
    await sleep(50); // wasm init + 모델 fetch 대기 (v-ai '로드 중 passthrough' 규약)
  }
  if (!b) {
    check('render', false, '300프레임 내 미합성');
  } else {
    const ok_dims = b.bitmap.width === W && b.bitmap.height === H;
    const view = $('view');
    view.width = W;
    view.height = H;
    const vc = view.getContext('2d');
    vc.drawImage(b.bitmap, 0, 0);
    const px = vc.getImageData(Math.floor(W * 0.95), Math.floor(H * 0.07), 1, 1).data;
    const ok_bg = px[1] > 90 && px[1] > px[0] && px[1] > px[2];
    check('render', ok_dims && ok_bg, `dims=${b.bitmap.width}x${b.bitmap.height} px=(${px[0]},${px[1]},${px[2]})`);
    b.bitmap.close();
  }

  // ── C: 집중도 (우리 확장 키 — 심이 그대로 통과) ──
  say('C focus…');
  await config({ focusDetection: { enabled: true, detectFps: 30 } });
  let focus = null;
  for (let i = 0; i < 120; i++) {
    const f = await sendFrame();
    f.bitmap.close();
    const q = waitMsg('focus');
    worker.postMessage({ type: 'focus' });
    focus = (await q).json;
    if (focus && focus.includes('FOCUSED')) break;
    await sleep(60);
  }
  check('focus', !!focus && focus.includes('"status":"FOCUSED"'), focus ?? 'null');

  // ── D: 웜 destroy → 재가동 (재로드 없이 곧 렌더) ──
  say('D warm destroy…');
  const pd = waitMsg('destroyed');
  worker.postMessage({ type: 'destroy' });
  await pd;
  await config({ background: '#00a05a' });
  let d = null;
  for (let i = 0; i < 60; i++) {
    d = await sendFrame();
    if (!d.passthrough) break;
    d.bitmap.close();
    d = null;
    await sleep(30);
  }
  check('warm', !!d, d ? '재합성 ok' : '60프레임 내 미복귀');
  if (d) d.bitmap.close();

  say(pass ? 'PASS' : 'FAIL');
  log(`vbengine verdict ${pass ? 'PASS' : 'FAIL'} ${verdicts.join(' ')}`);
  log('vbengine-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`vbengine fatal ${String(e).slice(0, 200)}`);
  log('vbengine-done');
});
