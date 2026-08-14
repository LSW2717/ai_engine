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

  // ── E: 3D 아이템 — 동시 적용 + 해제 (다중 아이템 회귀 게이트) ──
  say('E items…');
  const ec = new OffscreenCanvas(W, H);
  const ectx = ec.getContext('2d', { willReadFrequently: true });
  const snapPixels = async () => {
    let r = null;
    for (let i = 0; i < 200; i++) {
      r = await sendFrame();
      if (!r.passthrough) break;
      r.bitmap.close();
      r = null;
      await sleep(40);
    }
    if (!r) return null;
    ectx.drawImage(r.bitmap, 0, 0);
    r.bitmap.close();
    return ectx.getImageData(0, 0, W, H).data;
  };
  const diffPx = (a, b) => {
    if (!a || !b) return -1;
    let n = 0;
    for (let i = 0; i < a.length; i += 4) {
      if (
        Math.abs(a[i] - b[i]) > 24 ||
        Math.abs(a[i + 1] - b[i + 1]) > 24 ||
        Math.abs(a[i + 2] - b[i + 2]) > 24
      )
        n++;
    }
    return n;
  };
  // 아이템이 나타날 때까지 렌더 (GLB/face 모델 fetch 비동기)
  const snapUntilDiff = async (ref, min) => {
    let cur = null;
    for (let i = 0; i < 120; i++) {
      cur = await snapPixels();
      if (cur && diffPx(ref, cur) >= min) return cur;
      await sleep(80);
    }
    return cur;
  };
  const TH = 30; // "그려졌다" 판정 최소 변경 픽셀 수
  await config({ focusDetection: null }); // C의 집중도 off — 픽셀 잡음 제거
  const base = await snapPixels();
  await config({ faceEffects: { enabled: true, hat: 'hat1', eyewear: 'none', beard: 'none' } });
  const hatOnly = await snapUntilDiff(base, TH);
  const hatShown = diffPx(base, hatOnly) >= TH;
  check('item-hat', hatShown, `diff=${diffPx(base, hatOnly)}`);
  await config({ faceEffects: { enabled: true, hat: 'hat1', eyewear: 'glasses1', beard: 'none' } });
  const both = await snapUntilDiff(hatOnly, TH);
  await config({ faceEffects: { enabled: true, hat: 'none', eyewear: 'glasses1', beard: 'none' } });
  const glassesOnly = await snapUntilDiff(both, TH);
  // 동시 적용 증명: both 는 hatOnly 와도(안경 추가), glassesOnly 와도(모자 추가) 달라야 한다
  const dBoth1 = diffPx(hatOnly, both);
  const dBoth2 = diffPx(glassesOnly, both);
  check('items-multi', dBoth1 >= TH && dBoth2 >= TH, `vsHat=${dBoth1} vsGlasses=${dBoth2}`);
  // 해제 증명: 전부 끄면 base 로 복귀
  await config({ faceEffects: { enabled: false, hat: 'none', eyewear: 'none', beard: 'none' } });
  let cleared = null;
  for (let i = 0; i < 120; i++) {
    cleared = await snapPixels();
    if (cleared && diffPx(base, cleared) < TH) break;
    await sleep(80);
  }
  check('items-clear', diffPx(base, cleared) >= 0 && diffPx(base, cleared) < TH, `diff=${diffPx(base, cleared)}`);

  // ── F: mirror-only (VB 없음) — 호스트 2D 전처리 경로 (세그 불필요) ──
  say('F mirror…');
  await config({ background: null, blur: 0, brightness: 100, grayscale: 0 });
  let f0 = await sendFrame();
  for (let i = 0; i < 100 && !f0.passthrough; i++) {
    f0.bitmap.close();
    await sleep(40);
    f0 = await sendFrame();
  }
  const passOff = f0.passthrough === true;
  ectx.drawImage(f0.bitmap, 0, 0);
  f0.bitmap.close();
  const plain = ectx.getImageData(0, 0, W, H).data.slice();
  await config({ mirror: true });
  let fm = null;
  for (let i = 0; i < 100; i++) {
    fm = await sendFrame();
    if (!fm.passthrough) break;
    fm.bitmap.close();
    fm = null;
    await sleep(40);
  }
  let mirrorOk = false;
  if (fm) {
    ectx.drawImage(fm.bitmap, 0, 0);
    fm.bitmap.close();
    mirrorOk = diffPx(plain, ectx.getImageData(0, 0, W, H).data) >= TH;
  }
  await config({ mirror: false });
  let fb = null;
  for (let i = 0; i < 100; i++) {
    fb = await sendFrame();
    if (fb.passthrough) break;
    fb.bitmap.close();
    fb = null;
    await sleep(40);
  }
  check('mirror', passOff && mirrorOk && !!fb?.passthrough, `off=${passOff} flip=${mirrorOk} back=${!!fb?.passthrough}`);
  if (fb) fb.bitmap.close();

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
