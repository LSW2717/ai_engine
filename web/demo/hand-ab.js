// hand 좌표 diff 게이트 — MediaPipe HandLandmarker(wasm)와 HandTask를 같은
// 프레임으로 비교한다. 양쪽 다 hand_landmarker.task에서 나온 동일 tflite 가중치.
//
// 헤드리스: node tools/run_web.mjs demo/hand-ab.html
// 검사: ①좌표 diff (손별 21pt, 손목 최근접 매칭) ②handedness 라벨 일치
//       ③트래킹 계약 (2프레임째 디텍터 생략 — det 세션 frames 불변)
//       ④gesture export 스모크 (실손 무발화 + 합성 박수 시퀀스 발화)
//
// 테스트 프레임: web/models/mediapipe/hands.jpg (MediaPipe 공식 샘플, 두 손) —
// make convert-mediapipe가 받는다.

const BASE = '../models/mediapipe/';
const W = 320, H = 480;
// 디텍터 입력 192px가 프레임 480px(레터박스 긴 변)에 대응 → 1 det px ≈ 2.5 frame px
const TOL_PX = 4.0;

const say = (t) => (document.getElementById('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);
const fmt = (v) => (v == null ? '-' : v.toFixed(2));

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
  const blob = await (await fetch(BASE + 'hands.jpg')).blob();
  const bmp = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const cx = canvas.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0, W, H);
  const d = cx.getImageData(0, 0, W, H).data;
  const rgb = new Uint8Array(W * H * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
    rgb[j] = d[i];
    rgb[j + 1] = d[i + 1];
    rgb[j + 2] = d[i + 2];
  }
  return { canvas, rgb };
}

// 통일 표현: {points: [[x,y]×21 px], handedness: 'Left'|'Right', presence}
async function runMediapipe(canvas) {
  const vision = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );
  const fileset = await vision.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  const hl = await vision.HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: BASE + 'hand_landmarker.task', delegate: 'CPU' },
    runningMode: 'IMAGE',
    numHands: 2,
  });
  const res = hl.detect(canvas);
  hl.close();
  return res.landmarks.map((lm, i) => ({
    points: lm.map((p) => [p.x * W, p.y * H]),
    handedness: res.handedness[i][0].categoryName,
    presence: res.handedness[i][0].score,
  }));
}

const toPx = (r) => ({
  points: r.points.map((p) => [p[0] * W, p[1] * H]),
  handedness: r.handedness > 0.5 ? 'Left' : 'Right',
  presence: r.presence,
});

// 손목(0번) 최근접 매칭 후 21pt diff
function diffHands(ref, got) {
  let max = 0, sum = 0, n = 0, handedOk = true;
  for (const r of ref) {
    let best = null, bestD = Infinity;
    for (const g of got) {
      const d = Math.hypot(g.points[0][0] - r.points[0][0], g.points[0][1] - r.points[0][1]);
      if (d < bestD) { bestD = d; best = g; }
    }
    if (!best) return { max: Infinity, mean: Infinity, handedOk: false };
    for (let k = 0; k < 21; k++) {
      const d = Math.hypot(best.points[k][0] - r.points[k][0], best.points[k][1] - r.points[k][1]);
      max = Math.max(max, d);
      sum += d;
      n++;
    }
    handedOk = handedOk && best.handedness === r.handedness;
  }
  return { max, mean: sum / n, handedOk };
}

function draw(view, hands, color) {
  const cx = view.getContext('2d');
  cx.fillStyle = color;
  for (const h of hands)
    for (const [x, y] of h.points) cx.fillRect(x - 1, y - 1, 2, 2);
}

function row(name, hands, d) {
  const tr = document.createElement('tr');
  const pres = hands.length ? hands.map((h) => h.presence.toFixed(2)).join('/') : '-';
  const handed = hands.map((h) => h.handedness[0]).join('/');
  tr.innerHTML =
    `<td>${name}</td><td class="num">${hands.length}</td><td class="num">${pres}</td>` +
    `<td class="num">${d ? fmt(d.max) : '기준'}</td><td class="num">${d ? fmt(d.mean) : '기준'}</td>` +
    `<td>${handed}${d ? (d.handedOk ? ' =' : ' ≠') : ''}</td>`;
  document.getElementById('rows').appendChild(tr);
}

async function main() {
  say('프레임 로드…');
  const { canvas, rgb } = await loadFrame();
  const view = document.getElementById('view');
  view.getContext('2d').drawImage(canvas, 0, 0);

  say('MediaPipe HandLandmarker…');
  const mp = await runMediapipe(canvas);
  row('MediaPipe (기준)', mp, null);
  draw(view, mp, '#2da44e');
  log(`hand-ab mp n=${mp.length} handed=${mp.map((h) => h.handedness).join(',')}`);

  say('ai_engine wasm 로드…');
  await loadAiWasm();
  const fetchB = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const detB = await fetchB(BASE + 'hand_detector.sw');
  const lmB = await fetchB(BASE + 'hand_landmarks.sw');

  let pass = mp.length >= 1;
  if (!pass) log('hand-ab FAIL — MediaPipe가 손을 못 찾음 (프레임 확인)');
  const verdicts = [];

  for (const [tag, color, run] of [
    ['cpu', '#d1242f', async () => {
      const det = aiMod.load_model_cpu_h(detB).handle;
      const lm = aiMod.load_model_cpu_h(lmB).handle;
      const task = aiMod.hand_task_new(2);
      const r1 = aiMod.hand_task_cpu(task, det, lm, rgb, W, H, 0.0);
      // 트래킹 계약: 손을 다 찾았으면 2프레임째는 디텍터 생략
      const f1 = aiMod.model_stats_cpu_h(det).frames;
      const r2 = aiMod.hand_task_cpu(task, det, lm, rgb, W, H, 33.0);
      const f2 = aiMod.model_stats_cpu_h(det).frames;
      aiMod.hand_task_free(task);
      aiMod.unload_model_cpu_h(det);
      aiMod.unload_model_cpu_h(lm);
      return { r1, r2, detSkipped: r1.length >= 2 ? f2 === f1 : true };
    }],
    ['gpu', '#58a6ff', async () => {
      await aiMod.init_engine();
      const det = (await aiMod.load_model_h(detB)).handle;
      const lm = (await aiMod.load_model_h(lmB)).handle;
      const task = aiMod.hand_task_new(2);
      const r1 = await aiMod.hand_task_gpu(task, det, lm, rgb, W, H, 0.0);
      const f1 = aiMod.model_stats_h(det).frames;
      const r2 = await aiMod.hand_task_gpu(task, det, lm, rgb, W, H, 33.0);
      const f2 = aiMod.model_stats_h(det).frames;
      aiMod.hand_task_free(task);
      aiMod.unload_model_h(det);
      aiMod.unload_model_h(lm);
      return { r1, r2, detSkipped: r1.length >= 2 ? f2 === f1 : true };
    }],
  ]) {
    say(`HandTask ${tag}…`);
    try {
      const { r1, r2, detSkipped } = await run();
      const ours = r1.map(toPx);
      const d = diffHands(mp, ours);
      row(`ai_engine ${tag}`, ours, d);
      draw(view, ours, color);
      const ok =
        ours.length === mp.length && d.max <= TOL_PX && d.handedOk && detSkipped &&
        r2.length === r1.length;
      pass = pass && ok;
      verdicts.push(`${tag}=${ok ? 'ok' : 'FAIL'}`);
      log(
        `hand-ab ${tag} n=${ours.length} dpx max=${fmt(d.max)} mean=${fmt(d.mean)} ` +
          `handed=${d.handedOk ? 'ok' : 'MISMATCH'} track_skip=${detSkipped} n2=${r2.length}`
      );
    } catch (e) {
      pass = false;
      verdicts.push(`${tag}=ERR`);
      log(`hand-ab ${tag} ERR ${String(e).slice(0, 150)}`);
    }
  }

  // ── gesture export 스모크 ──
  // ①실손 2개(떨어져 있음) → clap 무발화 ②합성 접근→접촉 시퀀스 → 발화
  say('gesture export…');
  try {
    const g = aiMod.gesture_new();
    // 합성 손: 팜폭 0.1, 손끝이 서로를 향한 좌우 손 (gesture.rs 테스트와 동일 기하)
    const synth = (cx, dir) => {
      const l = new Array(21).fill(0).map(() => [cx, 0.5]);
      for (const i of [4, 8, 12, 16, 20]) l[i] = [cx + dir * 0.05, 0.5];
      l[5] = [cx - 0.05, 0.55];
      l[17] = [cx + 0.05, 0.55];
      l[9] = [cx, 0.55];
      l[13] = [cx, 0.55];
      return l;
    };
    const pair = (d) => {
      const centers = d * 0.1 + 0.1;
      const flat = new Float32Array(84);
      const put = (off, lm) => lm.forEach((p, i) => { flat[off + i * 2] = p[0]; flat[off + i * 2 + 1] = p[1]; });
      put(0, synth(0.5 - centers / 2, 1));
      put(42, synth(0.5 + centers / 2, -1));
      return flat;
    };
    const handed = new Float32Array([0.9, 0.1]); // Left, Right
    let fired = false;
    let t = 0;
    for (const d of [2.0, 1.6, 1.2, 0.8, 0.3]) {
      const ev = aiMod.gesture_classify(g, pair(d), handed, t);
      fired = fired || ev.some((e) => e.gesture === 'clap');
      t += 33;
    }
    // 실손(떨어져 있는 두 손)은 무발화여야 한다
    aiMod.gesture_reset(g);
    let realFired = false;
    if (mp.length >= 2) {
      const flat = new Float32Array(mp.length * 42);
      const hd = new Float32Array(mp.length);
      mp.forEach((h, hi) => {
        hd[hi] = h.handedness === 'Left' ? 0.9 : 0.1;
        h.points.forEach((p, i) => {
          flat[hi * 42 + i * 2] = p[0] / W;
          flat[hi * 42 + i * 2 + 1] = p[1] / H;
        });
      });
      for (let i = 0; i < 5; i++) {
        const ev = aiMod.gesture_classify(g, flat, hd, i * 33);
        realFired = realFired || ev.some((e) => e.gesture === 'clap');
      }
    }
    aiMod.gesture_free(g);
    const ok = fired && !realFired;
    pass = pass && ok;
    verdicts.push(`gesture=${ok ? 'ok' : 'FAIL'}`);
    log(`hand-ab gesture synth_clap=${fired} real_nofire=${!realFired}`);
  } catch (e) {
    pass = false;
    verdicts.push('gesture=ERR');
    log(`hand-ab gesture ERR ${String(e).slice(0, 150)}`);
  }

  say(pass ? `PASS (tol ${TOL_PX}px)` : 'FAIL — 표의 diff 확인');
  log(`hand-ab verdict ${pass ? 'PASS' : 'FAIL'} tol=${TOL_PX}px ${verdicts.join(' ')}`);
  log('hand-ab-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`hand-ab fatal ${String(e).slice(0, 200)}`);
  log('hand-ab-done');
});
