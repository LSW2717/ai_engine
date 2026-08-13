// face_detector 좌표 diff 게이트 — MediaPipe FaceDetector(wasm)와 우리 detect
// 경로(핸들 API + ai-tasks 앵커 디코드/가중 NMS/레터박스 역투영)를 같은 프레임으로
// 비교한다. 양쪽 다 face_landmarker.task에서 나온 동일 tflite 가중치다.
//
// 헤드리스: node tools/run_web.mjs demo/face-ab.html
// 게이트: 검출 수 일치 + 박스·키포인트 diff ≤ TOL_PX → PASS

const BASE = '../models/mediapipe/';
const W = 256, H = 144;
// 디텍터 입력 128px가 프레임 256px에 대응 → 디텍터 1px = 프레임 2px.
// 서브픽셀(리사이즈 보간 차) 여유까지 3px.
const TOL_PX = 3.0;
const TOL_SCORE = 0.05;

const say = (t) => (document.getElementById('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);
const fmt = (v) => (v == null ? '-' : v.toFixed(2));

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
  return canvas;
}

// 검정 캔버스에 비율 유지 중앙 정렬로 그려 [-1,1] RGB로 — ai-tasks
// detect::letterbox와 같은 기하 (MediaPipe ImageToTensor BORDER_ZERO 등가)
function letterbox(src, dw, dh) {
  const c = document.createElement('canvas');
  c.width = dw;
  c.height = dh;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.fillStyle = '#000';
  cx.fillRect(0, 0, dw, dh);
  const scale = Math.min(dw / src.width, dh / src.height);
  const cw = src.width * scale, ch = src.height * scale;
  cx.drawImage(src, (dw - cw) / 2, (dh - ch) / 2, cw, ch);
  const px = cx.getImageData(0, 0, dw, dh).data;
  const rgb = new Float32Array(dw * dh * 3);
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
    rgb[j] = px[i] / 127.5 - 1;
    rgb[j + 1] = px[i + 1] / 127.5 - 1;
    rgb[j + 2] = px[i + 2] / 127.5 - 1;
  }
  return rgb;
}

// 통일 표현: 픽셀 좌표 {score, xmin, ymin, xmax, ymax, kp:[[x,y]×6]}
async function runMediapipe(canvas) {
  const vision = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );
  const fileset = await vision.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  const fd = await vision.FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: BASE + 'face_detector.tflite', delegate: 'CPU' },
    runningMode: 'IMAGE',
    minDetectionConfidence: 0.5,
  });
  const res = fd.detect(canvas);
  fd.close();
  return res.detections.map((d) => ({
    score: d.categories[0].score,
    xmin: d.boundingBox.originX,
    ymin: d.boundingBox.originY,
    xmax: d.boundingBox.originX + d.boundingBox.width,
    ymax: d.boundingBox.originY + d.boundingBox.height,
    kp: d.keypoints.map((k) => [k.x * W, k.y * H]),
  }));
}

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

const toPx = (d) => ({
  score: d.score,
  xmin: d.xmin * W,
  ymin: d.ymin * H,
  xmax: d.xmax * W,
  ymax: d.ymax * H,
  kp: d.keypoints.map((k) => [k[0] * W, k[1] * H]),
});

async function runOursCpu(canvas, bytes) {
  const rep = aiMod.load_model_cpu_h(bytes);
  const rgb = letterbox(canvas, 128, 128);
  const dets = aiMod.detect_cpu(rep.handle, 'face', rgb, W, H);
  aiMod.unload_model_cpu_h(rep.handle);
  return dets.map(toPx);
}

async function runOursGpu(canvas, bytes) {
  if (!navigator.gpu) throw new Error('WebGPU 미지원');
  await aiMod.init_engine();
  const rep = await aiMod.load_model_h(bytes);
  const rgb = letterbox(canvas, 128, 128);
  const dets = await aiMod.detect_gpu(rep.handle, 'face', rgb, W, H);
  aiMod.unload_model_h(rep.handle);
  return dets.map(toPx);
}

// 검출 diff — 센터 최근접 매칭 후 박스 4좌표·키포인트 6점의 px 차 최대값
function diff(ref, got) {
  const center = (d) => [(d.xmin + d.xmax) / 2, (d.ymin + d.ymax) / 2];
  let boxMax = 0, kpMax = 0, scoreMax = 0;
  for (const r of ref) {
    const [rx, ry] = center(r);
    let best = null, bestD = Infinity;
    for (const g of got) {
      const [gx, gy] = center(g);
      const dd = Math.hypot(gx - rx, gy - ry);
      if (dd < bestD) { bestD = dd; best = g; }
    }
    if (!best) return { boxMax: Infinity, kpMax: Infinity, scoreMax: Infinity };
    boxMax = Math.max(
      boxMax,
      Math.abs(best.xmin - r.xmin), Math.abs(best.ymin - r.ymin),
      Math.abs(best.xmax - r.xmax), Math.abs(best.ymax - r.ymax)
    );
    for (let k = 0; k < r.kp.length; k++) {
      kpMax = Math.max(
        kpMax,
        Math.abs(best.kp[k][0] - r.kp[k][0]),
        Math.abs(best.kp[k][1] - r.kp[k][1])
      );
    }
    scoreMax = Math.max(scoreMax, Math.abs(best.score - r.score));
  }
  return { boxMax, kpMax, scoreMax };
}

function draw(view, dets, color, dash) {
  const cx = view.getContext('2d');
  cx.strokeStyle = color;
  cx.lineWidth = 0.7;
  cx.setLineDash(dash);
  for (const d of dets) {
    cx.strokeRect(d.xmin, d.ymin, d.xmax - d.xmin, d.ymax - d.ymin);
    for (const [x, y] of d.kp) cx.strokeRect(x - 1, y - 1, 2, 2);
  }
  cx.setLineDash([]);
}

function row(name, dets, d) {
  const tr = document.createElement('tr');
  const score = dets.length ? dets[0].score.toFixed(3) : '-';
  tr.innerHTML =
    `<td>${name}</td><td class="num">${dets.length}</td><td class="num">${score}</td>` +
    `<td class="num">${d ? fmt(d.boxMax) : '기준'}</td>` +
    `<td class="num">${d ? fmt(d.kpMax) : '기준'}</td>` +
    `<td class="num">${d ? d.scoreMax.toFixed(3) : '기준'}</td>`;
  document.getElementById('rows').appendChild(tr);
}

async function main() {
  const frame = await loadFrame();
  const view = document.getElementById('view');
  view.getContext('2d').drawImage(frame, 0, 0);

  say('MediaPipe FaceDetector 실행 중…');
  const mp = await runMediapipe(frame);
  row('MediaPipe (기준)', mp, null);
  draw(view, mp, '#2da44e', []);
  log(
    `face-ab mp n=${mp.length} score=${mp[0]?.score.toFixed(3)} ` +
      `box=[${mp[0] ? [mp[0].xmin, mp[0].ymin, mp[0].xmax, mp[0].ymax].map(fmt) : ''}]`
  );

  say('ai_engine wasm 로드 중…');
  await loadAiWasm();
  const bytes = new Uint8Array(
    await (await fetch(BASE + 'face_detector.sw')).arrayBuffer()
  );

  let pass = true;
  const verdicts = [];
  for (const [name, tag, run, color, dash] of [
    ['ai_engine CPU', 'cpu', runOursCpu, '#d1242f', [3, 2]],
    ['ai_engine GPU', 'gpu', runOursGpu, '#58a6ff', [1, 2]],
  ]) {
    say(`${name} 실행 중…`);
    try {
      const ours = await run(frame, bytes);
      const d = diff(mp, ours);
      row(name, ours, d);
      draw(view, ours, color, dash);
      const ok =
        ours.length === mp.length && d.boxMax <= TOL_PX && d.kpMax <= TOL_PX &&
        d.scoreMax <= TOL_SCORE;
      pass = pass && ok;
      verdicts.push(`${tag}=${ok ? 'ok' : 'FAIL'}`);
      log(
        `face-ab ${tag} n=${ours.length} box_dpx=${fmt(d.boxMax)} ` +
          `kp_dpx=${fmt(d.kpMax)} score_d=${d.scoreMax.toFixed(3)}`
      );
    } catch (e) {
      pass = false;
      verdicts.push(`${tag}=ERR`);
      log(`face-ab ${tag} ERR ${String(e).slice(0, 150)}`);
    }
  }

  // ── 랜드마크 스테이지: MediaPipe FaceLandmarker(IMAGE) vs FaceTask ──
  // 같은 프레임에서 검출→ROI→회전 크롭→478점→역투영 전체 파이프라인 파리티
  say('MediaPipe FaceLandmarker 실행 중…');
  try {
    const vision = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );
    const fileset = await vision.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    const flm = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: BASE + 'face_landmarker.task', delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
    });
    const mpLm = flm.detect(frame).faceLandmarks[0].map((p) => [p.x * W, p.y * H]);
    flm.close();
    log(`face-ab lm-mp n=${mpLm.length}`);
    draw(view, [], '#2da44e', []); // no-op — 아래서 점만 찍는다
    const vc = view.getContext('2d');
    vc.fillStyle = '#2da44e';
    for (const [x, y] of mpLm) vc.fillRect(x - 0.4, y - 0.4, 0.8, 0.8);

    // u8 RGB 프레임 (FaceTask는 픽셀 처리 전부를 엔진 안에서 한다)
    const img = frame.getContext('2d').getImageData(0, 0, W, H).data;
    const rgb = new Uint8Array(W * H * 3);
    for (let i = 0, j = 0; i < img.length; i += 4, j += 3) {
      rgb[j] = img[i];
      rgb[j + 1] = img[i + 1];
      rgb[j + 2] = img[i + 2];
    }
    const detB = new Uint8Array(
      await (await fetch(BASE + 'face_detector.sw')).arrayBuffer()
    );
    const lmB = new Uint8Array(
      await (await fetch(BASE + 'face_landmarks.sw')).arrayBuffer()
    );

    const lmDiff = (ours) => {
      let mx = 0, sum = 0;
      for (let i = 0; i < mpLm.length; i++) {
        const d = Math.hypot(ours[i][0] * W - mpLm[i][0], ours[i][1] * H - mpLm[i][1]);
        mx = Math.max(mx, d);
        sum += d;
      }
      return { max: mx, mean: sum / mpLm.length };
    };

    for (const [tag, color, run] of [
      ['cpu', '#d1242f', async () => {
        const det = aiMod.load_model_cpu_h(detB).handle;
        const lm = aiMod.load_model_cpu_h(lmB).handle;
        const task = aiMod.face_task_new(false);
        const r = aiMod.face_task_cpu(task, det, lm, rgb, W, H, 0.0);
        aiMod.face_task_free(task);
        aiMod.unload_model_cpu_h(det);
        aiMod.unload_model_cpu_h(lm);
        return r;
      }],
      ['gpu', '#58a6ff', async () => {
        await aiMod.init_engine();
        const det = (await aiMod.load_model_h(detB)).handle;
        const lm = (await aiMod.load_model_h(lmB)).handle;
        const task = aiMod.face_task_new(false);
        const r = await aiMod.face_task_gpu(task, det, lm, rgb, W, H, 0.0);
        aiMod.face_task_free(task);
        aiMod.unload_model_h(det);
        aiMod.unload_model_h(lm);
        return r;
      }],
    ]) {
      say(`FaceTask ${tag} 실행 중…`);
      try {
        const r = await run();
        if (!r) throw new Error('얼굴 못 찾음 (null)');
        const d = lmDiff(r.points);
        const ok = r.points.length === mpLm.length && d.max <= TOL_PX;
        pass = pass && ok;
        verdicts.push(`lm-${tag}=${ok ? 'ok' : 'FAIL'}`);
        log(
          `face-ab lm-${tag} n=${r.points.length} presence=${r.presence.toFixed(3)} ` +
            `dpx max=${d.max.toFixed(2)} mean=${d.mean.toFixed(2)}`
        );
        vc.fillStyle = color;
        for (const [x, y] of r.points) vc.fillRect(x * W - 0.4, y * H - 0.4, 0.8, 0.8);
      } catch (e) {
        pass = false;
        verdicts.push(`lm-${tag}=ERR`);
        log(`face-ab lm-${tag} ERR ${String(e).slice(0, 150)}`);
      }
    }
  } catch (e) {
    pass = false;
    verdicts.push('lm=ERR');
    log(`face-ab lm ERR ${String(e).slice(0, 150)}`);
  }

  // vision 워커 최소 상주(det+lm+게이즈) 스모크 — 핸들 API로 3모델을 GPU에
  // 동시에 올려 번갈아 추론한다 (게이즈 체인: face_detector→face_landmarks→gaze)
  say('3모델 동시 상주 스모크…');
  try {
    const handles = [];
    const times = [];
    for (const sw of ['face_detector.sw', 'face_landmarks.sw', 'gaze.sw']) {
      const b = new Uint8Array(await (await fetch(BASE + sw)).arrayBuffer());
      handles.push((await aiMod.load_model_h(b)).handle);
    }
    for (const h of handles) {
      const io = aiMod.model_io_h(h);
      const input = new Float32Array(io.h * io.w * io.c).fill(0.1);
      await aiMod.infer_frame_h(h, input, io.outputs[0]); // 워밍업
      const t0 = performance.now();
      await aiMod.infer_frame_h(h, input, io.outputs[0]);
      times.push(performance.now() - t0);
      const st = aiMod.model_stats_h(h);
      if (st.frames < 2) throw new Error(`핸들 ${h} 통계 미기록`);
    }
    for (const h of handles) aiMod.unload_model_h(h);
    log(
      `face-ab residency 3models ok det=${times[0].toFixed(2)}ms ` +
        `lm=${times[1].toFixed(2)}ms gaze=${times[2].toFixed(2)}ms`
    );
  } catch (e) {
    pass = false;
    log(`face-ab residency ERR ${String(e).slice(0, 150)}`);
  }

  say(pass ? `PASS (tol ${TOL_PX}px)` : 'FAIL — 표의 diff 확인');
  log(`face-ab verdict ${pass ? 'PASS' : 'FAIL'} tol=${TOL_PX}px ${verdicts.join(' ')}`);
  log('face-ab-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`face-ab fatal ${String(e).slice(0, 200)}`);
  log('face-ab-done');
});
