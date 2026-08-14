// studio — VideoPipeline 데모/게이트. 헤드리스:
//   node tools/run_web.mjs demo/studio.html --camera
// 완료 신호: AI_ENGINE_RESULT: studio ... / studio-done

const say = (t) => (document.getElementById('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);
const $ = (id) => document.getElementById(id);

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

function currentPatch() {
  const bgSel = $('bg').value;
  return JSON.stringify({
    background:
      bgSel === 'none' ? null : bgSel === 'color' ? '#00a05a' : 'image',
    blur: Number($('blur').value) / 100,
    brightness: Number($('bright').value) / 100,
    grayscale: Number($('gray').value) / 100,
    studioLight: $('light').checked
      ? {
          enabled: true,
          ambient: 0.85,
          lights: [
            { enabled: true, x: 0.3, y: 0.25, color: '#ffd9a0', intensity: 0.9, radius: 0.6, target: 'person' },
            { enabled: true, x: 0.85, y: 0.7, color: '#9fc4ff', intensity: 0.5, radius: 0.5, target: 'background' },
          ],
        }
      : null,
    // 인물 중앙 프레이밍 (웹 기본값 zoomMax 1.7 / headroom 0.15)
    framing: $('framing').checked ? { enabled: true, zoomMax: 1.7, headroom: 0.15 } : null,
    // 터치업/메이크업 — 랜드마크 필요 (FaceTask는 tick이 studio_face로 돌린다)
    touchUp: $('touchup').checked ? { enabled: true, strength: 0.7 } : null,
    makeup: $('makeup').checked
      ? {
          enabled: true,
          intensity: 1.0,
          lip: { color: '#d98f95', alpha: 0.45 },
          blush: { color: '#edaab2', alpha: 0.18, size: 0.23 },
          shadow: { color: '#b98d84', alpha: 0.16 },
        }
      : null,
    // 프레임 변환은 tick()의 2D preprocess가 하고, 엔진은 이미지 배경 좌표 보정만
    mirror: $('mirror').checked,
    degree: Number($('degree').value),
  });
}

// 배경↔프레임 종횡비 반전(cropFactor≥1.6) blur-fill 사전합성 — v-ai
// background-fit.js 이식 (배치 결정: 웹은 **호스트** 몫 — 캔버스 2D 1회 합성이라
// 엔진 이관 실익이 없고 seam 함정(정수 좌표·오버스캔·2패스)이 이미 검증됨.
// 모바일은 ai-ffi 세로 대응 때 Rust 포팅). 결과 비율=프레임 비율이라 엔진 cover
// 수학은 1:1 통과. 온화한 크롭(<1.6)은 null — 원본 cover 유지.
function fitBackgroundForCanvas(image, canvasWidth, canvasHeight) {
  const iw = image.width, ih = image.height;
  if (!iw || !ih || !canvasWidth || !canvasHeight) return null;
  const ir = iw / ih, cr = canvasWidth / canvasHeight;
  const cropFactor = ir > cr ? ir / cr : cr / ir;
  if (cropFactor < 1.6) return null;
  const width = Math.max(2, Math.round(canvasWidth));
  const height = Math.max(2, Math.round(canvasHeight));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  // 본층 contain — 전부 정수 좌표 (소수점 경계는 1px 반투명 seam 줄이 남는다)
  const drawScale = Math.min(width / iw, height / ih);
  const containW = Math.round(iw * drawScale);
  const containH = Math.round(ih * drawScale);
  const x0 = Math.round((width - containW) / 2);
  const y0 = Math.round((height - containH) / 2);
  const BAND_ZOOM = 1.6, PAD = 8;
  const bandOps = [];
  if (y0 > 0.5) {
    const bandH = y0;
    const stripSrc = Math.min(ih, (bandH + PAD) / (drawScale * BAND_ZOOM));
    bandOps.push(() => {
      ctx.translate(0, 2 * y0);
      ctx.scale(1, -1);
      ctx.drawImage(image, 0, 0, iw, stripSrc, x0 - PAD, y0, containW + 2 * PAD, bandH + PAD);
    });
    const y1 = y0 + containH;
    bandOps.push(() => {
      ctx.translate(0, 2 * y1);
      ctx.scale(1, -1);
      ctx.drawImage(
        image, 0, ih - stripSrc, iw, stripSrc,
        x0 - PAD, y1 - bandH - PAD, containW + 2 * PAD, bandH + PAD
      );
    });
  }
  if (x0 > 0.5) {
    const bandW = x0;
    const stripSrc = Math.min(iw, (bandW + PAD) / (drawScale * BAND_ZOOM));
    bandOps.push(() => {
      ctx.translate(2 * x0, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(image, 0, 0, stripSrc, ih, x0, y0 - PAD, bandW + PAD, containH + 2 * PAD);
    });
    const x1 = x0 + containW;
    bandOps.push(() => {
      ctx.translate(2 * x1, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(
        image, iw - stripSrc, 0, stripSrc, ih,
        x1 - bandW - PAD, y0 - PAD, bandW + PAD, containH + 2 * PAD
      );
    });
  }
  // 2패스: 선명 베이스코트 → 블러 덧칠 (반투명 감쇠가 같은 내용 위에 얹혀 seam 없음)
  for (const filter of ['none', 'blur(3px)']) {
    ctx.filter = filter;
    for (const op of bandOps) {
      ctx.save();
      op();
      ctx.restore();
    }
  }
  ctx.filter = 'none';
  ctx.drawImage(image, x0, y0, containW, containH);
  return canvas;
}

async function uploadBgBitmap(bmp) {
  // 종횡비 반전이면 blur-fill 사전합성 (프레임 크기 기준 — src는 카메라 후 확정)
  const srcC = $('src');
  const fitted = fitBackgroundForCanvas(bmp, srcC.width, srcC.height);
  const source = fitted || bmp;
  const c = document.createElement('canvas');
  c.width = source.width;
  c.height = source.height;
  const cx = c.getContext('2d');
  cx.drawImage(source, 0, 0);
  const d = cx.getImageData(0, 0, c.width, c.height);
  aiMod.studio_bg_image(new Uint8Array(d.data.buffer), c.width, c.height);
  if (fitted) log(`studio bg blur-fill 사전합성 (${bmp.width}×${bmp.height} → ${c.width}×${c.height})`);
}

function wireControls() {
  const push = () => {
    for (const [id, out] of [['blur', 'blurv'], ['bright', 'brightv'], ['gray', 'grayv']]) {
      $(out).textContent = $(id).value;
    }
    try { aiMod.studio_config(currentPatch()); } catch (e) { console.warn(e); }
  };
  $('bg').addEventListener('input', async () => {
    const v = $('bg').value;
    if (v.startsWith('asset:')) {
      // 실제 제품 배경 (v-room filters — make studio-assets 로 복사)
      const blob = await (await fetch(`assets/bg/${v.slice(6)}.jpg`)).blob();
      await uploadBgBitmap(await createImageBitmap(blob));
      $('bg').dataset.image = '1';
    }
    push();
  });
  for (const id of ['light', 'framing', 'mirror', 'degree', 'blur', 'bright', 'gray']) {
    $(id).addEventListener('input', push);
  }
  // B티어 수동 선택 — 이때 비로소 R11 로드. 수동 A 선택 = 강등 리셋(탈출구 —
  // auto의 "승격 없음" 규약은 유지하되 사용자가 명시적으로 되돌릴 수 있어야 한다)
  $('tier').addEventListener('input', () => {
    if ($('tier').value === 'b') ensureCpuSeg().catch(console.warn);
    if ($('tier').value === 'a') resetDemotion();
  });
  // 터치업/메이크업 — 켜면 FaceTask(랜드마크 소비자)도 준비
  for (const id of ['touchup', 'makeup']) {
    $(id).addEventListener('input', () => {
      if ($(id).checked) ensureFaceTask().catch(console.warn);
      push();
    });
  }
  $('item').addEventListener('input', () => setItem($('item').value).catch(console.warn));
  $('focus').addEventListener('input', () => {
    if ($('focus').checked) {
      // MULTIPLE_FACES 감시: 디텍터가 매 프레임 얼굴 수를 센다 (num_faces=2)
      ensureGaze()
        .then(() => face && aiMod.face_task_num_faces(face.task, 2))
        .catch(console.warn);
    } else {
      $('focushud').textContent = '';
      lastFocus = null;
      if (face) aiMod.face_task_num_faces(face.task, 1); // 디텍터 비용 제거
      if (gazeF) aiMod.gaze_task_reset(gazeF.task); // 스트림 파기 규약 — 재켬은 백지에서
    }
  });
  $('bgfile').addEventListener('change', async () => {
    const f = $('bgfile').files[0];
    if (!f) return;
    await uploadBgBitmap(await createImageBitmap(f));
    $('bg').value = 'image';
    push();
  });
  push();
}

// ── 3D 아이템 오버레이 — 엔진 wgpu items3d (three.js 대체, 웹·모바일 통일) ──
// Horn 피팅·PBR·씬 광원 매칭 전부 엔진 안 — JS는 GLB bytes·랜드마크만 넘긴다.
let face = null;  // { task, det, lm }
const itemsLoaded = new Set(); // GLB 주입 완료 종류

// ── B티어: CPU 추론(R11) + GPU 합성 ──
// 폴백 사다리 §1.5: 추론이 CPU로 떨어져도 합성 GPU가 살아있으면 효과 전부 산다.
// CPU→GPU 트래픽 = 로짓 업로드(288×160×2 f32 ≈ 360KB → 엔진이 Rg32Float 업로드)뿐.
let cpuSeg = null; // { io, canvas, ctx2d, rgb }
let cpuSegLoading = null;
let resetDemotion = () => {}; // main()이 실배선 (강등 상태가 main 스코프라)
// **지연 로드** — B티어가 실제로 필요해질 때(셀렉트 b / auto 강등)만 R11을
// 가져온다. 시작 시 무조건 로드하던 것은 자원 낭비 + "R11로 측정하나?" 오해의
// 원인이었다 (기본/perf 측정은 전부 RVM GPU — R11은 폴백 전용).
function ensureCpuSeg() {
  if (cpuSeg) return Promise.resolve();
  if (!cpuSegLoading) cpuSegLoading = loadCpuSeg();
  return cpuSegLoading;
}
async function loadCpuSeg() {
  try {
    const resp = await fetch('../models/segm_r11_160x288.sw');
    if (!resp.ok) throw new Error(`R11 없음(${resp.status}) — make convert-r11-web`);
    aiMod.load_model_cpu(new Uint8Array(await resp.arrayBuffer()));
    const io = aiMod.model_io_cpu();
    const c = document.createElement('canvas');
    c.width = io.w;
    c.height = io.h;
    cpuSeg = {
      io,
      canvas: c,
      ctx2d: c.getContext('2d', { willReadFrequently: true }),
      rgb: new Float32Array(io.w * io.h * 3),
    };
    log(`studio cpu-tier ready (R11 ${io.w}x${io.h})`);
  } catch (e) {
    log('studio cpu-tier unavailable: ' + String(e).slice(0, 120));
  }
}

let faceLoading = null;
async function ensureFaceTask() {
  // in-flight 가드 — item·focus가 같은 프레임에 동시에 부르면 이중 로드로
  // face가 덮여 num_faces 설정이 유령 핸들에 붙는다 (audit이 잡은 레이스)
  if (face) return;
  if (!faceLoading) {
    faceLoading = (async () => {
      const load = async (u) =>
        (await aiMod.load_model_h(new Uint8Array(await (await fetch(u)).arrayBuffer()))).handle;
      const det = await load('../models/mediapipe/face_detector.sw');
      const lm = await load('../models/mediapipe/face_landmarks.sw');
      face = { task: aiMod.face_task_new(false), det, lm };
      // 로드 완료 시점의 스위치 상태 반영 — 핸들러/로드 순서 경합 무해화
      if ($('focus').checked) aiMod.face_task_num_faces(face.task, 2);
    })();
  }
  return faceLoading;
}

// ── 집중도 (GazeTask) — FaceTask 478pt를 소비, CNN(gaze.sw)은 엔진 내부 페이싱 ──
let gazeF = null;     // { task, handle }
let lastFocus = null; // 마지막 FocusResult (HUD·헤드리스 로그)
let gazeLoading = null;
async function ensureGaze() {
  if (gazeF) return;
  if (gazeLoading) return gazeLoading;
  gazeLoading = ensureGazeInner();
  return gazeLoading;
}
async function ensureGazeInner() {
  await ensureFaceTask();
  const b = new Uint8Array(
    await (await fetch('../models/mediapipe/gaze.sw')).arrayBuffer()
  );
  // face_blendshapes — blink의 bs 절반 (없으면 EAR만으로 동작, bs=0)
  let bs = 0;
  try {
    const bb = new Uint8Array(
      await (await fetch('../models/mediapipe/face_blendshapes.sw')).arrayBuffer()
    );
    bs = (await aiMod.load_model_h(bb)).handle;
  } catch (e) {
    console.warn('face_blendshapes 로드 실패 — EAR 절반만', e);
  }
  gazeF = { task: aiMod.gaze_task_new(), handle: (await aiMod.load_model_h(b)).handle, bs };
  log(`studio gaze ready (bs=${bs ? 'on' : 'off'})`);
}

async function setItem(kind) {
  if (kind === 'none') {
    aiMod.studio_items('none', 'none', 'none');
    return;
  }
  await ensureFaceTask();
  if (!itemsLoaded.has(kind)) {
    const b = new Uint8Array(await (await fetch(`assets/glb/${kind}.glb`)).arrayBuffer());
    aiMod.studio_item_glb(kind, b);
    itemsLoaded.add(kind);
    log(`studio item ${kind} glb ok (${(b.length / 1024) | 0}KB)`);
  }
  // 종류 분류는 이름 접두사로 (엔진 KINDS 목록과 일치)
  aiMod.studio_items(
    kind.startsWith('hat') ? kind : 'none',
    kind.startsWith('glasses') ? kind : 'none',
    kind.startsWith('mustache') ? kind : 'none'
  );
}

// ?perf=1: 스테이지별 **페이싱 측정** — 프레임 제출→gpu_sync 완료 대기→다음.
// HUD의 5초 샘플은 큐 대기를 포함해 밀리면 부푼다(강등 판정용) — 프레임당
// 실비용은 이 페이싱 수치가 근거다 (측정 규율).
const QS = new URLSearchParams(location.search);
const PERF = QS.has('perf');
// ?audit=1: **자원 절약 감사** — 기능 off 시 엔진 추론 카운터(model_stats_h.frames)가
// 실제로 동결되는지 실루프로 검증 (on→off 델타 0 단언)
const AUDIT = QS.has('audit');
// 스크립트 스모크(자동 토글 시나리오)는 헤드리스에서만 — 인터랙티브 세션에서
// 스위치가 저절로 바뀌는 것을 방지 (playwright/자동화 = navigator.webdriver)
const SCRIPTED = navigator.webdriver === true;
async function perfProtocol(seg, src, sctx, video) {
  const lines = [];
  const stages = [
    ['기본', { bg: 'none', blur: 0, light: false, framing: false }],
    ['blur60', { bg: 'none', blur: 60, light: false, framing: false }],
    ['이미지배경', { bg: 'asset:1', blur: 0, light: false, framing: false }],
    ['이미지배경+blur60', { bg: 'asset:1', blur: 60, light: false, framing: false }],
    ['이미지배경+blur60+조명+프레이밍', { bg: 'asset:1', blur: 60, light: true, framing: true }],
  ];
  log(`studio-perf camera=${video.videoWidth}x${video.videoHeight}`);
  for (const [name, cfg] of stages) {
    $('bg').value = cfg.bg;
    if (cfg.bg.startsWith('asset:')) {
      const blob = await (await fetch(`assets/bg/${cfg.bg.slice(6)}.jpg`)).blob();
      await uploadBgBitmap(await createImageBitmap(blob));
    }
    $('blur').value = cfg.blur;
    $('light').checked = cfg.light;
    $('framing').checked = cfg.framing;
    aiMod.studio_config(currentPatch());
    const samples = [];
    for (let i = 0; i < 40; i++) {
      sctx.drawImage(video, 0, 0, src.width, src.height);
      const t0 = performance.now();
      await aiMod.studio_frame(seg, src);
      await aiMod.gpu_sync(); // 페이싱 — 큐를 비우고 다음 프레임
      samples.push(performance.now() - t0);
    }
    const st = [...samples.slice(10)].sort((a, b) => a - b); // 워밍업 10 제외
    const p50v = st[st.length >> 1];
    const p90v = st[Math.floor(st.length * 0.9)];
    const line = `${name}: p50=${p50v.toFixed(1)}ms p90=${p90v.toFixed(1)}ms`;
    log(`studio-perf ${line} (페이싱, ${st.length}샘플)`);
    lines.push(line);
    say(`페이싱 측정 중… ${lines.length}/5`);
  }
  // 결과를 화면에 — 측정 후 루프가 서는 건 설계지만(페이싱 프로토콜 종료),
  // 결과가 콘솔에만 있으면 "멈췄다"로만 보인다 (실사용 피드백)
  say('페이싱 측정 완료 (RVM GPU) — ' + lines.join(' · '));
  log('studio-perf verdict PASS');
  log('studio-done');
}

async function main() {
  say('카메라 여는 중…');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
  });
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  await video.play();

  say('엔진 초기화…');
  await loadAiWasm();
  if (!navigator.gpu) throw new Error('WebGPU 미지원');
  await aiMod.init_engine();
  const bytes = new Uint8Array(
    await (await fetch('../models/rvm_256x144.sw')).arrayBuffer()
  );
  const seg = (await aiMod.load_model_h(bytes)).handle;
  aiMod.studio_attach($('out'));
  wireControls();
  // R11(B티어)은 지연 로드 — 셀렉트 b / auto 강등 때 ensureCpuSeg()가 가져온다
  log('studio init ok');

  const src = $('src');
  // ⚠ src = 엔진 입력 해상도. 카메라 실해상도로 동기화 — 안 하면 카메라를
  // src 크기로 줄였다가 출력 서피스로 다시 업스케일해 화질이 뭉개진다 (실제로 당함)
  src.width = video.videoWidth || 1280;
  src.height = video.videoHeight || 720;
  const sctx = src.getContext('2d');
  if (PERF) {
    say('페이싱 측정 중…');
    await perfProtocol(seg, src, sctx, video);
    return;
  }
  // ── HUD 정직화 ──
  // times(제출 벽시계)는 허수다 — WebGPU 제출은 즉시 리턴한다 (측정 규율).
  // 정직한 수치 2개: ①rAF 간격(체감 스루풋, 항상) ②GPU 한 프레임 실비용
  // (1초마다 **페이싱 샘플** — 큐 배수 후 한 프레임만 제출·완료 대기, ?perf=1과
  // 같은 정의). 큐 소진식 비동기 샘플은 큐에 쌓인 프레임까지 포함해 실비용의
  // 3~6배 허수가 됐고 강등 오발의 원인이라 폐기했다.
  const times = [];   // 제출 벽시계 (참고용 유지)
  const rafIv = [];   // 프레임 간격 실측
  let lastTick = 0;
  let lastVision = 0; // 집중도 비전 틱 페이싱 (10fps)
  let gpuMs = null;
  let lastGpuSample = 0;
  let frames = 0;
  // 티어: auto는 gpu 실측 66ms 초과 2연속 샘플이면 B로 강등 (승격 없음 — v-ai 규약)
  let demoted = false;
  const gpuWin = [];
  let winCount = 0;
  let badWindows = 0;
  // 수동 A 선택 시 강등 상태 해제 (wireControls의 tier 핸들러가 부른다)
  resetDemotion = () => {
    demoted = false;
    badWindows = 0;
    gpuWin.length = 0;
  };
  let curTier = 'A';
  let cpuMs = null;
  // ── ?audit=1 자원 절약 감사 — 근거는 엔진 프레임 카운터 (JS 추정이 아니라
  // 세션이 실제로 추론한 횟수). 스케줄: ~30f 전부 off(로드 자체가 없어야) →
  // on(집중도+아이템+터치업+메이크업+프레이밍) → 100f에서 스냅샷+전부 off →
  // 110f 정착 스냅샷 → 170f 델타 0 단언 (seg만 상시 증가해야 정상).
  const mstat = (h) => (h ? aiMod.model_stats_h(h).frames : -1);
  const auditSnapshot = () => ({
    det: face ? mstat(face.det) : -1,
    lm: face ? mstat(face.lm) : -1,
    gaze: gazeF ? mstat(gazeF.handle) : -1,
    bs: gazeF && gazeF.bs ? mstat(gazeF.bs) : -1,
    seg: mstat(seg),
  });
  let auditBase = null;
  const auditFail = [];
  const toggle = (id, on) => {
    if ($(id).type === 'checkbox') $(id).checked = on;
    $(id).dispatchEvent(new Event('input'));
  };
  const auditMilestones = () => {
    if (frames === 30) {
      // OFF 상태: 부가 모델은 **로드조차** 안 돼 있어야 한다 (R11 포함)
      const ok = !face && !gazeF && !cpuSeg;
      if (!ok) auditFail.push('idle-load');
      log(`studio-audit idle30f: face=${!!face} gaze=${!!gazeF} r11=${!!cpuSeg} ${ok ? '미로드 ok' : 'FAIL'}`);
      $('item').value = 'hat1';
      $('item').dispatchEvent(new Event('input'));
      toggle('focus', true);
      toggle('touchup', true);
      toggle('makeup', true);
      toggle('framing', true);
    }
    if (frames === 100) {
      const s = auditSnapshot();
      // 무얼굴 가짜 카메라: det(재획득 시도)는 매 프레임 돌아야 한다
      const ok = s.det > 0;
      if (!ok) auditFail.push('on-det');
      log(
        `studio-audit on70f: det=${s.det} lm=${s.lm} gaze=${s.gaze} bs=${s.bs} seg=${s.seg} ` +
          `gpu=${gpuMs ? gpuMs.toFixed(1) + 'ms' : '?'} tier=${curTier}${demoted ? '(강등)' : ''} ${ok ? 'ok' : 'FAIL'}`
      );
      $('item').value = 'none';
      $('item').dispatchEvent(new Event('input'));
      toggle('focus', false);
      toggle('touchup', false);
      toggle('makeup', false);
      toggle('framing', false);
    }
    if (frames === 110) auditBase = auditSnapshot(); // off 후 비행 중 비동기 정착 10f
    if (frames === 170) {
      const s = auditSnapshot();
      const frozen =
        s.det === auditBase.det && s.lm === auditBase.lm &&
        s.gaze === auditBase.gaze && s.bs === auditBase.bs;
      if (!frozen) auditFail.push('off-frozen');
      if (cpuSeg) auditFail.push('r11-eager');
      // seg 카운터는 이 경로에서 무의미: model_stats.frames는 finish_frame(동기화)
      // 때만 기록되는데 studio 세그는 제출만 한다(성능 설계) — 가동 증거는 렌더
      // 출력 자체 (별도 스모크가 검증). 판정에서 제외, 참고로만 남긴다.
      log(
        `studio-audit off60f: det ${auditBase.det}→${s.det} lm ${auditBase.lm}→${s.lm} ` +
          `gaze ${auditBase.gaze}→${s.gaze} bs ${auditBase.bs}→${s.bs} ${frozen ? '동결 ok' : 'FAIL'} | ` +
          `r11 ${cpuSeg ? 'FAIL(로드됨)' : '미로드 ok'} | seg 카운터는 비동기 경로라 항상 0 (참고)`
      );
      log(`studio-audit verdict ${auditFail.length ? 'FAIL ' + auditFail.join(',') : 'PASS'}`);
      log('studio-done');
    }
  };
  const p50 = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
  const hudStats = () =>
    `tier=${curTier}${demoted ? '(강등)' : ''}` +
    ` 간격 p50 ${p50(rafIv.slice(-60)).toFixed(1)}ms` +
    `(~${Math.round(1000 / Math.max(1, p50(rafIv.slice(-60))))}fps)` +
    ` gpu=${gpuMs ? gpuMs.toFixed(2) + 'ms' : '측정중'}` +
    (curTier === 'B' && cpuMs !== null ? ` cpu_infer=${cpuMs.toFixed(1)}ms` : '') +
    ` submit_p50=${p50(times.slice(-25)).toFixed(2)}ms`;
  say('가동 중');
  const tick = async () => {
    const now0 = performance.now();
    if (lastTick) {
      rafIv.push(now0 - lastTick);
      if (rafIv.length > 120) rafIv.shift();
    }
    lastTick = now0;
    // mirror/degree는 추론 전 프레임에 적용 (v-ai _prepareSourceElement 등가 —
    // 좌표계 계약: 랜드마크·마스크가 화면 좌표와 일치해야 한다)
    const deg = Number($('degree').value);
    const mir = $('mirror').checked;
    if (mir || deg) {
      sctx.setTransform(1, 0, 0, 1, 0, 0);
      sctx.clearRect(0, 0, src.width, src.height);
      sctx.translate(src.width / 2, src.height / 2);
      if (mir) sctx.scale(-1, 1);
      if (deg) sctx.rotate((deg * Math.PI) / 180);
      sctx.drawImage(video, -src.width / 2, -src.height / 2, src.width, src.height);
      sctx.setTransform(1, 0, 0, 1, 0, 0);
    } else {
      sctx.drawImage(video, 0, 0, src.width, src.height);
    }
    const t0 = performance.now();
    // ── GPU 실측 = **페이싱 샘플** (1s마다): 큐를 먼저 비우고 → 이번 프레임 제출
    // → 이 프레임만 완료 대기. ?perf=1과 같은 정의의 "한 프레임 실비용"이다.
    // ⚠ 이전 방식(제출→큐 소진 비동기 샘플)은 큐에 쌓인 프레임까지 포함해
    // 120Hz rAF에서 실비용의 3~6배로 부풀었고, 그 허수가 66ms 문턱을 넘어
    // **배경+블러+메이크업에서 멀쩡한 GPU를 B로 강등**시켰다 (사용자 발견).
    // 실비용 15ms는 66ms를 절대 못 넘는다 — 판정 입력이 허수였던 것.
    // 샘플 프레임은 배수+동기화로 ~2프레임 멈칫하지만 1초에 한 번이라 비가시.
    const wantSample =
      frames === 15 || frames === 45 || frames === 75 || now0 - lastGpuSample > 1000;
    // 유효 티어 결정 (전환 스위치는 호스트 몫 — 폴백 사다리 규약)
    const sel = $('tier').value;
    const useCpu = !!cpuSeg && (sel === 'b' || (sel === 'auto' && demoted));
    curTier = useCpu ? 'B' : 'A';
    $('tierv').textContent = curTier + (demoted ? ' (강등됨)' : '');
    try {
      const submitFrame = async () => {
        if (useCpu) {
          // B티어: CPU 추론(R11) → 로짓을 엔진에 주입 → GPU 합성 (효과 전부 생존)
          const { io, ctx2d, rgb } = cpuSeg;
          ctx2d.drawImage(src, 0, 0, io.w, io.h);
          const d = ctx2d.getImageData(0, 0, io.w, io.h).data;
          for (let p = 0, q = 0; p < d.length; p += 4, q += 3) {
            rgb[q] = d[p] / 255;
            rgb[q + 1] = d[p + 1] / 255;
            rgb[q + 2] = d[p + 2] / 255;
          }
          const tc = performance.now();
          // view 대신 복사 반환 — 반환 뷰를 곧장 wasm 호출에 되넘기면 힙 성장 시
          // detach 위험 (infer_frame_cpu_view 규약: 뷰는 다음 호출 전 소비)
          const logits = aiMod.infer_frame_cpu(rgb, io.outputs[0]);
          cpuMs = performance.now() - tc;
          aiMod.studio_frame_mask(seg, src, logits, 2, io.w, io.h);
        } else {
          await aiMod.studio_frame(seg, src);
        }
      };
      if (wantSample) {
        lastGpuSample = now0;
        await aiMod.gpu_sync(); // 밀린 큐 배수 — 다음 측정이 한 프레임만 재도록
        const ts = performance.now();
        await submitFrame();
        await aiMod.gpu_sync();
        gpuMs = performance.now() - ts;
        // auto 강등: 실비용 창 p90 (승격 없음 — 수동 A 선택만이 리셋)
        gpuWin.push(gpuMs);
        if (gpuWin.length >= 10) {
          const s = gpuWin.slice().sort((a, b) => a - b);
          const p90 = s[Math.floor(s.length * 0.9)];
          gpuWin.length = 0;
          winCount++;
          if (winCount > 1) {
            // 웜업 창 폐기 (v-ai 규약)
            if (p90 > 66) {
              badWindows++;
              if (badWindows >= 2 && !demoted) {
                demoted = true;
                log(`studio demote → B (실비용 창 p90 ${p90.toFixed(1)}ms × ${badWindows})`);
                ensureCpuSeg().catch(console.warn); // 강등 확정 — 이제야 R11 로드
              }
            } else {
              badWindows = 0;
            }
          }
        }
      } else {
        await submitFrame();
      }
      // 3D 아이템 + 집중도 — FaceTask 랜드마크 공유 (얼굴 lm 1회 추론 공유가
      // v-ai 이중 로드 낭비의 수리 지점). 아이템은 매 프레임(오버레이 부드러움),
      // 집중도만 켜져 있으면 비전 틱 10fps(웹 visionFps)로 페이싱.
      // FaceTask는 studio_face — 파이프라인 프레임 텍스처 공유(GPU 전처리 직결,
      // 픽셀의 CPU 왕복 0). getImageData는 아직 u8 픽셀이 필요한 소비자(게이즈
      // CNN 크롭 — 비전 틱 / 광원 프로브 — 8틱 스로틀)에서만 뽑는다.
      const wantItem = $('item').value !== 'none' && face;
      const wantFocus = $('focus').checked && gazeF;
      // 터치업/메이크업도 랜드마크 소비자 — 켜져 있으면 FaceTask가 돌아야 한다
      const wantFx = ($('touchup').checked || $('makeup').checked) && face;
      const visionDue = wantFocus && now0 - lastVision >= 100;
      let faceR = null;
      if (wantItem || visionDue || wantFx) {
        faceR = await aiMod.studio_face(
          face.task, face.det, face.lm, performance.now()
        );
        const probeDue = wantItem && faceR && frames % 8 === 0;
        let rgb = null;
        if (visionDue || probeDue) {
          const d = sctx.getImageData(0, 0, src.width, src.height).data;
          rgb = new Uint8Array(src.width * src.height * 3);
          for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
            rgb[j] = d[i];
            rgb[j + 1] = d[i + 1];
            rgb[j + 2] = d[i + 2];
          }
        }
        if (wantItem) {
          if (faceR) {
            const flat = new Float32Array(faceR.points.length * 3);
            for (let i = 0; i < faceR.points.length; i++) {
              flat[i * 3] = faceR.points[i][0];
              flat[i * 3 + 1] = faceR.points[i][1];
              flat[i * 3 + 2] = faceR.points[i][2];
            }
            aiMod.studio_items_pose(flat);
            if (probeDue) aiMod.studio_items_probe(rgb, src.width, src.height); // 씬 광원
          } else {
            aiMod.studio_items_pose(new Float32Array(0)); // 소실 — 스무딩 리셋
          }
        }
        if (visionDue) {
          lastVision = now0;
          // FaceTask 랜드마크 배선: 정규화 478pt → flat [x,y]×N (없으면 빈 배열 = 소실 틱)
          let flat = new Float32Array(0);
          if (faceR) {
            flat = new Float32Array(faceR.points.length * 2);
            for (let i = 0; i < faceR.points.length; i++) {
              flat[i * 2] = faceR.points[i][0];
              flat[i * 2 + 1] = faceR.points[i][1];
            }
          }
          lastFocus = await aiMod.gaze_task_gpu(
            gazeF.task, gazeF.handle, gazeF.bs, rgb, src.width, src.height,
            flat, faceR ? faceR.faceCount || 1 : 0, performance.now()
          );
          $('focushud').textContent =
            `집중도: ${lastFocus.status} score=${lastFocus.score}` +
            (lastFocus.yaw != null
              ? ` yaw=${lastFocus.yaw.toFixed(1)}° pitch=${lastFocus.pitch.toFixed(1)}°`
              : '');
        }
      }
    } catch (e) {
      log(`studio frame ERR ${String(e).slice(0, 150)}`);
      log('studio-done');
      return;
    }
    times.push(performance.now() - t0);
    frames++;
    if (AUDIT) auditMilestones();
    if (SCRIPTED && !AUDIT && frames === 30) {
      // 워밍업 후 30프레임 시점에 통계 보고 (헤드리스 게이트)
      log(`studio frames=30 ${hudStats()} blur_test start`);
      $('blur').value = 60;
      $('blur').dispatchEvent(new Event('input'));
    }
    if (SCRIPTED && !AUDIT && frames === 60) {
      log(`studio blur ${hudStats()}`);
      // 실제 제품 배경 + 3D 아이템 경로 (가짜 카메라엔 얼굴이 없어 오버레이는
      // null 경로를 탄다 — 크래시 없이 도는지가 스모크 목적)
      $('bg').value = 'asset:1';
      $('bg').dispatchEvent(new Event('input'));
      $('light').checked = true;
      $('light').dispatchEvent(new Event('input'));
      // 프레이밍 — bbox 리덕션+20B 리드백 링 경로 스모크 (가짜 카메라라 크롭
      // 목표는 안 잡힐 수 있음 — 크래시 없이 도는지가 목적)
      $('framing').checked = true;
      $('framing').dispatchEvent(new Event('input'));
      // B티어 강제 — 완료 기준 검증: GPU 추론 없이 배경·블러·조명·프레이밍 생존.
      // R11은 지연 로드라 여기서 요청 — 로드 완료 프레임부터 B로 전환된다
      ensureCpuSeg()
        .then(() => {
          if (cpuSeg) $('tier').value = 'b';
        })
        .catch(console.warn);
      // 3D 아이템은 얼굴이 있어야 의미가 있다 — 가짜 카메라(무얼굴)에선 배경만 검증
      // 집중도 스모크 — 무얼굴이라 소실 틱 경로(INITIALIZING→NO_FACE)를 돈다
      $('focus').checked = true;
      $('focus').dispatchEvent(new Event('input'));
      // 3D 아이템 스모크 — GLB wasm 주입+파싱+오버레이 경로. 무얼굴이라
      // draw는 조용히 스킵된다 (크래시 없이 도는지가 목적, 에셋 없으면 warn만)
      $('item').value = 'hat1';
      $('item').dispatchEvent(new Event('input'));
      // 터치업/메이크업 스모크 — config 파싱+studio_face fx 경로 (무얼굴이라
      // 오버레이는 해제 상태 유지 — 크래시 없이 도는지가 목적)
      $('touchup').checked = true;
      $('touchup').dispatchEvent(new Event('input'));
      $('makeup').checked = true;
      $('makeup').dispatchEvent(new Event('input'));
    }
    if (SCRIPTED && !AUDIT && frames === 90) {
      log(`studio asset-bg+item ${hudStats()} face=${face ? 'loaded' : 'off'}`);
      log(
        `studio focus ${lastFocus ? `${lastFocus.status} score=${lastFocus.score}` : 'no-tick'}` +
          ` gaze=${gazeF ? 'loaded' : 'off'}`
      );
      log('studio verdict PASS');
      log('studio-done');
    }
    $('hud').textContent =
      `frame ${frames} | ${hudStats()} — 간격=체감 스루풋, gpu=한 프레임 실비용(1s 페이싱 샘플, 강등 판정 입력), submit=제출 벽시계(허수 참고)`;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

$('start').addEventListener('click', () => {
  main().catch((e) => {
    say('오류: ' + e);
    log(`studio fatal ${String(e).slice(0, 200)}`);
    log('studio-done');
  });
});
