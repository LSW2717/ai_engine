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
    // 프레임 변환은 tick()의 2D preprocess가 하고, 엔진은 이미지 배경 좌표 보정만
    mirror: $('mirror').checked,
    degree: Number($('degree').value),
  });
}

async function uploadBgBitmap(bmp) {
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  const cx = c.getContext('2d');
  cx.drawImage(bmp, 0, 0);
  const d = cx.getImageData(0, 0, c.width, c.height);
  aiMod.studio_bg_image(new Uint8Array(d.data.buffer), c.width, c.height);
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
  $('item').addEventListener('input', () => setItem($('item').value).catch(console.warn));
  $('bgfile').addEventListener('change', async () => {
    const f = $('bgfile').files[0];
    if (!f) return;
    await uploadBgBitmap(await createImageBitmap(f));
    $('bg').value = 'image';
    push();
  });
  push();
}

// ── 3D 아이템 오버레이 — three.js가 FaceTask(478pt) 랜드마크를 소비 ──
// 정밀 Horn 피팅(P3 Expression Stream)은 나중 — 여기선 유사변환(위치·스케일·롤)만.
let three = null; // { renderer, scene, camera, item, kind }
let face = null;  // { task, det, lm }

const ITEM_FIT = {
  // [앵커 계산, 폭 배율, 세로 오프셋(faceW 배)]
  hat1: ['hat', 1.9, -0.75], hat3: ['hat', 1.8, -0.7], hat_christmas: ['hat', 1.7, -0.8],
  hat_cat_ears: ['hat', 1.6, -0.75],
  glasses1: ['eyes', 1.15, 0], glasses_heart: ['eyes', 1.2, 0],
  mustache1: ['mouth', 0.6, 0],
};

// ── B티어: CPU 추론(R11) + GPU 합성 ──
// 폴백 사다리 §1.5: 추론이 CPU로 떨어져도 합성 GPU가 살아있으면 효과 전부 산다.
// CPU→GPU 트래픽 = 로짓 업로드(288×160×2 f32 ≈ 360KB → 엔진이 Rg32Float 업로드)뿐.
let cpuSeg = null; // { io, canvas, ctx2d, rgb }
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

async function ensureFaceTask() {
  if (face) return;
  const load = async (u) =>
    (await aiMod.load_model_h(new Uint8Array(await (await fetch(u)).arrayBuffer()))).handle;
  const det = await load('../models/mediapipe/face_detector.sw');
  const lm = await load('../models/mediapipe/face_landmarks.sw');
  face = { task: aiMod.face_task_new(false), det, lm };
}

async function setItem(kind) {
  const fx = $('fx');
  if (kind === 'none') {
    if (three) three.item.visible = false;
    fx.getContext && null;
    return;
  }
  await ensureFaceTask();
  const THREE = await import('three');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  if (!three) {
    const renderer = new THREE.WebGLRenderer({ canvas: fx, alpha: true, antialias: true });
    renderer.setSize(fx.width, fx.height, false);
    renderer.setClearColor(0x000000, 0); // 오버레이 — 반드시 투명 클리어
    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(-0.4, 1, 1);
    scene.add(key);
    // 픽셀 좌표계 정사영 (y 아래 방향)
    const camera = new THREE.OrthographicCamera(0, fx.width, 0, -fx.height, -2000, 2000);
    three = { THREE, renderer, scene, camera, item: null, kind: null };
  }
  if (three.kind !== kind) {
    if (three.item) three.scene.remove(three.item);
    const gltf = await new GLTFLoader().loadAsync(`assets/glb/${kind}.glb`);
    const root = gltf.scene;
    // 정규화: bbox 중심을 원점으로 — 프레임마다 위치·스케일·롤만 갱신
    const box = new three.THREE.Box3().setFromObject(root);
    const c = box.getCenter(new three.THREE.Vector3());
    root.position.sub(c);
    const holder = new three.THREE.Group();
    holder.add(root);
    holder.userData.width = box.getSize(new three.THREE.Vector3()).x || 1;
    three.scene.add(holder);
    three.item = holder;
    three.kind = kind;
  }
  three.item.visible = true;
}

function drawItem(pts, W, H) {
  if (!three || !three.item || !three.item.visible) return;
  const px = (i) => [pts[i][0] * W, pts[i][1] * H];
  const [lx, ly] = px(33);
  const [rx, ry] = px(263);
  const [c1x, c1y] = px(234);
  const [c2x, c2y] = px(454);
  const faceW = Math.hypot(c2x - c1x, c2y - c1y);
  const roll = Math.atan2(ry - ly, rx - lx);
  const [mode, widthK, dyK] = ITEM_FIT[three.kind] || ['eyes', 1, 0];
  let ax;
  let ay;
  if (mode === 'hat') {
    [ax, ay] = px(10); // 이마 상단
  } else if (mode === 'eyes') {
    ax = (lx + rx) / 2;
    ay = (ly + ry) / 2;
  } else {
    [ax, ay] = px(164); // 인중
  }
  // 세로 오프셋은 얼굴 '위' 방향(눈선에 수직)으로
  ax += Math.sin(roll) * -dyK * faceW * -1;
  ay += Math.cos(roll) * dyK * faceW;
  const s = (faceW * widthK) / three.item.userData.width;
  three.item.position.set(ax, -ay, 0);
  three.item.scale.setScalar(s);
  three.item.rotation.z = -roll;
  three.renderer.render(three.scene, three.camera);
}

function clearItem() {
  if (three) three.renderer.clear(true, true, true);
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
  await loadCpuSeg(); // B티어 준비 (없으면 강제 B만 비활성 — A는 그대로 산다)
  log('studio init ok');

  const src = $('src');
  // ⚠ src = 엔진 입력 해상도. 카메라 실해상도로 동기화 — 안 하면 카메라를
  // src 크기로 줄였다가 출력 서피스로 다시 업스케일해 화질이 뭉개진다 (실제로 당함)
  src.width = video.videoWidth || 1280;
  src.height = video.videoHeight || 720;
  const sctx = src.getContext('2d');
  // ── HUD 정직화 ──
  // times(제출 벽시계)는 허수다 — WebGPU 제출은 즉시 리턴한다 (측정 규율).
  // 정직한 수치 2개: ①rAF 간격(체감 스루풋, 항상) ②GPU 실측(5초마다 논블로킹
  // onSubmittedWorkDone 샘플 — 루프를 세우지 않는다).
  const times = [];   // 제출 벽시계 (참고용 유지)
  const rafIv = [];   // 프레임 간격 실측
  let lastTick = 0;
  let gpuMs = null;
  let gpuPending = false;
  let lastGpuSample = 0;
  let frames = 0;
  // 티어: auto는 gpu 실측 66ms 초과 2연속 샘플이면 B로 강등 (승격 없음 — v-ai 규약)
  let demoted = false;
  let badSamples = 0;
  let curTier = 'A';
  let cpuMs = null;
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
    // GPU 실측 샘플 — t0(제출 시작)부터 GPU 큐가 빌 때까지. 등록만 하고 안 기다린다
    // 헤드리스(90프레임)에서도 단계별 gpu 수치가 잡히게 15/45/75에 강제 샘플
    const wantSample =
      !gpuPending &&
      (frames === 15 || frames === 45 || frames === 75 || now0 - lastGpuSample > 5000);
    // 유효 티어 결정 (전환 스위치는 호스트 몫 — 폴백 사다리 규약)
    const sel = $('tier').value;
    const useCpu = !!cpuSeg && (sel === 'b' || (sel === 'auto' && demoted));
    curTier = useCpu ? 'B' : 'A';
    $('tierv').textContent = curTier + (demoted ? ' (강등됨)' : '');
    try {
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
      if (wantSample) {
        gpuPending = true;
        lastGpuSample = now0;
        aiMod
          .gpu_sync()
          .then(() => {
            gpuMs = performance.now() - t0;
            gpuPending = false;
            // auto 강등 판정 (66ms 초과 2연속 — 승격 없음)
            if (gpuMs > 66) {
              badSamples++;
              if (badSamples >= 2 && !demoted) {
                demoted = true;
                log(`studio demote → B (gpu ${gpuMs.toFixed(1)}ms × ${badSamples})`);
              }
            } else {
              badSamples = 0;
            }
          })
          .catch(() => {
            gpuPending = false;
          });
      }
      // 3D 아이템 — FaceTask 랜드마크로 오버레이 갱신
      // (FaceTask 입력이 아직 u8 프레임이라 여기만 CPU 픽셀 경유 — P3에서 GPU화)
      if ($('item').value !== 'none' && face) {
        const d = sctx.getImageData(0, 0, src.width, src.height).data;
        const rgb = new Uint8Array(src.width * src.height * 3);
        for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
          rgb[j] = d[i];
          rgb[j + 1] = d[i + 1];
          rgb[j + 2] = d[i + 2];
        }
        const r = await aiMod.face_task_gpu(
          face.task, face.det, face.lm, rgb, src.width, src.height, performance.now()
        );
        if (r) drawItem(r.points, $('fx').width, $('fx').height);
        else clearItem();
      } else if (three) {
        clearItem();
      }
    } catch (e) {
      log(`studio frame ERR ${String(e).slice(0, 150)}`);
      log('studio-done');
      return;
    }
    times.push(performance.now() - t0);
    frames++;
    if (frames === 30) {
      // 워밍업 후 30프레임 시점에 통계 보고 (헤드리스 게이트)
      log(`studio frames=30 ${hudStats()} blur_test start`);
      $('blur').value = 60;
      $('blur').dispatchEvent(new Event('input'));
    }
    if (frames === 60) {
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
      // B티어 강제 — 완료 기준 검증: GPU 추론 없이 배경·블러·조명·프레이밍 생존
      if (cpuSeg) $('tier').value = 'b';
      // 3D 아이템은 얼굴이 있어야 의미가 있다 — 가짜 카메라(무얼굴)에선 배경만 검증

    }
    if (frames === 90) {
      log(`studio asset-bg+item ${hudStats()} face=${face ? 'loaded' : 'off'}`);
      log('studio verdict PASS');
      log('studio-done');
    }
    $('hud').textContent =
      `frame ${frames} | ${hudStats()} — 간격=체감 스루풋, gpu=제출→GPU 완료 실측(5s 샘플), submit=제출 벽시계(허수 참고)`;
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
