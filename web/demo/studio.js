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
  for (const id of ['light', 'blur', 'bright', 'gray']) $(id).addEventListener('input', push);
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
    video: { width: 640, height: 360 },
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
  log('studio init ok');

  const src = $('src');
  const sctx = src.getContext('2d');
  const times = [];
  let frames = 0;
  say('가동 중');
  const tick = async () => {
    sctx.drawImage(video, 0, 0, src.width, src.height);
    const t0 = performance.now();
    try {
      await aiMod.studio_frame(seg, src);
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
      const s = [...times.slice(-25)].sort((a, b) => a - b);
      log(`studio frames=30 p50=${s[s.length >> 1].toFixed(2)}ms blur_test start`);
      $('blur').value = 60;
      $('blur').dispatchEvent(new Event('input'));
    }
    if (frames === 60) {
      const s = [...times.slice(-25)].sort((a, b) => a - b);
      log(`studio blur p50=${s[s.length >> 1].toFixed(2)}ms`);
      // 실제 제품 배경 + 3D 아이템 경로 (가짜 카메라엔 얼굴이 없어 오버레이는
      // null 경로를 탄다 — 크래시 없이 도는지가 스모크 목적)
      $('bg').value = 'asset:1';
      $('bg').dispatchEvent(new Event('input'));
      $('light').checked = true;
      $('light').dispatchEvent(new Event('input'));
      // 3D 아이템은 얼굴이 있어야 의미가 있다 — 가짜 카메라(무얼굴)에선 배경만 검증

    }
    if (frames === 90) {
      const s = [...times.slice(-25)].sort((a, b) => a - b);
      log(`studio asset-bg+item p50=${s[s.length >> 1].toFixed(2)}ms face=${face ? 'loaded' : 'off'}`);
      log('studio verdict PASS');
      log('studio-done');
    }
    const s = [...times.slice(-30)].sort((a, b) => a - b);
    $('hud').textContent =
      `frame ${frames} | p50 ${s.length ? s[s.length >> 1].toFixed(2) : '-'}ms (업로드+전처리+추론+합성 벽시계)`;
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
