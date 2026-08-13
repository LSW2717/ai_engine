// vb-diff — 웹 파이프라인 픽셀 diff 게이트 (P1 완료 조건).
//
// 같은 결정적 프레임(640×360) + 같은 마스크(256×144)를
//   A) ai_engine WGSL 스택 (vb_gate_frame — 추론 없음, 마스크 직주입)
//   B) v-ai GLSL 스택 (vai-stack.js — vendor 사본에서 스테이지 추출 조립)
// 에 주입해 최종 합성 RGBA를 채널별 diff한다.
//
// 프로토콜 2상 (EMA 양자화 고착·분기 경계 문제를 설계로 회피 — 정찰 §4):
//  [S] 공간 스택: ch=1 알파 직주입(v-ai는 EMA 이후 지점), 우리는 ema=false.
//      마스크는 1/255 격자로 사전 양자화 → 양쪽 u8 동일 → 프레임 1장이 결정적.
//      모드 5종 × 마스크 2종.
//  [E] softmax+EMA: ch=2 로짓 주입(v-ai buildSoftmaxStage + tflite 스텁).
//      frame1 = 제로 히스토리(diff=curr, 0.3 경계는 1/255 격자가 비껴감),
//      frame2 = |Δprob|≤0.18로 설계(동적 α 분기 경계 0.3에서 마진 0.12 —
//      히스토리 1LSB 차이로 분기가 갈리는 픽셀을 원천 차단).
//
// 헤드리스: make vai-gate-assets && node tools/run_web.mjs demo/vb-diff.html
// 완료 신호: AI_ENGINE_RESULT: vbdiff ... / vbdiff-done

import { createVaiStack } from './vai-stack.js';

const say = (t) => (document.getElementById('status').textContent = t);
const logEl = document.getElementById('log');
const log = (l) => {
  console.log('AI_ENGINE_RESULT: ' + l);
  logEl.textContent += l + '\n';
};

const FW = 640, FH = 360;   // 프레임(합성) 해상도 — 5의 배수 (blur 0.2 스케일 정수)
const MW = 256, MH = 144;   // 마스크(모델) 해상도 — RVM 256×144

// ── 결정적 픽스처 (캔버스 드로잉 없이 순수 산술 — 브라우저 무관 재현) ──

function makeFrame() {
  const d = new Uint8Array(FW * FH * 4);
  for (let y = 0; y < FH; y++) {
    for (let x = 0; x < FW; x++) {
      const i = (y * FW + x) * 4;
      // 부드러운 그라데이션 + 고대비 블록/체커 (JBF 색 가중·엣지 정제 자극)
      let r = Math.round((x / FW) * 255);
      let g = Math.round((y / FH) * 255);
      let b = Math.round(((x + y) / (FW + FH)) * 255);
      if (x > FW * 0.55 && x < FW * 0.8 && y > FH * 0.2 && y < FH * 0.5) {
        r = 220; g = 60; b = 40;
      }
      if (((x >> 4) + (y >> 4)) % 2 === 0 && x < FW * 0.25 && y > FH * 0.6) {
        r = 255 - r; g = 255 - g; b = 255 - b;
      }
      d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255;
    }
  }
  return d;
}

// 배경 이미지 (image 모드) — 프레임과 같은 종횡비 800×450 (cover 항등,
// fitBackgroundForCanvas 비개입 — 크롭 수학 파리티는 mirror/degree 작업에서)
const BGW = 800, BGH = 450;
function makeBgImage() {
  const d = new Uint8Array(BGW * BGH * 4);
  for (let y = 0; y < BGH; y++) {
    for (let x = 0; x < BGW; x++) {
      const i = (y * BGW + x) * 4;
      d[i] = Math.round(128 + 100 * Math.sin(x * 0.03));
      d[i + 1] = Math.round(128 + 100 * Math.sin(y * 0.05));
      d[i + 2] = Math.round((1 - y / BGH) * 200);
      d[i + 3] = 255;
    }
  }
  return d;
}

const q255 = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255) / 255;

// 소프트 타원 마스크 (인물 근사) — 1/255 격자 사전 양자화 (양쪽 u8 동일 보장)
function makeMask(cx) {
  const m = new Float32Array(MW * MH);
  for (let y = 0; y < MH; y++) {
    for (let x = 0; x < MW; x++) {
      const nx = (x + 0.5) / MW - cx;
      const ny = (y + 0.5) / MH - 0.58;
      const d = Math.hypot(nx / 0.21, ny / 0.4);
      let v = 1 - (d - 1) / 0.25; // 내부 1, 소프트 엣지
      if (y < MH * 0.18) v = 0;   // 하드 컷 (하드 엣지도 게이트)
      m[y * MW + x] = q255(v);
    }
  }
  return m;
}

// [E상] 로짓 픽스처: bg=0, person=logit(p) — frame1은 maskA 기반,
// frame2는 |Δprob|≤0.18로 히스토리 시뮬레이션 기반 설계 (동적 α 경계 회피)
const logit = (p) => Math.log(p / (1 - p));
function makeLogitsA(maskA) {
  const l = new Float32Array(MW * MH * 2);
  for (let i = 0; i < MW * MH; i++) {
    const p = Math.min(0.996, Math.max(0.004, maskA[i]));
    l[i * 2] = 0;
    l[i * 2 + 1] = logit(p);
  }
  return l;
}
function makeLogitsB(logitsA) {
  const l = new Float32Array(MW * MH * 2);
  for (let i = 0; i < MW * MH; i++) {
    const pA = 1 / (1 + Math.exp(-logitsA[i * 2 + 1]));
    // frame1 EMA 시뮬레이션: prev=0, α = pA>0.3 ? 0.9 : 0.03, u8 양자화
    const prev = q255((pA > 0.3 ? 0.9 : 0.03) * pA);
    const x = i % MW, y = (i / MW) | 0;
    const delta = 0.18 * Math.sin(x * 0.05) * Math.cos(y * 0.07);
    const pB = Math.min(0.99, Math.max(0.01, prev + delta));
    l[i * 2] = 0;
    l[i * 2 + 1] = logit(pB);
  }
  return l;
}

// ── diff ──

function diffReport(a, b) {
  let max = 0, sum = 0, over2 = 0;
  const n = FW * FH;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a[i * 4 + c] - b[i * 4 + c]);
      if (d > max) max = d;
      sum += d;
      if (d > 2) over2++;
    }
  }
  return { max, mean: sum / (n * 3), over2frac: over2 / (n * 3) };
}

function paint(id, rgba) {
  document.getElementById(id).getContext('2d').putImageData(
    new ImageData(new Uint8ClampedArray(rgba), FW, FH), 0, 0
  );
}

// v-ai _compositeFrame의 캔버스 2D 크롭 transform 재현 (전체 크롭 참고 비교용)
function crop2d(rgba, scale, cx, cy) {
  const src = document.createElement('canvas');
  src.width = FW;
  src.height = FH;
  src.getContext('2d').putImageData(
    new ImageData(new Uint8ClampedArray(rgba), FW, FH), 0, 0
  );
  const dst = document.createElement('canvas');
  dst.width = FW;
  dst.height = FH;
  const c = dst.getContext('2d');
  const z = 1 / scale;
  const sx = (cx - scale / 2) * FW;
  const sy = (cy - scale / 2) * FH;
  c.setTransform(z, 0, 0, z, -sx * z, -sy * z);
  c.drawImage(src, 0, 0, FW, FH);
  return new Uint8Array(c.getImageData(0, 0, FW, FH).data);
}

function paintDiff(a, b) {
  const d = new Uint8ClampedArray(FW * FH * 4);
  for (let i = 0; i < FW * FH; i++) {
    for (let c = 0; c < 3; c++) {
      d[i * 4 + c] = Math.min(255, Math.abs(a[i * 4 + c] - b[i * 4 + c]) * 16);
    }
    d[i * 4 + 3] = 255;
  }
  document.getElementById('diffv').getContext('2d').putImageData(new ImageData(d, FW, FH), 0, 0);
}

// ── 게이트 모드 ──

const LIGHTS = {
  enabled: true,
  ambient: 0.85,
  lights: [
    { enabled: true, x: 0.3, y: 0.25, color: '#ffd9a0', intensity: 0.9, radius: 0.6, target: 'person' },
    { enabled: true, x: 0.85, y: 0.7, color: '#9fc4ff', intensity: 0.5, radius: 0.5, target: 'background' },
  ],
};

// EffectsPatch 모양 — 우리 쪽은 JSON 그대로, v-ai 쪽은 vai-stack.configure가 매핑
const MODES = [
  ['pass', { background: null, blur: 0, brightness: 1, grayscale: 0, studioLight: null }],
  ['blur', { background: null, blur: 0.6, brightness: 1, grayscale: 0, studioLight: null }],
  ['color', { background: '#00a05a', blur: 0, brightness: 1, grayscale: 0, studioLight: null }],
  ['image', { background: 'image', blur: 0, brightness: 1, grayscale: 0, studioLight: null }],
  ['image-blur', { background: 'image', blur: 0.5, brightness: 1, grayscale: 0, studioLight: null }],
  ['bright-gray', { background: null, blur: 0.6, brightness: 1.25, grayscale: 0.6, studioLight: null }],
  ['light', { background: '#334155', blur: 0, brightness: 1, grayscale: 0, studioLight: LIGHTS }],
];

// 판정 문턱 (채널값 0..255): GL↔WebGPU 보간·반올림·중간 r8 양자화 누적은 허용,
// 수식·상수 불일치는 잡는다.
const TOL = { max: 4, mean: 0.6 };

async function main() {
  say('엔진 초기화…');
  let aiMod;
  try {
    aiMod = await import('../pkg-relaxed/ai_engine.js');
    await aiMod.default();
  } catch {
    aiMod = await import('../pkg/ai_engine.js');
    await aiMod.default();
  }
  if (!navigator.gpu) throw new Error('WebGPU 미지원');
  await aiMod.init_engine();
  const seg = (
    await aiMod.load_model_h(
      new Uint8Array(await (await fetch('../models/rvm_256x144.sw')).arrayBuffer())
    )
  ).handle;

  const frame = makeFrame();
  const bgImg = makeBgImage();
  const maskA = makeMask(0.5);
  const maskB = makeMask(0.58);
  paint('srcv', frame);

  say('v-ai GLSL 스택 준비…');
  const vai = await createVaiStack(document.getElementById('vai'), FW, FH, MW, MH);

  let allPass = true;
  const check = (name, ours, theirs, tol = TOL) => {
    const d = diffReport(ours, theirs);
    const pass = d.max <= tol.max && d.mean <= tol.mean;
    allPass = allPass && pass;
    log(
      `vbdiff ${name} max=${d.max} mean=${d.mean.toFixed(3)} ` +
        `over2=${(d.over2frac * 100).toFixed(3)}% ${pass ? 'PASS' : 'FAIL'}`
    );
    paint('ours', ours);
    // 'vai' 캔버스는 GL 컨텍스트 — 마지막 렌더가 그대로 표시된다 (putImageData 불가)
    paintDiff(ours, theirs);
    return pass;
  };

  // ── [S] 공간 스택: ch=1, EMA off, 모드 × 마스크 ──
  for (const [name, cfg] of MODES) {
    aiMod.vb_gate_reset();
    if (cfg.background === 'image') aiMod.vb_gate_bg_image(bgImg, BGW, BGH);
    aiMod.vb_gate_config(JSON.stringify(cfg));
    vai.reset();
    vai.configure(cfg, cfg.background === 'image' ? { data: bgImg, w: BGW, h: BGH } : null);
    for (const [mn, mask] of [['A', maskA], ['B', maskB]]) {
      const ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, mask, 1, false);
      const theirs = vai.frame(frame, mask, 1);
      check(`S/${name}/${mn}`, ours, theirs);
    }
  }

  // ── [F] 프레이밍 크롭 수학: 고정 (scale, cx, cy) — 스무딩 우회 ──
  // image 스테이지 경로(단색/이미지 배경)는 v-ai GL updateFraming과 정확 비교.
  // 전체 크롭(blur 모드)은 v-ai에선 캔버스 2D transform(래스터 후 재샘플 = 이중
  // 필터)이고 우리는 셰이더 단일 필터 — v-ai 자신도 스테이지가 지원하면 셰이더
  // 크롭을 우선하므로(framingInShader) 참고 수치로만 기록, 판정 제외.
  {
    const FR = [0.62, 0.46, 0.55];
    for (const name of ['color', 'image']) {
      const cfg = MODES.find(([n]) => n === name)[1];
      aiMod.vb_gate_reset();
      if (cfg.background === 'image') aiMod.vb_gate_bg_image(bgImg, BGW, BGH);
      aiMod.vb_gate_config(JSON.stringify(cfg));
      aiMod.vb_gate_framing(...FR);
      vai.reset();
      vai.configure(cfg, cfg.background === 'image' ? { data: bgImg, w: BGW, h: BGH } : null);
      vai.setFraming(...FR);
      const ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, maskA, 1, false);
      const theirs = vai.frame(frame, maskA, 1);
      check(`F/${name}/person-crop`, ours, theirs);
    }
    // 참고: 전체 크롭 (v-ai 2D transform 재현 — 이중 필터라 수 LSB 차이가 정상)
    {
      const cfg = MODES.find(([n]) => n === 'blur')[1];
      aiMod.vb_gate_reset();
      aiMod.vb_gate_config(JSON.stringify(cfg));
      aiMod.vb_gate_framing(...FR);
      vai.reset();
      vai.configure(cfg, null);
      const ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, maskA, 1, false);
      const theirs = crop2d(vai.frame(frame, maskA, 1), ...FR);
      const d = diffReport(ours, theirs);
      log(
        `vbdiff F/blur/whole-crop(참고) max=${d.max} mean=${d.mean.toFixed(3)} ` +
          `over2=${(d.over2frac * 100).toFixed(3)}% (판정 제외 — 이중 필터 차)`
      );
    }
  }

  // ── [E] softmax+EMA: ch=2, EMA on, blur 모드 ──
  {
    const cfg = MODES[1][1]; // blur 0.6
    aiMod.vb_gate_reset();
    aiMod.vb_gate_config(JSON.stringify(cfg));
    vai.reset();
    vai.configure(cfg, null);
    const logitsA = makeLogitsA(maskA);
    const logitsB = makeLogitsB(logitsA);
    let ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, logitsA, 2, true);
    let theirs = vai.frame(frame, logitsA, 2);
    check('E/blur/f1-zero-history', ours, theirs);
    ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, logitsB, 2, true);
    theirs = vai.frame(frame, logitsB, 2);
    check('E/blur/f2-small-delta', ours, theirs);
  }

  log(`vbdiff verdict ${allPass ? 'PASS' : 'FAIL'}`);
  log('vbdiff-done');
  say(allPass ? '전체 PASS' : '불일치 있음 — 로그 확인');
}

main().catch((e) => {
  console.error(e);
  log(`vbdiff fatal ${String(e).slice(0, 300)}`);
  log('vbdiff-done');
});
