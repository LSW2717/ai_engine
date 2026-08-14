// ffi-diff — 네이티브(ai-ffi가 쓰는 GateHarness, Metal/naga)와 웹(Chrome/Tint)이
// 같은 WGSL 스택·같은 픽스처에서 같은 픽셀을 내는지 교차 증명.
//
// 선행: cargo test -p ai-ffi --release (web/models/ffi_native_*.bin 덤프)
// 헤드리스: node tools/run_web.mjs demo/ffi-diff.html
// 완료 신호: AI_ENGINE_RESULT: ffidiff ... / ffidiff-done

const say = (t) => (document.getElementById('status').textContent = t);
const logEl = document.getElementById('log');
const log = (l) => {
  console.log('AI_ENGINE_RESULT: ' + l);
  logEl.textContent += l + '\n';
};

const FW = 640, FH = 360;
const MW = 256, MH = 144;
// 같은 WGSL·같은 입력 — 컴파일러(naga vs Tint) 반올림 + blur 체인 누적만 허용
const TOL = { max: 4, mean: 0.6 };

// vb-diff.js 픽스처 1:1 (네이티브 테스트도 같은 식 — 3벌 동일)
function makeFrame() {
  const d = new Uint8Array(FW * FH * 4);
  for (let y = 0; y < FH; y++) {
    for (let x = 0; x < FW; x++) {
      const i = (y * FW + x) * 4;
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
const q255 = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255) / 255;
function makeMask(cx) {
  const m = new Float32Array(MW * MH);
  for (let y = 0; y < MH; y++) {
    for (let x = 0; x < MW; x++) {
      const nx = (x + 0.5) / MW - cx;
      const ny = (y + 0.5) / MH - 0.58;
      const d = Math.hypot(nx / 0.21, ny / 0.4);
      let v = 1 - (d - 1) / 0.25;
      if (y < MH * 0.18) v = 0;
      m[y * MW + x] = q255(v);
    }
  }
  return m;
}

const MODES = [
  ['color', { background: '#00a05a', blur: 0, brightness: 1, grayscale: 0, studioLight: null }],
  ['blur', { background: null, blur: 0.6, brightness: 1, grayscale: 0, studioLight: null }],
];

function diffReport(a, b) {
  let max = 0, sum = 0;
  for (let i = 0; i < a.length; i++) {
    if ((i & 3) === 3) continue; // 알파 제외
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
    sum += d;
  }
  return { max, mean: sum / (a.length * 0.75) };
}

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
  const mask = makeMask(0.5);
  let pass = true;

  for (const [name, cfg] of MODES) {
    say(`${name} 비교…`);
    const resp = await fetch(`../models/ffi_native_${name}.bin`);
    if (!resp.ok) {
      log(`ffidiff ${name} SKIP — 덤프 없음 (cargo test -p ai-ffi --release 먼저)`);
      pass = false;
      continue;
    }
    const native = new Uint8Array(await resp.arrayBuffer());
    aiMod.vb_gate_reset();
    aiMod.vb_gate_config(JSON.stringify(cfg));
    const ours = await aiMod.vb_gate_frame(seg, frame, FW, FH, mask, 1, MW, MH, false);
    const d = diffReport(ours, native);
    const ok = d.max <= TOL.max && d.mean <= TOL.mean;
    pass = pass && ok;
    log(`ffidiff ${name} max=${d.max} mean=${d.mean.toFixed(3)} ${ok ? 'PASS' : 'FAIL'}`);
    // 웹 결과 눈검증
    const cx = document.getElementById('ours').getContext('2d');
    const img = cx.createImageData(FW, FH);
    img.data.set(ours);
    cx.putImageData(img, 0, 0);
  }

  say(pass ? 'PASS' : 'FAIL');
  log(`ffidiff verdict ${pass ? 'PASS' : 'FAIL'}`);
  log('ffidiff-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`ffidiff fatal ${String(e).slice(0, 200)}`);
  log('ffidiff-done');
});
