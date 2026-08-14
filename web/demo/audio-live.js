// audio-live — 10초 녹음 → 모델 선택 → 출력.
//
// 흐름: 마이크 10초를 버퍼에 담고(녹음 중 재생 없음), 선택한 엔진으로
// **오프라인 일괄 처리**(ai_engine ~1초, ORT ~1.5초)한 뒤 재생한다.
// 같은 녹음을 모델만 바꿔 반복 청취 — 결과는 엔진별 캐시, 새 녹음 시 초기화.
// 처리 완료 트랙은 WAV 다운로드 링크로도 제공.
//
// 헤드리스 스모크: node tools/run_web.mjs 'demo/audio-live.html?smoke=1' --camera
//   (녹음 2초로 단축, 가짜 장치 톤 → 양 엔진 처리·재생까지 자동)

const say = (t) => (document.getElementById('status').textContent = t);
const log = (l) => console.log('AI_ENGINE_RESULT: ' + l);
const $ = (id) => document.getElementById(id);

const BASE = '../models/fastenhancer/';
const SR = 48000;
const SMOKE = new URLSearchParams(location.search).has('smoke');
const REC_SECS = SMOKE ? 2 : 10;

let aiMod = null;
let h = null;
let hop = 512;
let ctx = null;
let stream = null;
let recBuf = null; // Float32Array — 마지막 녹음
let results = {}; // engine별 처리 결과 캐시 { engine, ort, off }
let procMs = {}; // engine별 처리 시간
let srcNode = null; // 재생 중 노드

// ── ORT (lazy) ──
let ort = null;
let ortSess = null;

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

async function ensureEngine() {
  if (aiMod) return;
  say('wasm + 모델 로드…');
  await loadAiWasm();
  const fetchB = async (u) => {
    const r = await fetch(BASE + u);
    if (!r.ok) throw new Error(`${u} 없음 (make convert-fastenhancer)`);
    return new Uint8Array(await r.arrayBuffer());
  };
  h = aiMod.enhancer_new(await fetchB('fe48/graph.json'), await fetchB('fe48/weights.bin'));
  hop = aiMod.enhancer_frame_len(h);
}

async function ensureOrt() {
  if (ortSess) return;
  say('onnxruntime 로드…');
  ort = await import('../compare/ort/ort.bundle.min.mjs');
  ort.env.wasm.numThreads = 1;
  ortSess = await ort.InferenceSession.create(BASE + 'fastenhancer_b_48k.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

// ── 녹음: ScriptProcessor 캡처 (hop 정렬, 48k) ──
async function record() {
  await ensureEngine();
  if (!ctx) ctx = new AudioContext({ sampleRate: SR });
  await ctx.resume();
  if (!stream) {
    say('마이크 여는 중…');
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }
  stopPlayback();
  results = {};
  procMs = {};
  $('dl').textContent = '';
  $('play').disabled = true;

  const total = SR * REC_SECS;
  recBuf = new Float32Array(Math.floor(total / hop) * hop);
  let off = 0;
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(hop, 1, 1);
  await new Promise((resolve) => {
    proc.onaudioprocess = (e) => {
      const inp = e.inputBuffer.getChannelData(0);
      if (off < recBuf.length) {
        recBuf.set(inp.subarray(0, Math.min(hop, recBuf.length - off)), off);
        off += hop;
        say(`● 녹음 중… ${Math.max(0, REC_SECS - off / SR).toFixed(1)}초 — 말해보세요`);
      } else {
        resolve();
      }
    };
    src.connect(proc);
    proc.connect(ctx.destination); // 그래프 구동용 (out 미기록 = 무음)
  });
  proc.disconnect();
  src.disconnect();
  say('녹음 완료 — 모델을 고르고 ▶ 출력');
  $('play').disabled = false;
}

// ── 처리 (엔진별 캐시) ──
async function processWith(kind) {
  if (results[kind]) return results[kind];
  const hops = recBuf.length / hop;
  let out;
  if (kind === 'off') {
    out = recBuf;
  } else if (kind === 'engine') {
    say('ai_engine 처리 중…');
    aiMod.enhancer_reset(h);
    const t0 = performance.now();
    out = new Float32Array(recBuf.length);
    for (let i = 0; i < hops; i++) {
      out.set(aiMod.enhancer_process(h, recBuf.subarray(i * hop, (i + 1) * hop)), i * hop);
    }
    procMs[kind] = performance.now() - t0;
  } else {
    await ensureOrt();
    say('onnxruntime 처리 중…');
    const caches = {
      cache_in_0: new ort.Tensor('float32', new Float32Array(hop), [1, hop]),
      cache_in_1: new ort.Tensor('float32', new Float32Array(hop), [1, hop]),
      cache_in_2: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
      cache_in_3: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
      cache_in_4: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
    };
    const t0 = performance.now();
    out = new Float32Array(recBuf.length);
    for (let i = 0; i < hops; i++) {
      const o = await ortSess.run({
        wav_in: new ort.Tensor(
          'float32',
          Float32Array.from(recBuf.subarray(i * hop, (i + 1) * hop)),
          [1, hop]
        ),
        ...caches,
      });
      out.set(o.wav_out.data, i * hop);
      for (let k = 0; k < 5; k++) caches[`cache_in_${k}`] = o[`cache_out_${k}`];
    }
    procMs[kind] = performance.now() - t0;
  }
  results[kind] = out;
  updateDl();
  return out;
}

// ── 재생 ──
function stopPlayback() {
  if (srcNode) {
    try {
      srcNode.stop();
    } catch {
      /* 이미 종료 */
    }
    srcNode = null;
  }
}

async function play() {
  const kind = $('engine').value;
  const buf = await processWith(kind);
  stopPlayback();
  const ab = ctx.createBuffer(1, buf.length, SR);
  ab.copyToChannel(buf, 0);
  srcNode = ctx.createBufferSource();
  srcNode.buffer = ab;
  srcNode.connect(ctx.destination);
  srcNode.onended = () => say('재생 끝 — 모델 바꿔서 다시 ▶');
  srcNode.start();
  const label =
    kind === 'engine' ? 'ai_engine' : kind === 'ort' ? 'onnxruntime' : '원음';
  say(`▶ 재생 중 — ${label}`);
  $('hud').textContent = Object.entries(procMs)
    .map(([k, v]) => `${k === 'engine' ? 'ai_engine' : 'ORT'} 처리 ${(v / 1000).toFixed(2)}s`)
    .join('  |  ');
}

// ── WAV 다운로드 ──
const wavBlob = (samples) => {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const wstr = (o, s2) => {
    for (let i = 0; i < s2.length; i++) v.setUint8(o + i, s2.charCodeAt(i));
  };
  wstr(0, 'RIFF');
  v.setUint32(4, 36 + n * 2, true);
  wstr(8, 'WAVEfmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, SR, true);
  v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  wstr(36, 'data');
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
};

function updateDl() {
  const names = { off: '원음', engine: 'ai_engine', ort: 'onnxruntime' };
  const links = Object.entries(results)
    .map(([k, buf]) => {
      const url = URL.createObjectURL(wavBlob(buf));
      return `<a href="${url}" download="rec_${names[k]}.wav">${names[k]}.wav</a>`;
    })
    .join(' · ');
  $('dl').innerHTML = links ? '저장: ' + links : '';
}

$('start').addEventListener('click', () => {
  record()
    .then(async () => {
      results.off = recBuf; // 원음은 즉시 사용 가능
      updateDl();
      if (SMOKE) {
        await processWith('engine');
        await processWith('ort');
        $('engine').value = 'engine';
        await play();
        const ok = recBuf.some((v) => v !== 0) && !!results.engine && !!results.ort;
        log(
          `audio-live rec=${REC_SECS}s engine=${(procMs.engine / 1000).toFixed(2)}s ` +
            `ort=${(procMs.ort / 1000).toFixed(2)}s`
        );
        log(`audio-live verdict ${ok ? 'PASS' : 'FAIL'}`);
        log('audio-live-done');
      }
    })
    .catch((e) => {
      say('오류: ' + e);
      log(`audio-live fatal ${String(e).slice(0, 200)}`);
      log('audio-live-done');
    });
});

$('play').addEventListener('click', () => {
  play().catch((e) => say('오류: ' + e));
});

// 재생 중 모델을 바꾸면 즉시 그 모델로 다시 출력
$('engine').addEventListener('input', () => {
  if (srcNode) play().catch((e) => say('오류: ' + e));
});
