// audio-ab — fastenhancer wasm 게이트. 네이티브 audio.rs e2e와 같은 검사
// (SNR vs wav2wav + hop 예산)를 브라우저 CPU에서 재현한다.
//
// 헤드리스: node tools/run_web.mjs demo/audio-ab.html
// 완료 신호: AI_ENGINE_RESULT: audio ... / audio-done

const say = (t) => (document.getElementById('status').textContent = t);
const logEl = document.getElementById('log');
const log = (l) => {
  console.log('AI_ENGINE_RESULT: ' + l);
  logEl.textContent += l + '\n';
};

const BASE = '../models/fastenhancer/';
const HOP_BUDGET_MS = 1000 / (48000 / 512); // 10.67ms

async function main() {
  say('wasm 로드…');
  let aiMod;
  try {
    aiMod = await import('../pkg-relaxed/ai_engine.js');
    await aiMod.default();
  } catch {
    aiMod = await import('../pkg/ai_engine.js');
    await aiMod.default();
  }
  // ⚠ init_engine 호출하지 않음 — audio 워커엔 GPU가 없다 (계약 검증)

  say('자산 로드…');
  const fetchB = async (u) => {
    const r = await fetch(BASE + u);
    if (!r.ok) throw new Error(`${u} 없음 (make convert-fastenhancer)`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const graph = await fetchB('fe48/graph.json');
  const weights = await fetchB('fe48/weights.bin');
  const input = new Float32Array((await fetchB('in_48k.f32')).buffer);
  const reference = new Float32Array((await fetchB('ref_48k_wav2wav.f32')).buffer);

  // wasm op별 프로파일 (병목 확정용 — 헤드리스 로그)
  try {
    const prof = aiMod.enhancer_profile(graph, weights, 30);
    const total = prof.reduce((a, [, v]) => a + v, 0);
    log(
      'audio profile ' +
        prof
          .filter(([, v]) => v > 0.01)
          .map(([op, v]) => `${op}=${v.toFixed(3)}ms(${((v / total) * 100) | 0}%)`)
          .join(' ') +
        ` TOTAL=${total.toFixed(3)}ms`
    );
  } catch (e) {
    log('audio profile ERR ' + String(e).slice(0, 100));
  }

  const h = aiMod.enhancer_new(graph, weights);
  const hop = aiMod.enhancer_frame_len(h);
  if (hop !== 512) throw new Error(`hop ${hop} ≠ 512`);
  const hops = Math.floor(reference.length / hop);

  say(`처리 중… (${hops} hops)`);
  const out = new Float32Array(hops * hop);
  const t0 = performance.now();
  for (let i = 0; i < hops; i++) {
    const seg = input.subarray(i * hop, (i + 1) * hop);
    out.set(aiMod.enhancer_process(h, seg), i * hop);
  }
  const perHop = (performance.now() - t0) / hops;
  aiMod.enhancer_free(h);

  let se = 0, sr = 0;
  for (let i = 0; i < hops * hop; i++) {
    sr += reference[i] * reference[i];
    const d = out[i] - reference[i];
    se += d * d;
  }
  const snr = 10 * Math.log10(sr / Math.max(se, 1e-30));
  const snrOk = snr >= 45;
  const perfOk = perHop < HOP_BUDGET_MS * 0.5;
  log(
    `audio fe48 snr=${snr.toFixed(1)}dB hop=${perHop.toFixed(2)}ms ` +
      `(예산 ${HOP_BUDGET_MS.toFixed(2)}ms) snr=${snrOk ? 'ok' : 'FAIL'} ` +
      `perf=${perfOk ? 'ok' : 'FAIL'}`
  );
  const pass = snrOk && perfOk;

  // ── 비교 상대: ORT wasm으로 원본 wav2wav ONNX (1T — audio 워커 조건) ──
  // 참고용 — DFT op이 ORT wasm CPU EP에서 안 돌면 그 자체가 기록 (우리가 유일 경로)
  try {
    say('ORT wasm 비교…');
    const ort = await import('../compare/ort/ort.bundle.min.mjs');
    ort.env.wasm.numThreads = 1;
    const sess = await ort.InferenceSession.create(BASE + 'fastenhancer_b_48k.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const caches = {
      cache_in_0: new ort.Tensor('float32', new Float32Array(hop), [1, hop]),
      cache_in_1: new ort.Tensor('float32', new Float32Array(hop), [1, hop]),
      cache_in_2: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
      cache_in_3: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
      cache_in_4: new ort.Tensor('float32', new Float32Array(36 * 36), [1, 36, 36]),
    };
    // 워밍업 5 hop
    for (let i = 0; i < 5; i++) {
      await sess.run({
        wav_in: new ort.Tensor('float32', input.slice(i * hop, (i + 1) * hop), [1, hop]),
        ...caches,
      });
    }
    const t1 = performance.now();
    for (let i = 0; i < hops; i++) {
      const o = await sess.run({
        wav_in: new ort.Tensor('float32', input.slice(i * hop, (i + 1) * hop), [1, hop]),
        ...caches,
      });
      for (let k = 0; k < 5; k++) caches[`cache_in_${k}`] = o[`cache_out_${k}`];
    }
    const ortHop = (performance.now() - t1) / hops;
    await sess.release();
    log(
      `audio ort-wasm hop=${ortHop.toFixed(2)}ms — 우리 ${perHop.toFixed(2)}ms ` +
        `(${(ortHop / perHop).toFixed(2)}배)`
    );
  } catch (e) {
    log(`audio ort-wasm 불가: ${String(e).slice(0, 140)} (참고 — 우리 경로가 유일)`);
  }
  say(pass ? `PASS — SNR ${snr.toFixed(1)}dB, ${perHop.toFixed(2)}ms/hop` : 'FAIL');
  log(`audio verdict ${pass ? 'PASS' : 'FAIL'}`);
  log('audio-done');
}

main().catch((e) => {
  say('오류: ' + e);
  log(`audio fatal ${String(e).slice(0, 200)}`);
  log('audio-done');
});
