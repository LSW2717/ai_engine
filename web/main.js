// ai_engine WebGPU 데모 — 순수 ES 모듈, 빌드 프레임워크 없음.
// ?auto=1 이면 로드 직후 테스트+벤치를 자동 실행하고 결과를 콘솔에 남긴다(헤드리스 검증용).

import initWasm, {
  init_engine,
  run_tests,
  run_benchmarks,
  load_model,
  model_bench,
} from './pkg/ai_engine.js';

const $ = (id) => document.getElementById(id);
const showError = (msg) => {
  const el = $('error');
  el.style.display = 'block';
  el.textContent = msg;
};

function renderTests(results) {
  const failed = results.filter((r) => !r.passed).length;
  const rows = results
    .map(
      (r) => `<tr>
        <td>${r.name}</td>
        <td class="num">${r.max_err.toExponential(2)}</td>
        <td class="num">${r.tol.toExponential(0)}</td>
        <td class="${r.passed ? 'pass' : 'fail'}">${r.passed ? 'PASS' : 'FAIL'}</td>
      </tr>`
    )
    .join('');
  $('tests').innerHTML = `
    <h2>정확도: ${results.length - failed}/${results.length} 통과</h2>
    <table>
      <thead><tr><th>케이스</th><th class="num">max err</th><th class="num">tol</th><th>결과</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  return failed;
}

function renderBench(results) {
  const us = (v) => (v == null ? '-' : v.toFixed(1) + 'µs');
  const rows = results
    .map(
      (r) => `<tr>
        <td>${r.name}</td>
        <td class="num">${us(r.gpu_min_us)}</td>
        <td class="num">${us(r.gpu_median_us)}</td>
        <td class="num">${us(r.wall_us)}</td>
        <td class="num">${r.gflops.toFixed(1)}</td>
        <td class="num">${r.pipeline_ms.toFixed(2)}ms</td>
      </tr>`
    )
    .join('');
  const total = results.reduce((s, r) => s + (r.gpu_min_us ?? r.wall_us), 0);
  $('bench').innerHTML = `
    <h2>벤치마크 (디스패치당)</h2>
    <table>
      <thead><tr><th>커널</th><th class="num">GPU min</th><th class="num">GPU med</th>
        <th class="num">wall</th><th class="num">GFLOP/s</th><th class="num">파이프라인</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p>표 전체 합(GPU min 기준): <b>${(total / 1000).toFixed(3)} ms</b></p>`;
}

async function main() {
  if (!navigator.gpu) {
    $('banner').textContent = '';
    showError(
      'WebGPU 미지원 환경입니다. Chrome/Edge 최신 버전을 사용하거나, ' +
        'Safari는 WebGPU 기능 플래그를 켜주세요. (폴백 티어: webgl2-engine / ORT wasm)'
    );
    console.log('AI_ENGINE_RESULT: no-webgpu');
    return;
  }

  try {
    await initWasm();
    const info = await init_engine();
    $('banner').innerHTML =
      `adapter: <b>${info.name}</b> (${info.backend}) · ` +
      `f16 ${info.f16 ? '✓' : '✗'} · timestamps ${info.timestamps ? '✓' : '✗'}`;
  } catch (e) {
    $('banner').textContent = '';
    showError(`엔진 초기화 실패: ${e}`);
    console.log('AI_ENGINE_RESULT: init-failed', String(e));
    return;
  }

  const btnT = $('btn-tests');
  const btnB = $('btn-bench');
  const btnR = $('btn-rvm');
  btnT.disabled = btnB.disabled = btnR.disabled = false;

  const doTests = async () => {
    btnT.disabled = btnB.disabled = true;
    btnT.textContent = '테스트 실행 중…';
    try {
      const results = await run_tests();
      const failed = renderTests(results);
      console.log(`AI_ENGINE_RESULT: tests ${results.length - failed}/${results.length} passed`);
      return failed;
    } catch (e) {
      showError(`테스트 실패: ${e}`);
      console.log('AI_ENGINE_RESULT: tests-error', String(e));
    } finally {
      btnT.textContent = '정확도 테스트 실행';
      btnT.disabled = btnB.disabled = false;
    }
  };

  const doBench = async () => {
    btnT.disabled = btnB.disabled = true;
    btnB.textContent = '벤치마크 실행 중…';
    try {
      const results = await run_benchmarks();
      renderBench(results);
      const total = results.reduce((s, r) => s + (r.gpu_min_us ?? r.wall_us), 0);
      console.log(`AI_ENGINE_RESULT: bench total ${(total / 1000).toFixed(3)}ms over ${results.length} kernels`);
    } catch (e) {
      showError(`벤치마크 실패: ${e}`);
      console.log('AI_ENGINE_RESULT: bench-error', String(e));
    } finally {
      btnB.textContent = '벤치마크 실행';
      btnT.disabled = btnB.disabled = false;
    }
  };

  const doModel = async () => {
    btnT.disabled = btnB.disabled = btnR.disabled = true;
    btnR.textContent = 'RVM 실행 중…';
    try {
      const t0 = performance.now();
      const resp = await fetch('./models/rvm_256x144.sw');
      if (!resp.ok) throw new Error(`모델 fetch 실패 (${resp.status}) — make convert-rvm-web 후 재시도`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      const tFetch = performance.now();
      const report = await load_model(bytes);
      const tLoad = performance.now();
      const bench = await model_bench(30);
      $('model').innerHTML = `
        <h2>RVM (${report.name}) — <b>${bench.ms_per_frame.toFixed(2)}ms/frame</b></h2>
        <p>모델 ${report.weights_mb.toFixed(1)}MB · op ${report.ops} · 파이프라인 ${report.unique_pipelines}
          · arena ${report.arena_mb.toFixed(1)}MB<br/>
          fetch ${(tFetch - t0).toFixed(0)}ms · 로드(컴파일+워밍업) ${(tLoad - tFetch).toFixed(0)}ms
          · 출력: ${bench.output_names.join(', ')}</p>`;
      console.log(
        `AI_ENGINE_RESULT: rvm ${bench.ms_per_frame.toFixed(2)}ms/frame load ${(tLoad - tFetch).toFixed(0)}ms`
      );
    } catch (e) {
      showError(`RVM 실패: ${e}`);
      console.log('AI_ENGINE_RESULT: rvm-error', String(e));
    } finally {
      btnR.textContent = 'RVM 모델 로드 + 30프레임 추론';
      btnT.disabled = btnB.disabled = btnR.disabled = false;
    }
  };

  btnT.addEventListener('click', doTests);
  btnB.addEventListener('click', doBench);
  btnR.addEventListener('click', doModel);

  if (new URLSearchParams(location.search).get('auto') === '1') {
    await doTests();
    await doBench();
    await doModel();
    console.log('AI_ENGINE_RESULT: auto-done');
  }
}

main();
