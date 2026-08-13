// wasm CPU 프로파일 — cpu-ab 페이지를 실제 GPU 크로미움으로 띄우고 V8 샘플링
// 프로파일러(CDP)로 함수별 자기시간을 집계한다. 네이티브 예산표가 wasm에서
// 얼마나 뒤틀리는지 보는 용도 (run_web.mjs와 같은 실행 골격).
//
//   node tools/profile_web.mjs [demo/cpu-ab.html?only=ours] [--secs=15]

import { createRequire } from 'module';
import { spawn } from 'child_process';

const require_ = createRequire(import.meta.url);
const { chromium } = require_('/opt/homebrew/lib/node_modules/playwright');

const args = process.argv.slice(2);
const path = args.find((a) => !a.startsWith('--')) || 'demo/cpu-ab.html?only=ours';
const secs = Number((args.find((a) => a.startsWith('--secs=')) || '--secs=15').slice(7));
const PORT = 8124;

const server = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'web'], {
  stdio: 'ignore',
});
process.on('exit', () => server.kill());
await new Promise((r) => setTimeout(r, 700));

const browser = await chromium.launch({
  headless: true,
  args: [
    '--use-angle=metal', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
    '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
  ],
});
const page = await browser.newPage();
await page.context().grantPermissions(['camera']).catch(() => {});
page.on('console', (m) => {
  if (m.text().startsWith('AI_ENGINE_RESULT:')) console.error(m.text());
});

await page.goto(`http://127.0.0.1:${PORT}/${path}`, { waitUntil: 'domcontentloaded' });
const cdp = await page.context().newCDPSession(page);
await page.click('#start');
await new Promise((r) => setTimeout(r, 3000)); // 로드+웜업(티어업 포함) 흡수

// --ops: wasm 안에서 스텝별 반복 벤치 (100µs 타이머 양자화를 반복 합산으로 우회)
if (args.includes('--ops')) {
  const rows = await page.evaluate(() => window.__aiMod.profile_cpu(40));
  const total = rows.reduce((s, [, ms]) => s + ms, 0);
  const agg = new Map();
  for (const [label, ms] of rows) agg.set(label, (agg.get(label) || 0) + ms);
  const sorted = [...agg.entries()].sort((a, b) => b[1] - a[1]);
  console.log(`\n== wasm 스텝별 벤치 (40회 평균, 합계 ${total.toFixed(2)}ms) ==`);
  let cum = 0;
  for (const [label, ms] of sorted) {
    cum += ms;
    console.log(`${ms.toFixed(3).padStart(8)} ms  ${label}  (누적 ${((cum / total) * 100).toFixed(0)}%)`);
  }
  await browser.close();
  process.exit(0);
}

await cdp.send('Profiler.enable');
await cdp.send('Profiler.setSamplingInterval', { interval: 100 }); // 100µs
await cdp.send('Profiler.start');
await new Promise((r) => setTimeout(r, secs * 1000));
const { profile } = await cdp.send('Profiler.stop');

// 자기시간 집계: node.hitCount × 평균 샘플 간격
const total = profile.endTime - profile.startTime; // µs
const hits = profile.nodes.reduce((s, n) => s + (n.hitCount || 0), 0);
const perHit = total / hits;
const byFn = new Map();
for (const n of profile.nodes) {
  if (!n.hitCount) continue;
  const f = n.callFrame;
  let name = f.functionName || '(anonymous)';
  // wasm 프레임은 "name-worker.wasm-function[123]:0x..." 꼴 — 이름부만 남긴다
  name = name.replace(/:0x[0-9a-f]+$/, '');
  const key = `${name}  [${(f.url || '').split('/').pop().slice(0, 40)}]`;
  byFn.set(key, (byFn.get(key) || 0) + n.hitCount);
}
const rows = [...byFn.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30);
console.log(`\n== V8 샘플 프로파일 ${secs}s (샘플 ${hits}개, ${(perHit).toFixed(0)}µs/샘플) ==`);
for (const [name, h] of rows) {
  const ms = (h * perHit) / 1000;
  console.log(`${(ms / secs).toFixed(2).padStart(7)} ms/s  ${((h / hits) * 100).toFixed(1).padStart(5)}%  ${name}`);
}

await browser.close();
process.exit(0);
