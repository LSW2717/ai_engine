// web/ 페이지를 실제 GPU 크로미움으로 자동 실행하고 콘솔의 AI_ENGINE_RESULT 줄을 받는다.
// 헤드리스 크로미움 기본값은 SwiftShader(소프트웨어)라 GPU 수치가 무의미하므로
// --use-angle=metal + GPU 강제 플래그로 실제 GPU를 쓰게 한다.
//
//   node tools/run_web.mjs compare/index.html [--headed] [--webkit]

import { createRequire } from 'module';
import { spawn } from 'child_process';

const require_ = createRequire(import.meta.url);
const { chromium, webkit } = require_('/opt/homebrew/lib/node_modules/playwright');

const args = process.argv.slice(2);
const path = args.find((a) => !a.startsWith('--')) || 'compare/index.html';
const headed = args.includes('--headed');
const useWebkit = args.includes('--webkit');
// --camera: 합성 카메라를 물려 데모 페이지를 자동 검증한다 (권한 프롬프트도 자동 승인)
const camera = args.includes('--camera');
const PORT = 8123;

const server = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'web'], {
  stdio: 'ignore',
});
const stop = () => server.kill();
process.on('exit', stop);

await new Promise((r) => setTimeout(r, 700));

const browser = useWebkit
  ? await webkit.launch({ headless: !headed })
  : await chromium.launch({
      headless: !headed,
      args: [
        '--use-angle=metal',
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--enable-gpu-rasterization',
        '--disable-gpu-sandbox',
        '--use-gl=angle',
        ...(camera
          ? [
              '--use-fake-device-for-media-stream',
              '--use-fake-ui-for-media-stream',
              // --video=<y4m>: 합성 패턴 대신 실제 사람 영상을 카메라로 물린다
              ...(args.find((a) => a.startsWith('--video='))
                ? [`--use-file-for-fake-video-capture=${args.find((a) => a.startsWith('--video=')).slice(8)}`]
                : []),
            ]
          : []),
      ],
    });

if (camera) {
  await browser.contexts()[0]?.grantPermissions?.(['camera']).catch(() => {});
}
const page = await browser.newPage();
if (camera) {
  await page.context().grantPermissions(['camera']).catch(() => {});
}
page.setDefaultTimeout(600000);

let done = false;
page.on('console', (m) => {
  const t = m.text();
  if (t.startsWith('AI_ENGINE_RESULT:')) {
    console.log(t);
    if (t.includes('-done')) done = true;
  } else if (t.includes('[ai-')) {
    console.log(t);
  } else if (/error|fail|exception/i.test(t)) {
    console.error(`[console] ${t}`);
  }
});
page.on('pageerror', (e) => console.error(`[pageerror] ${e.message}`));

await page.goto(`http://127.0.0.1:${PORT}/${path}`, { waitUntil: 'domcontentloaded' });
if (camera) {
  await page.click('#start');
  // --mask: 마스크만 보기 체크 (알파 자체를 눈으로 확인)
  if (args.includes('--bench')) {
    await new Promise((r) => setTimeout(r, 3000));
    await page.click('#bench').catch(() => {});
    await new Promise((r) => setTimeout(r, 6000));
  }
  if (args.includes('--mask')) {
    await new Promise((r) => setTimeout(r, 2000));
    await page.click('#showMask').catch(() => {});
  }
}

const renderer = await page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  return gl ? gl.getParameter(gl.RENDERER) : 'no webgl2';
});
console.error(`[renderer] ${renderer}`);
if (/swiftshader|software|llvmpipe/i.test(renderer)) {
  console.error('[경고] 소프트웨어 래스터라이저다. 타이밍 수치는 무의미.');
}

const t0 = Date.now();
const budget = camera ? (args.includes('--long') ? 90000 : 12000) : 300000;
while (!done && Date.now() - t0 < budget) await new Promise((r) => setTimeout(r, 300));
if (!done) console.error('[경고] 완료 신호 없이 타임아웃');

const shot = args.find((a) => a.startsWith('--shot='));
if (shot) {
  await page.screenshot({ path: shot.slice('--shot='.length) });
  console.error(`[shot] ${shot.slice('--shot='.length)}`);
}
await browser.close();
stop();
process.exit(0);
