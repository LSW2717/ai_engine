// vb-worker.js — VbEngine 게이트용 미니 워커: v-ai pipeline.worker의 프레임/설정
// 경로를 최소로 흉내 내 3함수 계약을 **진짜 Worker 경계**에서 검증한다.
//   in  {type:'config', config}                (VBOptions partial)
//   in  {type:'frame', bitmap, time, seq}      (bitmap transfer)
//   out {type:'frame', bitmap, passthrough, same, seq} (bitmap transfer —
//        same = 반환이 입력과 동일 객체였는가: 제로카피 passthrough 증명)
//   in  {type:'focus'} → out {type:'focus', json}
//   in  {type:'gesture'} → out {type:'gesture', json|null}
//   in  {type:'destroy'} → out {type:'destroyed'}
import * as engine from '../vb-engine.js';

self.onmessage = async (e) => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'config':
        engine.configCustomVideoStream(m.config);
        self.postMessage({ type: 'configured' });
        break;
      case 'frame': {
        const inBitmap = m.bitmap;
        const out = await engine.processWorkerFrame(inBitmap, m.time);
        const same = out.bitmap === inBitmap;
        self.postMessage(
          { type: 'frame', bitmap: out.bitmap, passthrough: out.passthrough, same, seq: m.seq },
          [out.bitmap]
        );
        break;
      }
      case 'focus':
        self.postMessage({ type: 'focus', json: engine.getFocusState() });
        break;
      case 'gesture':
        self.postMessage({ type: 'gesture', json: engine.pollGesture() });
        break;
      case 'destroy':
        engine.destroyCustomVideoStream();
        self.postMessage({ type: 'destroyed' });
        break;
    }
  } catch (err) {
    // 계약: 티어가 throw하면 워커는 입력 비트맵으로 패스스루 강등
    if (m.type === 'frame') {
      try {
        self.postMessage(
          { type: 'frame', bitmap: m.bitmap, passthrough: true, same: true, seq: m.seq, error: String(err) },
          [m.bitmap]
        );
      } catch {
        /* 이미 detach — 프레임 유실 */
      }
    } else {
      self.postMessage({ type: 'error', message: String(err), req: m.type });
    }
  }
};
self.postMessage({ type: 'ready' });
