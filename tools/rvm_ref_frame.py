#!/usr/bin/env python3
"""실제 이미지 1프레임에 대한 ORT 기준값(pha) 생성기 — 정확도 게이트의 정답지.

왜 필요한가: 엔진 자체 CPU 실행기와 GPU는 **같은 lowering을 공유**하므로 둘이 같이
틀리면 통과한다. 실제로 tanh 오버플로 버그가 그렇게 빠져나가 마스크를 반토막 냈고,
`rvm_e2e`는 그 상태에서도 5.8e-6으로 통과했다. 랜덤 노이즈 입력은 활성화 전 값이
작아 그 버그를 만들지도 못한다. **외부 구현(ORT) + 자연 이미지**만이 진짜 게이트다.

사용 (onnxruntime 있는 인터프리터로):
  /Users/foxcom/Desktop/segm-ft/venv/bin/python tools/rvm_ref_frame.py \
      models/rvm_fp32.onnx tests/data/frame_256x144.rgb tests/data/pha_ref_256x144.f32
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

W, H = 256, 144


def main() -> None:
    model, frame_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = np.frombuffer(Path(frame_path).read_bytes(), dtype=np.uint8)
    assert raw.size == W * H * 3, f"rgb24 {W}x{H}가 아님: {raw.size}"
    # ONNX RVM은 NCHW [1,3,H,W], 0..1
    src = raw.reshape(H, W, 3).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    feeds = {"src": src, "downsample_ratio": np.array([1.0], dtype=np.float32)}
    # 상태는 심볼릭 — [1,1,1,1] 0을 주면 Expand가 목표 shape으로 브로드캐스트한다
    # (zero-state 첫 프레임과 동치).
    for i in range(1, 5):
        feeds[f"r{i}i"] = np.zeros((1, 1, 1, 1), dtype=np.float32)
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(None, feeds)))

    pha = np.ascontiguousarray(outs["pha"].reshape(-1).astype(np.float32))
    Path(out_path).write_bytes(pha.tobytes())
    fg = float((pha > 0.5).mean())
    print(f"pha {pha.shape} 범위 [{pha.min():.4f}, {pha.max():.4f}] 전경비율 {fg * 100:.1f}%")
    print(f"-> {out_path}")
    if not 0.02 < fg < 0.95:
        print("⚠ 전경비율이 극단적이다 — 사람이 없는 프레임일 수 있다", file=sys.stderr)


if __name__ == "__main__":
    main()
