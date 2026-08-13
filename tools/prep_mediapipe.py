#!/usr/bin/env python3
"""MediaPipe .task → ai-convert가 먹는 onnx 준비 (unzip → tf2onnx → 후처리).

.task는 zip이고 MediaPipe가 실제로 쓰는 tflite가 그대로 들어 있다 (NEXT.md §3).
tf2onnx --tflite 경로는 tensorflow 없이 돈다 (flatbuffer 직접 파싱).

후처리 2종:
  1. tf2onnx가 선언만 하고 안 쓰는 ai.onnx.ml opset 제거 (임포터가 도메인 선언에서 거절)
  2. 심볼릭 배치 차원(unk__*) → 1 고정 (face_landmarks가 걸림)

사용: /usr/bin/python3 tools/prep_mediapipe.py <face_landmarker.task|hand_landmarker.task|*.tflite>... -o models/mediapipe
필요: /usr/bin/python3 (onnx 1.17) + pip install --user tf2onnx
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import onnx


def postprocess(path: Path):
    m = onnx.load(str(path))
    used = {n.domain for n in m.graph.node}
    keep = [o for o in m.opset_import if o.domain in used]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    for vi in list(m.graph.input) + list(m.graph.output):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:
                d.ClearField("dim_param")
                d.dim_value = 1
    onnx.save(m, str(path))


def tflite_to_onnx(tflite: Path, out_dir: Path):
    out = out_dir / (tflite.stem + ".onnx")
    subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(tflite),
         "--output", str(out), "--opset", "13"],
        check=True, capture_output=True,
    )
    postprocess(out)
    print(f"{tflite.name} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="models/mediapipe")
    args = ap.parse_args()
    out_root = Path(args.out)
    for inp in map(Path, args.inputs):
        if inp.suffix == ".task":
            # face_landmarker.task -> models/mediapipe/face/ 처럼 짧은 이름 디렉토리
            sub = out_root / inp.stem.split("_")[0]
            sub.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(inp) as z:
                z.extractall(sub)
            for t in sorted(sub.glob("*.tflite")):
                tflite_to_onnx(t, sub)
        elif inp.suffix == ".tflite":
            out_root.mkdir(parents=True, exist_ok=True)
            tflite_to_onnx(inp, out_root)
        elif inp.suffix == ".onnx":
            postprocess(inp)
            print(f"{inp} 후처리 완료")
        else:
            sys.exit(f"모르는 입력: {inp}")


if __name__ == "__main__":
    main()
