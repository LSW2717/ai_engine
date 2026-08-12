#!/usr/bin/env python3
"""onnxruntime 오라클 — 시드 입력으로 ONNX를 실행해 입력·출력(·중간 텐서)을 .npy로 덤프.

ai_engine 변환기/런타임의 정확도 기준값 생성기. /usr/bin/python3 (onnx 1.17, ort 1.19) 전용.

사용:
  /usr/bin/python3 tools/onnx_oracle.py models/rvm_fp32.onnx --size 256x144 \
      --set-input downsample_ratio=1.0 --seed 7 --out target/oracle_rvm --intermediates

상태 입력(심볼릭)은 [1,1,1,1] 0으로 먹인다 — RVM의 Expand(rNi, Shape(x))가
0을 목표 shape으로 브로드캐스트하므로 zero-state 첫 프레임과 동치다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--size", required=True, help="WxH")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set-input", action="append", default=[], help="NAME=FLOAT")
    ap.add_argument("--input-shape", action="append", default=[], help="NAME=1x16x72x128 (0으로 채움)")
    ap.add_argument("--intermediates", action="store_true")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.split("x"))
    out_dir = Path(args.out)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.model)
    graph = model.graph
    orig_outputs = [o.name for o in graph.output]

    if args.intermediates:
        existing = set(orig_outputs)
        for node in graph.node:
            for out in node.output:
                if out and out not in existing:
                    graph.output.append(onnx.helper.make_empty_tensor_value_info(out))
                    existing.add(out)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # 결정성
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )

    set_inputs = dict(kv.split("=") for kv in args.set_input)
    input_shapes = {
        kv.split("=")[0]: [int(d) for d in kv.split("=")[1].split("x")]
        for kv in args.input_shape
    }
    initializers = {t.name for t in graph.initializer}
    rng = np.random.RandomState(args.seed)

    feeds = {}
    for vi in graph.input:
        name = vi.name
        if name in initializers:
            continue
        if name in set_inputs:
            feeds[name] = np.array([float(set_inputs[name])], dtype=np.float32)
        elif name in input_shapes:
            feeds[name] = np.zeros(input_shapes[name], dtype=np.float32)  # 상태 = 0
        else:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(dims) == 4 and (dims[1] == 3 or dims[1] == 0):
                feeds[name] = rng.uniform(0, 1, (1, 3, h, w)).astype(np.float32)
            elif len(dims) == 4 and all(d > 0 for d in dims):
                feeds[name] = rng.uniform(0, 1, dims).astype(np.float32)
            else:
                feeds[name] = np.zeros((1, 1, 1, 1), dtype=np.float32)

    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]

    manifest = {"inputs": {}, "outputs": {}, "intermediates": {}, "seed": args.seed, "size": [w, h]}
    for name, arr in feeds.items():
        f = f"inputs/{sanitize(name)}.npy"
        np.save(out_dir / f, arr)
        manifest["inputs"][name] = {"file": f, "shape": list(arr.shape)}
    for name, arr in zip(out_names, outs):
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
            continue
        f = f"{sanitize(name)}.npy"
        np.save(out_dir / f, arr)
        bucket = "outputs" if name in orig_outputs else "intermediates"
        manifest[bucket][name] = {"file": f, "shape": list(arr.shape)}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"oracle OK: 입력 {len(manifest['inputs'])} | 출력 {len(manifest['outputs'])} | "
        f"중간 {len(manifest['intermediates'])} → {out_dir}"
    )


if __name__ == "__main__":
    sys.exit(main())
