#!/usr/bin/env python3
"""onnxruntime 기준값 덤프 — 시드 입력 1개와 전체 출력을 raw .f32로.

ai-cpu/tests/gate_ort_models.rs가 소비한다. /usr/bin/python3 (ort 1.19) 전용.

사용: /usr/bin/python3 tools/ort_dump.py <model.onnx> <출력 디렉토리>

입력은 [0,1) 균일난수 (이미지 입력과 같은 레인지 — 활성화가 죽지 않게).
NCHW 입력 모델(gaze)은 엔진용으로 NHWC 전치본을 쓴다. 파일:
  input_nhwc.f32              엔진 set_input용 (논리 NHWC 평탄)
  out__<출력이름>.f32          ORT 출력 (ONNX 평탄 순서 그대로)
"""

import os
import re
import sys

import numpy as np
import onnxruntime as ort

model, outdir = sys.argv[1], sys.argv[2]
inter = "--intermediates" in sys.argv
os.makedirs(outdir, exist_ok=True)
np.random.seed(7)

model_bytes = open(model, "rb").read()
if inter:
    # 전 노드 출력을 그래프 출력으로 승격 — 층별 대조용
    import onnx
    m = onnx.load_from_string(model_bytes)
    existing = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in existing:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
    model_bytes = m.SerializeToString()

sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [1 if not isinstance(d, int) else d for d in inp.shape]
x = np.random.rand(*shape).astype(np.float32)
res = sess.run(None, {inp.name: x})

xe = x
if len(shape) == 4 and shape[1] <= 4 and shape[2] > 8:  # NCHW 휴리스틱 (gaze)
    xe = np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1)))
xe.tofile(os.path.join(outdir, "input_nhwc.f32"))

for o, v in zip(sess.get_outputs(), res):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", o.name)
    v.astype(np.float32).tofile(os.path.join(outdir, f"out__{safe}.f32"))
    print(f"{o.name} -> out__{safe}.f32 {list(v.shape)}")
print(f"input {shape} -> input_nhwc.f32 {list(xe.shape)}")
