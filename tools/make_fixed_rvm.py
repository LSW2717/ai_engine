#!/usr/bin/env python3
"""RVM 고정 export 재생성 — models/rvm_fp32.onnx(공식, 동적 shape) →
models/rvm_fixed_144x256.onnx(고정 144×256, downsample_ratio 동결, 116 op급).

원본 고정 export 파일이 유실돼(gitignore 모델류) 재생성한다. 방법:
  1) 입력 dim 고정 (src 1×3×144×256, r1i~r4i 실측 상태 shape)
  2) downsample_ratio 입력 → 상수 initializer 1.0
  3) onnxruntime BASIC 그래프 최적화로 저장 — 상수 접기/불필요 노드 제거만
     (표준 ONNX 유지 — EXTENDED의 ORT 전용 융합 op는 안 씀)
  4) 입력명 리네임: src→input_1, r{i}i→r{i} (기존 고정 export 계약 —
     prof_isolated/diag_* 테스트와 webgl2 plan.json의 input_1 규약)
  5) 게이트: 원본(ratio=1.0 피드) vs 고정본을 같은 시드 입력으로 ORT 실행,
     6개 출력 max|diff| 출력 — 1e-5 초과면 실패로 종료.

/usr/bin/python3 전용 (onnx 1.17, ort 1.19 — onnx_oracle.py와 동일 규약).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference
import onnxruntime as ort

SRC = "models/rvm_fp32.onnx"
DST = "models/rvm_fixed_144x256.onnx"
H, W = 144, 256
STATE_SHAPES = {
    "r1i": [1, 16, 72, 128],
    "r2i": [1, 20, 36, 64],
    "r3i": [1, 40, 18, 32],
    "r4i": [1, 64, 9, 16],
}
RENAME = {"src": "input_1", "r1i": "r1", "r2i": "r2", "r3i": "r3", "r4i": "r4"}


def fix_dims(model):
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if inp.name == "src":
            for d, v in zip(dims, [1, 3, H, W]):
                d.ClearField("dim_param")
                d.dim_value = v
        elif inp.name in STATE_SHAPES:
            for d, v in zip(dims, STATE_SHAPES[inp.name]):
                d.ClearField("dim_param")
                d.dim_value = v


def freeze_ratio(model):
    g = model.graph
    keep = [i for i in g.input if i.name != "downsample_ratio"]
    del g.input[:]
    g.input.extend(keep)
    g.initializer.append(
        helper.make_tensor("downsample_ratio", TensorProto.FLOAT, [1], [1.0])
    )


def ort_fold(path_in, path_out):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = path_out
    ort.InferenceSession(path_in, so, providers=["CPUExecutionProvider"])


def rename_inputs(model):
    g = model.graph
    for inp in g.input:
        if inp.name in RENAME:
            new = RENAME[inp.name]
            for n in g.node:
                for i, x in enumerate(n.input):
                    if x == inp.name:
                        n.input[i] = new
            inp.name = new


def seeded(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.rand(*shape).astype(np.float32)


def main():
    model = onnx.load(SRC)
    fix_dims(model)
    freeze_ratio(model)
    tmp = DST + ".tmp.onnx"
    onnx.save(model, tmp)

    ort_fold(tmp, DST)

    model = onnx.load(DST)
    rename_inputs(model)
    # ORT가 얹는 비표준 opset 선언 제거 — 노드는 전부 표준 도메인이라 선언만 정리
    # (ai-convert가 "커스텀 opset 도메인"으로 거부하는 원인)
    keep_ops = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
    del model.opset_import[:]
    model.opset_import.extend(keep_ops)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, DST)

    n_src = len(onnx.load(SRC).graph.node)
    n_dst = len(model.graph.node)
    print(f"nodes: {n_src} -> {n_dst}")

    # ── 게이트: 원본 vs 고정본 출력 일치 ──
    src_in = seeded([1, 3, H, W], 7)
    states = {k: np.zeros(v, np.float32) for k, v in STATE_SHAPES.items()}
    s0 = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    ref = s0.run(
        None,
        {"src": src_in, "downsample_ratio": np.array([1.0], np.float32), **states},
    )
    s1 = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    got = s1.run(
        None,
        {"input_1": src_in, **{RENAME[k]: v for k, v in states.items()}},
    )
    names = [o.name for o in s0.get_outputs()]
    worst = 0.0
    for name, a, b in zip(names, ref, got):
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print(f"  {name}: max|diff| {d:.3e}")
    if worst > 1e-5:
        print("FAIL: 출력 불일치")
        sys.exit(1)
    print(f"PASS (worst {worst:.3e}) → {DST}")


if __name__ == "__main__":
    main()
