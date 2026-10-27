#!/usr/bin/env python3
"""Silero VAD(v6.2, 16k) ONNX → 정적 1-청크 그래프 재구성 + 검증 + Rust 실행기 export.

공개 silero_vad_16k_op15.onnx는 배치·길이 가변을 위한 Shape/Gather/If/ConstantOfShape와
PyTorch LSTM→ONNX LSTM 게이트 재배열(Slice/Concat)이 섞여 있다. 모델 자체는 고정
구조라 같은 가중치로 **정적 그래프를 새로 짠다** — 우리 오디오 미니 실행기 op만 쓴다:

  input [1,576] (앞 청크 꼬리 64 + 이번 512 @16k)
    → Pad(reflect, 오른쪽 64) → Conv(STFT 기저 258×256, stride 128) → [1,258,4]
    → |X| = sqrt(re² + im²) [1,129,4]
    → (Conv k3 + Relu) ×4 (stride 1·2·2·1) → [1,128,1]
    → LSTMCell(128) — Gemm 2개 + Split(i,f,g,o) + σ/tanh
    → Relu → Conv 1×1 → Sigmoid → prob [1,1]
  상태 h, c [1,128] 는 입출력으로 노출 (Rust 가 청크마다 이어 준다).

--verify 는 원본과 재구성 그래프를 같은 입력열(실음성 + 난수, 상태 연쇄)로 돌려
prob·상태를 대조한다 — PASS 한 그래프만 export 한다.

usage:
  /usr/bin/python3 tools/prep_silero.py \
      --src <v-ai>/assets/models/silero_vad_16k_op15.onnx \
      -o models/silero/silero16_static.onnx --verify --export models/silero/vad16 \
      [--fixture-wav speech_16k.wav]
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prep_fastenhancer import export_runtime

CHUNK = 512
CONTEXT = 64
HIDDEN = 128


def weights(model):
    w = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    need = [
        "model.stft.forward_basis_buffer",
        *[f"model.encoder.{k}.reparam_conv.{p}" for k in range(4) for p in ("weight", "bias")],
        "model.decoder.rnn.weight_ih", "model.decoder.rnn.weight_hh",
        "model.decoder.rnn.bias_ih", "model.decoder.rnn.bias_hh",
        "model.decoder.decoder.2.weight", "model.decoder.decoder.2.bias",
    ]
    missing = [n for n in need if n not in w]
    if missing:
        raise SystemExit(f"가중치 없음: {missing}")
    return w


def build_static(w):
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def node(op, ins, outs=None, **attrs):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]
        nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0] if len(outs) == 1 else outs

    basis = w["model.stft.forward_basis_buffer"]  # [2·F, 1, n_fft]
    two_f, _, n_fft = basis.shape
    bins = two_f // 2
    hop = n_fft // 2
    const("basis", basis.astype(np.float32))

    x = node("Pad", ["input", const("pads", np.array([0, 0, 0, CONTEXT], np.int64))],
             mode="reflect")
    x = node("Unsqueeze", [x, const("ax1", np.array([1], np.int64))])
    spec = node("Conv", [x, "basis"], kernel_shape=[n_fft], strides=[hop], pads=[0, 0])
    re = node("Slice", [spec, const("s0", np.array([0], np.int64)),
                        const("s1", np.array([bins], np.int64)), "ax1"])
    im = node("Slice", [spec, "s1", const("s2", np.array([two_f], np.int64)), "ax1"])
    mag = node("Sqrt", [node("Add", [node("Mul", [re, re]), node("Mul", [im, im])])])

    x = mag
    for k, stride in enumerate((1, 2, 2, 1)):
        wk = const(f"enc{k}_w", w[f"model.encoder.{k}.reparam_conv.weight"])
        bk = const(f"enc{k}_b", w[f"model.encoder.{k}.reparam_conv.bias"])
        x = node("Relu", [node("Conv", [x, wk, bk], kernel_shape=[3], strides=[stride],
                               pads=[1, 1])])
    x = node("Reshape", [x, const("shape_bh", np.array([1, HIDDEN], np.int64))])

    gi = node("Gemm", [x, const("w_ih", w["model.decoder.rnn.weight_ih"]),
                       const("b_ih", w["model.decoder.rnn.bias_ih"])], transB=1)
    gh = node("Gemm", ["h", const("w_hh", w["model.decoder.rnn.weight_hh"]),
                       const("b_hh", w["model.decoder.rnn.bias_hh"])], transB=1)
    gates = node("Add", [gi, gh])
    i, f, g, o = node("Split", [gates, const("split4", np.array([HIDDEN] * 4, np.int64))],
                      ["gate_i", "gate_f", "gate_g", "gate_o"], axis=1)
    i, f, o = (node("Sigmoid", [t]) for t in (i, f, o))
    g = node("Tanh", [g])
    node("Add", [node("Mul", [f, "c"]), node("Mul", [i, g])], ["c_out"])
    node("Mul", [o, node("Tanh", ["c_out"])], ["h_out"])

    y = node("Relu", ["h_out"])
    y = node("Reshape", [y, const("shape_bh1", np.array([1, HIDDEN, 1], np.int64))])
    y = node("Conv", [y, const("dec_w", w["model.decoder.decoder.2.weight"]),
                      const("dec_b", w["model.decoder.decoder.2.bias"])],
             kernel_shape=[1], strides=[1], pads=[0, 0])
    y = node("Sigmoid", [y])
    node("Reshape", [y, const("shape_11", np.array([1, 1], np.int64))], ["prob"])

    f32 = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes, "silero_vad16_static",
        [helper.make_tensor_value_info("input", f32, [1, CONTEXT + CHUNK]),
         helper.make_tensor_value_info("h", f32, [1, HIDDEN]),
         helper.make_tensor_value_info("c", f32, [1, HIDDEN])],
        [helper.make_tensor_value_info("prob", f32, [1, 1]),
         helper.make_tensor_value_info("h_out", f32, [1, HIDDEN]),
         helper.make_tensor_value_info("c_out", f32, [1, HIDDEN])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def run_original(sess, audio):
    """원본 + 공식 래퍼(OnnxWrapper)와 같은 루프 — 64 꼬리 문맥, 상태 연쇄."""
    state = np.zeros((2, 1, HIDDEN), np.float32)
    ctx = np.zeros((1, CONTEXT), np.float32)
    probs = []
    for s in range(0, len(audio) - CHUNK + 1, CHUNK):
        inp = np.concatenate([ctx, audio[None, s:s + CHUNK]], 1).astype(np.float32)
        p, state = sess.run(None, {"input": inp, "state": state,
                                   "sr": np.array(16000, np.int64)})
        ctx = inp[:, -CONTEXT:]
        probs.append(float(p.reshape(-1)[0]))
    return np.array(probs, np.float32), state


def run_static(sess, audio):
    h = np.zeros((1, HIDDEN), np.float32)
    c = np.zeros((1, HIDDEN), np.float32)
    ctx = np.zeros((1, CONTEXT), np.float32)
    probs = []
    for s in range(0, len(audio) - CHUNK + 1, CHUNK):
        inp = np.concatenate([ctx, audio[None, s:s + CHUNK]], 1).astype(np.float32)
        p, h, c = sess.run(None, {"input": inp, "h": h, "c": c})
        ctx = inp[:, -CONTEXT:]
        probs.append(float(p.reshape(-1)[0]))
    return np.array(probs, np.float32), np.stack([h, c])


def read_wav16(path):
    import wave
    with wave.open(path) as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16k mono s16 wav 이어야 한다")
        return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768


def verify(src, static_path, speech):
    import onnxruntime as ort
    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(static_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    cases = {"noise": (rng.standard_normal(16000 * 3) * 0.1).astype(np.float32)}
    if speech is not None:
        cases["speech"] = speech
    worst = 0.0
    for name, audio in cases.items():
        pa, sa = run_original(a, audio)
        pb, sb = run_static(b, audio)
        dp = float(np.abs(pa - pb).max())
        # 상태(c)는 청크마다 누적돼 크기가 커진다 — 절대오차가 아니라 상대오차로 본다
        # (구현 차이가 아닌 f32 누적 순서 차이; prob 은 1e-7 대로 일치).
        ds = float(np.abs(sa - sb).max() / max(1.0, np.abs(sa).max()))
        worst = max(worst, dp, ds)
        print(f"  {name}: chunks {len(pa)}, prob mean {pa.mean():.3f}, "
              f"max|Δprob| {dp:.2e}, max|Δstate|/max|state| {ds:.2e} "
              f"(max|state| {np.abs(sa).max():.1f})")
    ok = worst < 1e-5
    print("VERIFY", "PASS" if ok else f"FAIL (max {worst:.3e})")
    return ok


def write_fixture(src, speech, out_dir):
    """Rust e2e 게이트용 — 16k 입력과 원본 모델 prob 열 (f32 raw)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    probs, _ = run_original(sess, speech)
    speech.astype(np.float32).tofile(f"{out_dir}/in_16k.f32")
    probs.tofile(f"{out_dir}/ref_probs_16k.f32")
    print(f"fixture: {out_dir}/in_16k.f32 ({len(speech)} 샘플), "
          f"ref_probs_16k.f32 ({len(probs)} 청크, mean {probs.mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--export", metavar="DIR",
                    help="Rust 미니 실행기용 graph.json + weights.bin + 오라클 덤프")
    ap.add_argument("--fixture-wav", help="16k mono 음성 wav — 검증 입력 + e2e fixture")
    args = ap.parse_args()

    src = onnx.load(args.src)
    static = build_static(weights(src))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx.save(static, args.out)
    print(f"정적 그래프: {args.out} (nodes {len(static.graph.node)})")

    speech = read_wav16(args.fixture_wav) if args.fixture_wav else None
    if args.verify and not verify(args.src, args.out, speech):
        raise SystemExit(1)
    if args.export:
        export_runtime(args.out, args.export, None)
        if speech is not None:
            write_fixture(args.src, speech, os.path.dirname(args.out))


if __name__ == "__main__":
    main()
