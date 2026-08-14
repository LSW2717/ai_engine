#!/usr/bin/env python3
"""fastenhancer wav2wav(스트리밍 1-hop) ONNX → spec2spec 서브그래프 추출.

공개 fastenhancer_b_{16k,48k}.onnx는 이미 1-hop 스트리밍(wav_in [1,hop] + 캐시
5개)이고 STFT/iSTFT(DFT op)가 모델 안에 있다. 우리 엔진은 DFT가 없으므로
vcxrust_ai vcx-noise와 같은 경계로 자른다 — 단, 더 좁게:

  입력: mul_1(압축 복소 스펙 [1,F,1,2]) + GRU 캐시 3개([Fr,C])
  출력: convolution(복소 마스크 [1,2,F]) + GRU 캐시 3개([1,Fr,C])

전/후처리(프레이밍·Hann·rfft·압축 X·|X|^(α−1)·복소곱·역압축·zero-pad c2c
irfft·window_istft·OLA)는 Rust(ai-tasks features/audio)가 맡는다.
--verify가 그 수학을 numpy로 재현해 원본 wav2wav와 1-hop 비트 수준 대조한다 —
여기서 PASS한 식이 Rust 이식의 단일 기준이다.

usage:
  /usr/bin/python3 tools/prep_fastenhancer.py \
      --src <v-ai>/assets/models/fastenhancer_b_48k.onnx \
      -o models/fastenhancer/fe48_spec2spec.onnx --verify
"""
import argparse
import collections

import numpy as np
import onnx
from onnx.utils import extract_model


def tensor_shapes(model):
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    return {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(g.value_info) + list(g.input) + list(g.output)
    }


def discover_boundaries(model):
    """구조 휴리스틱으로 절단 텐서를 찾는다 (16k/48k 동일 아키텍처).

    - 압축 스펙: forward DFT → Unsqueeze → Slice → Mul (ReduceL2 형제) 의 출력
    - 마스크: 유일한 ConvTranspose의 출력
    - 캐시 입력: SplitToSequence(cache_in_k) → SequenceAt 출력 (배치 squeeze 항등)
    - 캐시 출력: cache_out_k ← Unsqueeze ← Squeeze ← X 의 X (항등 왕복 소거)
    """
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)

    fwd_dft = next(
        n for n in g.node
        if n.op_type == "DFT"
        and not any(a.name == "inverse" and a.i == 1 for a in n.attribute)
    )
    t = fwd_dft.output[0]
    unsq = next(c for c in cons[t] if c.op_type == "Unsqueeze")
    slc = next(c for c in cons[unsq.output[0]] if c.op_type == "Slice")
    mul = next(c for c in cons[slc.output[0]] if c.op_type == "Mul")
    compressed = mul.output[0]

    deconv = next(n for n in g.node if n.op_type == "ConvTranspose")
    mask = deconv.output[0]

    cache_ins = []
    cache_outs = []
    for k in (2, 3, 4):
        sts = next(
            n for n in g.node
            if n.op_type == "SplitToSequence" and n.input[0] == f"cache_in_{k}"
        )
        seq_at = next(c for c in cons[sts.output[0]] if c.op_type == "SequenceAt")
        cache_ins.append(seq_at.output[0])
        unsq_out = prod[f"cache_out_{k}"]
        assert unsq_out.op_type == "Unsqueeze", unsq_out.op_type
        sq = prod[unsq_out.input[0]]
        assert sq.op_type == "Squeeze", sq.op_type
        cache_outs.append(sq.input[0])
    return compressed, mask, cache_ins, cache_outs


def strip_sequence_identity(model):
    """SplitToSequence(axis0, size1) + SequenceAt(0) = 배치 squeeze 항등 —
    Squeeze(axes=[0])로 치환한다 (컨버터에 시퀀스 op을 들이지 않기 위해).
    GRU layer 루프의 잔재라 전부 이 패턴이다 — 아니면 assert로 잡는다."""
    g = model.graph
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    remove = []
    axes_init = onnx.numpy_helper.from_array(
        np.array([0], np.int64), name="fe_squeeze_axis0"
    )
    g.initializer.append(axes_init)
    for n in list(g.node):
        if n.op_type != "SplitToSequence":
            continue
        users = cons[n.output[0]]
        assert all(u.op_type == "SequenceAt" for u in users), users
        for u in users:
            sq = onnx.helper.make_node(
                "Squeeze", [n.input[0], "fe_squeeze_axis0"], list(u.output),
                name=u.name + "_sq",
            )
            g.node.append(sq)
            remove.append(u)
        remove.append(n)
    for n in remove:
        g.node.remove(n)
    # 토폴로지 순서 복구 (뒤에 붙인 Squeeze를 소비자 앞으로)
    order = {o: i for i, n in enumerate(g.node) for o in n.output}
    nodes = sorted(g.node, key=lambda n: max((order.get(i, -1) for i in n.input), default=-1))
    # 안정적 위상 정렬 재구성
    produced = {i.name for i in g.initializer} | {i.name for i in g.input}
    for n in g.node:
        for i in n.input:
            if not i:
                produced.add(i)
    remaining = list(g.node)
    ordered = []
    while remaining:
        progressed = False
        for n in list(remaining):
            if all((not i) or i in produced for i in n.input):
                ordered.append(n)
                produced.update(n.output)
                remaining.remove(n)
                progressed = True
        assert progressed, "위상 정렬 실패"
    del g.node[:]
    g.node.extend(ordered)


# ── --verify: 전/후처리 numpy 재현 (Rust 이식의 기준 구현) ──────────────────

def hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def window_istft(n_fft, hop):
    """vcx-noise stft.rs와 동일: window / sum-of-squared-windows (COLA)"""
    w = hann_periodic(n_fft)
    k = (n_fft + hop - 1) // hop
    l = hop * (2 * k - 1) + (n_fft - hop)
    ws = np.zeros(l)
    for j in range(2 * k - 1):
        ws[j * hop:j * hop + n_fft] += w * w
    off = (k - 1) * hop
    s = ws[off:off + n_fft]
    return np.where(s > 1e-12, w / s, 0.0)


def reference_hop(sub_path, wav_in, caches, n_fft, hop, alpha=0.5, beta=2.0,
                  clip_min=None, state=None):
    """서브그래프 + numpy 전/후처리로 1-hop 실행 — 원본 wav2wav 등가여야 한다.

    ⚠ alpha/beta/clip_min은 그래프 상수에서 읽어 넘긴다 (기본값은 자리표시).
    state: {'in_cache': [n_fft-hop], 'ola': [n_fft-hop]} (없으면 0)
    """
    import onnxruntime as ort
    bins_used = n_fft // 2  # 마지막 bin(Nyquist)은 슬라이스로 버려진다
    st = state or {
        "in_cache": np.zeros(n_fft - hop, np.float32),
        "ola": np.zeros(n_fft - hop, np.float32),
    }
    frame = np.concatenate([st["in_cache"], wav_in]).astype(np.float32)
    st["in_cache"] = frame[hop:].copy()
    spec = np.fft.rfft(frame * hann_periodic(n_fft))  # [bins]
    x = spec[:bins_used]  # Nyquist 드랍 (slice_4)
    # 압축: mul_1 = X · clip(|X|)^(alpha-1)
    mag = np.abs(x)
    magc = np.maximum(mag, clip_min) if clip_min is not None else mag
    comp = x * (magc ** (alpha - 1.0))
    mul_1 = np.stack([comp.real, comp.imag], -1).astype(np.float32)[None, :, None, :]

    sess = ort.InferenceSession(sub_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    feeds = {in_names[0]: mul_1}
    for name, c in zip(in_names[1:], caches):
        feeds[name] = c
    outs = sess.run(None, feeds)
    mask = outs[0][0]  # [2, F]
    new_caches = outs[1:]

    m = mask[0] + 1j * mask[1]
    y = m * comp  # 복소 마스크 곱
    # 역압축: Y · |Y|^(beta-1)
    ymag = np.abs(y)
    y = y * np.where(ymag > 0, ymag ** (beta - 1.0), 0.0)
    # Nyquist 0 복원 + zero-pad c2c inverse의 실수부 (원본 그래프의 irfft 트릭)
    full = np.zeros(n_fft, np.complex64)
    full[:bins_used] = y
    ifft = np.fft.ifft(full).real * 2.0  # zero-pad c2c: conj 확장 irfft의 절반 스케일
    ifft -= np.real(full[0]).real / n_fft  # DC 이중 계상 보정
    out_frame = ifft * window_istft(n_fft, hop)
    out_frame[: n_fft - hop] += st["ola"]
    st["ola"] = out_frame[hop:].copy()
    return out_frame[:hop].astype(np.float32), new_caches, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--export", metavar="DIR",
                    help="Rust 미니 실행기용 graph.json + weights.bin + 오라클 덤프")
    args = ap.parse_args()

    model = onnx.load(args.src)
    compressed, mask, cache_ins, cache_outs = discover_boundaries(model)
    print(f"절단: in=[{compressed}, {cache_ins}] out=[{mask}, {cache_outs}]")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    extract_model(args.src, args.out, [compressed] + cache_ins, [mask] + cache_outs)
    sub = onnx.load(args.out)
    strip_sequence_identity(sub)
    onnx.save(sub, args.out)
    ops = collections.Counter(n.op_type for n in sub.graph.node)
    print("서브그래프 ops:", dict(ops.most_common(40)))

    if args.verify:
        verify(args.src, args.out, model)
    if args.export:
        export_runtime(args.out, args.export, pre_post_consts(model))


def pre_post_consts(model):
    """전/후처리 상수 — Rust가 하드코딩하지 않게 manifest로 넘긴다"""
    g = model.graph
    hop = next(i for i in g.input if i.name == "wav_in").type.tensor_type.shape.dim[1].dim_value
    pows, clips = [], []
    for n in g.node:
        if n.op_type == "Pow":
            v = graph_const(model, n.input[1])
            if v is not None:
                pows.append(float(v))
        if n.op_type == "Clip" and len(n.input) > 1 and n.input[1]:
            v = graph_const(model, n.input[1])
            if v is not None:
                clips.append(float(v))
    return {
        "n_fft": hop * 2,
        "hop": hop,
        "alpha": pows[0] + 1.0,
        "beta": pows[1] + 1.0,
        "clip_min": clips[0] if clips else 0.0,
    }


def export_runtime(sub_path, out_dir, pre_post):
    """서브그래프 → Rust 미니 실행기 포맷:
      graph.json  — 노드(op/attr)·f32 상수(오프셋)·int 상수(인라인)·입출력
      weights.bin — f32 상수 블롭 (json의 offset/len이 가리킴)
      oracle_*.bin — 시드 입력과 ORT 출력 (Rust 게이트가 1e-5로 대조)
    """
    import json, os
    import onnxruntime as ort

    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(sub_path)
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph

    shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(g.value_info) + list(g.input) + list(g.output)
    }

    blob = bytearray()
    consts_f32 = []
    consts_int = {}
    def add_const(name, arr):
        if arr.dtype in (np.int64, np.int32):
            consts_int[name] = [int(v) for v in np.atleast_1d(arr).ravel()]
        else:
            a = np.ascontiguousarray(arr, np.float32)
            off = len(blob)
            blob.extend(a.tobytes())
            consts_f32.append({"name": name, "shape": list(a.shape),
                               "offset": off // 4, "len": int(a.size)})
    for i in g.initializer:
        add_const(i.name, onnx.numpy_helper.to_array(i))
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    add_const(n.output[0], onnx.numpy_helper.to_array(a.t))

    nodes = []
    for n in g.node:
        if n.op_type == "Constant":
            continue
        attrs = {}
        for a in n.attribute:
            v = onnx.helper.get_attribute_value(a)
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, (list, tuple)):
                v = [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in v]
            elif isinstance(v, (int, np.integer)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v)
            else:
                continue  # graph/텐서 attr 없음 (Constant는 위에서 처리)
            attrs[a.name] = v
        nodes.append({"op": n.op_type, "name": n.name,
                      "inputs": list(n.input), "outputs": list(n.output),
                      "attrs": attrs})

    manifest = {
        "inputs": [{"name": i.name, "shape": shapes[i.name]} for i in g.input],
        "outputs": [o.name for o in g.output],
        "shapes": {k: v for k, v in shapes.items()},
        "nodes": nodes,
        "consts_f32": consts_f32,
        "consts_int": consts_int,
        "pre_post": pre_post,
    }
    with open(f"{out_dir}/graph.json", "w") as f:
        json.dump(manifest, f)
    with open(f"{out_dir}/weights.bin", "wb") as f:
        f.write(bytes(blob))

    # 오라클: 시드 입력 → ORT 출력 (f32 raw, [입력들..., 출력들...] 순서 연결)
    rng = np.random.default_rng(11)
    sess = ort.InferenceSession(sub_path, providers=["CPUExecutionProvider"])
    feeds = {}
    order = []
    for i in g.input:
        arr = (rng.standard_normal(shapes[i.name]) * 0.5).astype(np.float32)
        feeds[i.name] = arr
        order.append(arr)
    outs = sess.run(None, feeds)
    with open(f"{out_dir}/oracle_io.bin", "wb") as f:
        for a in order + list(outs):
            f.write(np.ascontiguousarray(a, np.float32).tobytes())
    io_meta = {
        "inputs": [{"name": i.name, "shape": shapes[i.name]} for i in g.input],
        "outputs": [{"name": o.name, "shape": shapes[o.name]} for o in g.output],
    }
    with open(f"{out_dir}/oracle_io.json", "w") as f:
        json.dump(io_meta, f)
    print(f"export: {out_dir} (nodes {len(nodes)}, f32 상수 {len(consts_f32)}, "
          f"blob {len(blob)//1024}KB)")


def graph_const(model, name):
    g = model.graph
    for i in g.initializer:
        if i.name == name:
            return onnx.numpy_helper.to_array(i)
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return onnx.numpy_helper.to_array(a.t)
    return None


def verify(src, sub_path, model):
    """원본 wav2wav 1-hop vs (numpy 전/후처리 + 서브그래프) — 3 hop 연쇄 대조"""
    import onnxruntime as ort

    g = model.graph
    hop = next(i for i in g.input if i.name == "wav_in").type.tensor_type.shape.dim[1].dim_value
    n_fft = hop * 2
    cache_shape = [
        [d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in g.input if i.name == "cache_in_2"
    ][0]

    # 압축/역압축 상수 발굴: Pow 지수 + Clip 최소값
    pows, clips = [], []
    for n in g.node:
        if n.op_type == "Pow":
            v = graph_const(model, n.input[1])
            if v is not None:
                pows.append(float(v))
        if n.op_type == "Clip" and len(n.input) > 1 and n.input[1]:
            v = graph_const(model, n.input[1])
            if v is not None:
                clips.append(float(v))
    print(f"발굴 상수: pow 지수 {pows}, clip 최소 {clips}")
    # pow[0] = alpha-1 (압축), pow[1] = beta-1 (역압축)
    alpha = pows[0] + 1.0
    beta = pows[1] + 1.0
    clip_min = clips[0] if clips else None

    rng = np.random.default_rng(7)
    wav = (rng.standard_normal(hop * 3) * 0.1).astype(np.float32)

    full = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    fcaches = {
        "cache_in_0": np.zeros((1, hop), np.float32),
        "cache_in_1": np.zeros((1, hop), np.float32),
        "cache_in_2": np.zeros([1] + cache_shape[1:], np.float32),
        "cache_in_3": np.zeros([1] + cache_shape[1:], np.float32),
        "cache_in_4": np.zeros([1] + cache_shape[1:], np.float32),
    }
    scaches = [np.zeros(cache_shape[1:], np.float32) for _ in range(3)]
    state = None
    worst = 0.0
    for h in range(3):
        seg = wav[h * hop:(h + 1) * hop]
        fo = full.run(None, {"wav_in": seg[None, :], **fcaches})
        ref_out = fo[0][0]
        for k in range(5):
            fcaches[f"cache_in_{k}"] = fo[1 + k]
        ours, new_caches, state = reference_hop(
            sub_path, seg, scaches, n_fft, hop, alpha, beta, clip_min, state
        )
        # 서브그래프 캐시 rank 정합 (출력이 [1,Fr,C]일 수 있음)
        scaches = [c.reshape(cache_shape[1:]) for c in new_caches]
        d = float(np.abs(ours - ref_out).max())
        worst = max(worst, d)
        print(f"hop {h}: max|Δwav| = {d:.3e}")
    print("VERIFY", "PASS" if worst < 1e-4 else f"FAIL (max {worst:.3e})")


if __name__ == "__main__":
    main()
