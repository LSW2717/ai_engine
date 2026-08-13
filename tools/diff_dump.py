#!/usr/bin/env python3
"""층별 발산 추적 — CpuExec 덤프(dump_all.rs)와 ORT 중간값(--intermediates)을
이름으로 매칭해 tid 순서대로 오차를 출력한다. 첫 발산 지점이 버그 자리다.

사용: /usr/bin/python3 tools/diff_dump.py <ort_dump_dir> <cpu_dump_dir>

ORT는 NCHW, 엔진은 NHWC 평탄이라 4D 중간텐서는 shape을 모른 채 평탄 비교가
불가 — 대신 **정렬 후 비교**(같은 원소 집합인지)와 통계(mean/std)로 본다.
집합이 같으면 값은 맞고 순서 문제만 남는 것.
"""

import os
import re
import sys

import numpy as np

ort_dir, cpu_dir = sys.argv[1], sys.argv[2]

# act 융합 보정: 우리 conv 출력엔 act가 융합돼 있다 — ORT 쪽은 같은 이름이
# 융합 전 값이므로, 그 텐서를 소비하는 Clip/Relu "노드의 출력"과 비교해야 한다.
# (res 융합된 Add도 동일 — Add 출력으로 매핑)
act_map = {}
if len(sys.argv) > 3:
    import onnx

    m = onnx.load(sys.argv[3])
    for n in m.graph.node:
        if n.op_type in ("Clip", "Relu", "PRelu", "Sigmoid", "Add") and n.input:
            act_map[n.input[0]] = (n.op_type, n.output[0])


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


ort_files = {}
for f in os.listdir(ort_dir):
    if f.startswith("out__") and f.endswith(".f32"):
        ort_files[f[5:-4]] = os.path.join(ort_dir, f)

rows = []
for f in sorted(os.listdir(cpu_dir)):
    m = re.match(r"(\d+)__(.*)\.f32$", f)
    if not m:
        continue
    tid, name = int(m.group(1)), m.group(2)
    if name not in ort_files:
        continue
    # 원명이 act에 먹혔으면 act 출력과 비교 (여러 단 걸릴 수 있어 반복)
    src = name
    for orig, (op, out) in list(act_map.items()):
        if safe(orig) == name and safe(out) in ort_files:
            name = f"{name} [{op}후]"
            src = safe(out)
            break
    a = np.fromfile(ort_files[src], dtype=np.float32)
    b = np.fromfile(os.path.join(cpu_dir, f), dtype=np.float32)
    if a.size != b.size:
        rows.append((tid, name, f"길이 불일치 {a.size} vs {b.size}", 1e9))
        continue
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1.0)
    sa, sb = np.sort(a), np.sort(b)
    srel = float(np.max(np.abs(sa - sb) / np.maximum(np.abs(sa), 1.0)))
    rows.append((tid, name, f"flat {rel.max():.2e} sorted {srel:.2e} (n={a.size})", rel.max()))

first_bad = None
for tid, name, msg, err in rows:
    mark = ""
    if err > 1e-3 and first_bad is None:
        first_bad = (tid, name)
        mark = "  ← 첫 발산"
    print(f"{tid:4} {msg}  {name[-60:]}{mark}")
if first_bad:
    print(f"\n첫 발산: tid {first_bad[0]} {first_bad[1]}")
else:
    print("\n발산 없음")
