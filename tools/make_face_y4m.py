#!/usr/bin/env python3
"""실얼굴 픽스처(RGB raw) → y4m 가짜 카메라 파일.

헤드리스 크로미움의 --use-file-for-fake-video-capture(run_web.mjs --video=)에
물려 **얼굴이 있는** 카메라로 스모크/감사를 돌릴 때 쓴다 (기본 합성 패턴 카메라는
무얼굴이라 lm/gaze/bs 경로가 잠들어 검증이 약해진다). 전 세션에서 스크래치에
만들었다 유실된 것을 도구화한 것.

사용: python3 tools/make_face_y4m.py [-o out.y4m] [--frames 90]
입력: tests/data/frame_256x144.rgb (256×144 u8 RGB 타이트)
출력: YUV4MPEG2 C420 (BT.601 full-range — 엔진 yuv.rs와 같은 계수)
"""

import argparse
import os

import numpy as np

W, H = 256, 144
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "tests", "data", "frame_256x144.rgb")


def rgb_to_i420(rgb: np.ndarray):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    # 2×2 평균 서브샘플
    u8 = np.clip(u.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)), 0, 255).astype(np.uint8)
    v8 = np.clip(v.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)), 0, 255).astype(np.uint8)
    return y8, u8, v8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "..", "web", "models", "mediapipe", "face_256x144.y4m"))
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()

    rgb = np.fromfile(SRC, dtype=np.uint8)
    assert rgb.size == W * H * 3, f"픽스처 크기 불일치: {rgb.size}"
    y8, u8, v8 = rgb_to_i420(rgb.reshape(H, W, 3))
    frame = b"FRAME\n" + y8.tobytes() + u8.tobytes() + v8.tobytes()
    with open(args.o, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420\n".encode())
        for _ in range(args.frames):
            f.write(frame)
    print(f"{args.o}: {args.frames}프레임 {os.path.getsize(args.o)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
