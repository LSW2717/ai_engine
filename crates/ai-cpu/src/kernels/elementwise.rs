//! 원소별 연산 — Binary(tensor/scalar/cvec)/Act/Mix.
//!
//! 입력은 뷰(alias 무복사), 출력은 항상 밀집 슬롯. 채널 4배수 구간은 벡터,
//! 꼬리는 스칼라 (뷰의 마지막 4레인 로드가 백킹 이웃 채널을 읽으면 안 되므로
//! 벡터 로드는 c/4*4까지 — dw와 같은 규약).

use ai_core::ops::BinaryOp;
use ai_core::Activation;

use crate::kernels::apply4;
use crate::simd::F32x4;
use crate::view::View;

/// BinaryOp 4레인 적용
#[inline(always)]
fn bin4(op: BinaryOp, a: F32x4, b: F32x4) -> F32x4 {
    match op {
        BinaryOp::Add => a.add(b),
        BinaryOp::Mul => a.mul(b),
        BinaryOp::Sub => a.fma(b, F32x4::splat(-1.0)), // a + b*(-1)
        // max(a,0) + b*min(a,0) — 분기 없는 prelu (b = 채널 slope)
        BinaryOp::Prelu => {
            let z = F32x4::splat(0.0);
            a.max(z).fma(a.min(z), b)
        }
    }
}

/// out[p,ch] = act(a[p,ch] ∘ b[p,ch])
pub fn binary_tensor(op: BinaryOp, a: View, b: View, px: usize, act: Activation, out: &mut [f32]) {
    debug_assert_eq!(a.c, b.c);
    let c = a.c;
    let cv = c / 4 * 4;
    if a.c_off == 0 && a.stride == c && b.c_off == 0 && b.stride == c && cv == c {
        let n = px * c;
        let mut i = 0usize;
        while i < n {
            apply4(act, bin4(op, F32x4::load(a.data, i), F32x4::load(b.data, i))).store(out, i);
            i += 4;
        }
        return;
    }
    for p in 0..px {
        let (ab, bb) = (a.base(p), b.base(p));
        let mut cc = 0usize;
        while cc < cv {
            apply4(act, bin4(op, F32x4::load(a.data, ab + cc), F32x4::load(b.data, bb + cc)))
                .store(out, p * c + cc);
            cc += 4;
        }
        for ch in cv..c {
            out[p * c + ch] = act.apply(op.apply(a.data[ab + ch], b.data[bb + ch]));
        }
    }
}

/// out = act(a ∘ scalar) (first면 scalar ∘ a)
pub fn binary_scalar(
    op: BinaryOp,
    a: View,
    v: f32,
    first: bool,
    px: usize,
    act: Activation,
    out: &mut [f32],
) {
    let c = a.c;
    let cv = c / 4 * 4;
    let vv = F32x4::splat(v);
    for p in 0..px {
        let ab = a.base(p);
        let mut cc = 0usize;
        while cc < cv {
            let x = F32x4::load(a.data, ab + cc);
            let r = if first { bin4(op, vv, x) } else { bin4(op, x, vv) };
            apply4(act, r).store(out, p * c + cc);
            cc += 4;
        }
        for ch in cv..c {
            let x = a.data[ab + ch];
            let r = if first { op.apply(v, x) } else { op.apply(x, v) };
            out[p * c + ch] = act.apply(r);
        }
    }
}

/// out[p,ch] = act(a[p,ch] ∘ vec[ch]) — 채널 벡터(상수 또는 SE 게이트 출력)
pub fn binary_cvec(
    op: BinaryOp,
    a: View,
    vec: &[f32],
    px: usize,
    act: Activation,
    out: &mut [f32],
) {
    debug_assert_eq!(a.c, vec.len());
    let c = a.c;
    let cv = c / 4 * 4;
    // dense 뷰 + 4배수 채널: 플랫 루프 (픽셀당 인덱스 산술 제거 — PRelu가
    // 2 GF/s로 기던 원인)
    if a.c_off == 0 && a.stride == c && cv == c {
        let n = px * c;
        let mut i = 0usize;
        let mut cc = 0usize;
        while i < n {
            apply4(act, bin4(op, F32x4::load(a.data, i), F32x4::load(vec, cc)))
                .store(out, i);
            i += 4;
            cc += 4;
            if cc == c {
                cc = 0;
            }
        }
        return;
    }
    for p in 0..px {
        let ab = a.base(p);
        let mut cc = 0usize;
        while cc < cv {
            apply4(act, bin4(op, F32x4::load(a.data, ab + cc), F32x4::load(vec, cc)))
                .store(out, p * c + cc);
            cc += 4;
        }
        for ch in cv..c {
            out[p * c + ch] = act.apply(op.apply(a.data[ab + ch], vec[ch]));
        }
    }
}

/// 단독 활성화
pub fn act(a: View, px: usize, f: Activation, out: &mut [f32]) {
    let c = a.c;
    let cv = c / 4 * 4;
    if a.c_off == 0 && a.stride == c && cv == c {
        let n = px * c;
        let mut i = 0usize;
        while i < n {
            apply4(f, F32x4::load(a.data, i)).store(out, i);
            i += 4;
        }
        return;
    }
    for p in 0..px {
        let ab = a.base(p);
        let mut cc = 0usize;
        while cc < cv {
            apply4(f, F32x4::load(a.data, ab + cc)).store(out, p * c + cc);
            cc += 4;
        }
        for ch in cv..c {
            out[p * c + ch] = f.apply(a.data[ab + ch]);
        }
    }
}

/// GRU mix: out = a + z*(b-a)
pub fn mix(z: View, a: View, b: View, px: usize, out: &mut [f32]) {
    debug_assert!(a.c == b.c && a.c == z.c);
    let c = a.c;
    let cv = c / 4 * 4;
    let m1 = F32x4::splat(-1.0);
    for p in 0..px {
        let (zb, ab, bb) = (z.base(p), a.base(p), b.base(p));
        let mut cc = 0usize;
        while cc < cv {
            let av = F32x4::load(a.data, ab + cc);
            let bv = F32x4::load(b.data, bb + cc);
            let zv = F32x4::load(z.data, zb + cc);
            // a + z*(b-a) = a + z*b - z*a
            av.fma(zv, bv).fma(zv.mul(m1), av).store(out, p * c + cc);
            cc += 4;
        }
        for ch in cv..c {
            let (zv, av, bv) = (z.data[zb + ch], a.data[ab + ch], b.data[bb + ch]);
            out[p * c + ch] = av + zv * (bv - av);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::reference;
    use ai_core::rng::XorShift32;

    #[test]
    fn matches_reference() {
        let mut rng = XorShift32::new(1);
        let (px, c) = (12usize, 5usize);
        let a = rng.vec_f32(px * c);
        let b = rng.vec_f32(px * c);

        let want = reference::elementwise::binary(BinaryOp::Mul, &a, &b, Activation::Relu);
        let mut got = vec![0f32; px * c];
        binary_tensor(
            BinaryOp::Mul, View::dense(&a, c), View::dense(&b, c), px, Activation::Relu, &mut got,
        );
        assert_eq!(want, got);

        let want = reference::elementwise::binary_scalar(BinaryOp::Sub, &a, 1.0, true, Activation::None);
        binary_scalar(BinaryOp::Sub, View::dense(&a, c), 1.0, true, px, Activation::None, &mut got);
        assert_eq!(want, got);
    }

    #[test]
    fn cvec_and_mix_hand_computed() {
        // cvec mul: 채널 [2, 0.5]
        let a = vec![1.0, 4.0, 3.0, 8.0]; // 2px, c=2
        let mut got = vec![0f32; 4];
        binary_cvec(BinaryOp::Mul, View::dense(&a, 2), &[2.0, 0.5], 2, Activation::None, &mut got);
        assert_eq!(got, vec![2.0, 2.0, 6.0, 4.0]);

        // mix: z=0 → a, z=1 → b, z=0.5 → 중간
        let z = vec![0.0, 1.0, 0.5, 0.5];
        let av = vec![1.0, 1.0, 2.0, 4.0];
        let bv = vec![3.0, 3.0, 4.0, 8.0];
        mix(View::dense(&z, 2), View::dense(&av, 2), View::dense(&bv, 2), 2, &mut got);
        assert_eq!(got, vec![1.0, 3.0, 3.0, 6.0]);
    }

    /// 뷰 입력(채널 슬라이스) — 밀집 실체화와 동일해야 함
    #[test]
    fn strided_view_input() {
        let mut rng = XorShift32::new(2);
        let (px, big_c, c, off) = (6usize, 7usize, 3usize, 2usize);
        let big = rng.vec_f32(px * big_c);
        let b = rng.vec_f32(px * c);
        let mut dense = vec![0f32; px * c];
        for p in 0..px {
            dense[p * c..(p + 1) * c].copy_from_slice(&big[p * big_c + off..p * big_c + off + c]);
        }
        let want = reference::elementwise::binary(BinaryOp::Add, &dense, &b, Activation::Sigmoid);
        let mut got = vec![0f32; px * c];
        binary_tensor(
            BinaryOp::Add,
            View { data: &big, c_off: off, stride: big_c, c },
            View::dense(&b, c),
            px,
            Activation::Sigmoid,
            &mut got,
        );
        assert_eq!(want, got);
    }
}
