//! 원소별 연산 — Binary(tensor/scalar/cvec)/Act/Mix.
//!
//! 입력은 뷰(alias 무복사), 출력은 항상 밀집 슬롯. 전체 프레임 비용에서
//! 비중이 작아 스칼라 루프로 둔다 — 프로파일에 뜨면 그때 벡터화.

use ai_core::ops::BinaryOp;
use ai_core::Activation;

use crate::view::View;

/// out[p,ch] = act(a[p,ch] ∘ b[p,ch])
pub fn binary_tensor(op: BinaryOp, a: View, b: View, px: usize, act: Activation, out: &mut [f32]) {
    debug_assert_eq!(a.c, b.c);
    let c = a.c;
    for p in 0..px {
        let (ab, bb) = (a.base(p), b.base(p));
        for ch in 0..c {
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
    for p in 0..px {
        let ab = a.base(p);
        for ch in 0..c {
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
    for p in 0..px {
        let ab = a.base(p);
        for ch in 0..c {
            out[p * c + ch] = act.apply(op.apply(a.data[ab + ch], vec[ch]));
        }
    }
}

/// 단독 활성화
pub fn act(a: View, px: usize, f: Activation, out: &mut [f32]) {
    let c = a.c;
    for p in 0..px {
        let ab = a.base(p);
        for ch in 0..c {
            out[p * c + ch] = f.apply(a.data[ab + ch]);
        }
    }
}

/// GRU mix: out = a + z*(b-a)
pub fn mix(z: View, a: View, b: View, px: usize, out: &mut [f32]) {
    debug_assert!(a.c == b.c && a.c == z.c);
    let c = a.c;
    for p in 0..px {
        let (zb, ab, bb) = (z.base(p), a.base(p), b.base(p));
        for ch in 0..c {
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
