//! 원소별 연산 CPU 레퍼런스.

use crate::activation::Activation;
use crate::ops::BinaryOp;

/// tensor ∘ tensor, 이후 activation
pub fn binary(op: BinaryOp, a: &[f32], b: &[f32], act: Activation) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| act.apply(op.apply(*x, *y))).collect()
}

/// tensor ∘ scalar (scalar_first면 scalar ∘ tensor), 이후 activation
pub fn binary_scalar(
    op: BinaryOp,
    a: &[f32],
    scalar: f32,
    scalar_first: bool,
    act: Activation,
) -> Vec<f32> {
    a.iter()
        .map(|x| {
            let v = if scalar_first { op.apply(scalar, *x) } else { op.apply(*x, scalar) };
            act.apply(v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_and_scalar_hand_computed() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![0.5, 1.0, -1.0];
        assert_eq!(binary(BinaryOp::Add, &a, &b, Activation::None), vec![1.5, -1.0, 2.0]);
        assert_eq!(binary(BinaryOp::Mul, &a, &b, Activation::Relu), vec![0.5, 0.0, 0.0]);
        // scalar_first: 1 - x
        assert_eq!(
            binary_scalar(BinaryOp::Sub, &a, 1.0, true, Activation::None),
            vec![0.0, 3.0, -2.0]
        );
    }
}
