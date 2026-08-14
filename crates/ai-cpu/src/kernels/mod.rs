//! CPU 커널 계열 — 파일 하나 = 커널 하나, 각자 레퍼런스 대조 테스트를 소유한다
//! (ai-gpu/src/kernels와 같은 규율).

use ai_core::Activation;

use crate::simd::F32x4;

pub mod conv;
pub mod dw;
pub mod elementwise;
pub mod im2row;
pub mod pool;
pub mod pw_dot;
pub mod resize;
pub mod segate;
pub mod shape;

/// 활성화 4레인 벡터 적용 — 초월함수(sigmoid/tanh)만 스칼라로 내려간다.
/// 핫루프 안에서 act는 고정이라 match 분기는 완전 예측된다.
#[inline(always)]
pub fn apply4(act: Activation, v: F32x4) -> F32x4 {
    match act {
        Activation::None => v,
        Activation::Relu => v.max(F32x4::splat(0.0)),
        // x * clamp(x/6 + 0.5, 0, 1)
        Activation::Hardswish => v.mul(
            v.mul(F32x4::splat(1.0 / 6.0))
                .add(F32x4::splat(0.5))
                .max(F32x4::splat(0.0))
                .min(F32x4::splat(1.0)),
        ),
        Activation::Hardsigmoid => v
            .mul(F32x4::splat(1.0 / 6.0))
            .add(F32x4::splat(0.5))
            .max(F32x4::splat(0.0))
            .min(F32x4::splat(1.0)),
        Activation::Clamp01 => v.max(F32x4::splat(0.0)).min(F32x4::splat(1.0)),
        Activation::Relu6 => v.max(F32x4::splat(0.0)).min(F32x4::splat(6.0)),
        Activation::Neg => v.mul(F32x4::splat(-1.0)),
        Activation::Sqrt | Activation::Recip | Activation::Sigmoid | Activation::Tanh => {
            let a = v.to_array();
            F32x4::from_array([
                act.apply(a[0]),
                act.apply(a[1]),
                act.apply(a[2]),
                act.apply(a[3]),
            ])
        }
    }
}
