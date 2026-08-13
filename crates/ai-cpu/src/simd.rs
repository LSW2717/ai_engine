//! f32x4 SIMD 추상화 — 커널이 아키텍처를 모르게 하는 최소 표면.
//!
//! 백엔드: aarch64 NEON / wasm32 SIMD128(`+simd128` 필요) / 스칼라 폴백.
//! 새 아키텍처(x86 AVX 등)는 이 파일에 cfg 블록 하나 추가로 끝나야 한다 —
//! 커널 코드는 절대 `core::arch`를 직접 만지지 않는다.
//!
//! `load/store`는 슬라이스 기반(디버그에서 경계 검증, 릴리스에서 무검사)이다.
//! 핫루프가 인덱스 산술을 소유하고, 여기는 4레인 연산만 소유한다.

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct F32x4(float32x4_t);

    impl F32x4 {
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            unsafe { Self(vdupq_n_f32(v)) }
        }
        #[inline(always)]
        pub fn load(s: &[f32], i: usize) -> Self {
            debug_assert!(i + 4 <= s.len());
            unsafe { Self(vld1q_f32(s.as_ptr().add(i))) }
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f32], i: usize) {
            debug_assert!(i + 4 <= s.len());
            unsafe { vst1q_f32(s.as_mut_ptr().add(i), self.0) }
        }
        /// self + a*b (융합 곱셈-누산)
        #[inline(always)]
        pub fn fma(self, a: Self, b: Self) -> Self {
            unsafe { Self(vfmaq_f32(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            unsafe { Self(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            unsafe { Self(vmulq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(vmaxq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(vminq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut a = [0f32; 4];
            unsafe { vst1q_f32(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline(always)]
        pub fn sum(self) -> f32 {
            unsafe { vaddvq_f32(self.0) }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F32x4(v128);

    impl F32x4 {
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            Self(f32x4_splat(v))
        }
        #[inline(always)]
        pub fn load(s: &[f32], i: usize) -> Self {
            debug_assert!(i + 4 <= s.len());
            unsafe { Self(v128_load(s.as_ptr().add(i) as *const v128)) }
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f32], i: usize) {
            debug_assert!(i + 4 <= s.len());
            unsafe { v128_store(s.as_mut_ptr().add(i) as *mut v128, self.0) }
        }
        /// self + a*b — SIMD128에는 fma가 없어 mul+add (relaxed-simd 비의존)
        #[inline(always)]
        pub fn fma(self, a: Self, b: Self) -> Self {
            Self(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(f32x4_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(f32x4_mul(self.0, o.0))
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(f32x4_max(self.0, o.0))
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(f32x4_min(self.0, o.0))
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut a = [0f32; 4];
            unsafe { v128_store(a.as_mut_ptr() as *mut v128, self.0) };
            a
        }
        #[inline(always)]
        pub fn sum(self) -> f32 {
            let a = self.to_array();
            (a[0] + a[1]) + (a[2] + a[3])
        }
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod imp {
    /// 스칼라 폴백 — 컴파일러 자동 벡터화에 맡긴다 (x86 등 미지원 아키)
    #[derive(Clone, Copy)]
    pub struct F32x4([f32; 4]);

    impl F32x4 {
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            Self([v; 4])
        }
        #[inline(always)]
        pub fn load(s: &[f32], i: usize) -> Self {
            Self([s[i], s[i + 1], s[i + 2], s[i + 3]])
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f32], i: usize) {
            s[i..i + 4].copy_from_slice(&self.0);
        }
        #[inline(always)]
        pub fn fma(self, a: Self, b: Self) -> Self {
            let mut o = self.0;
            for l in 0..4 {
                o[l] += a.0[l] * b.0[l];
            }
            Self(o)
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            let mut r = self.0;
            for l in 0..4 {
                r[l] += o.0[l];
            }
            Self(r)
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            let mut r = self.0;
            for l in 0..4 {
                r[l] *= o.0[l];
            }
            Self(r)
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            let mut r = self.0;
            for l in 0..4 {
                r[l] = r[l].max(o.0[l]);
            }
            Self(r)
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            let mut r = self.0;
            for l in 0..4 {
                r[l] = r[l].min(o.0[l]);
            }
            Self(r)
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            self.0
        }
        #[inline(always)]
        pub fn sum(self) -> f32 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }
}

pub use imp::F32x4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_ops() {
        let s = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let a = F32x4::load(&s, 0);
        let b = F32x4::load(&s, 1);
        // fma: 10 + a*b = [10+2, 10+6, 10+12, 10+20]
        let r = F32x4::splat(10.0).fma(a, b).to_array();
        assert_eq!(r, [12.0, 16.0, 22.0, 30.0]);
        assert_eq!(a.add(b).to_array(), [3.0, 5.0, 7.0, 9.0]);
        assert_eq!(a.mul(b).to_array(), [2.0, 6.0, 12.0, 20.0]);
        assert_eq!(a.max(F32x4::splat(2.5)).to_array(), [2.5, 2.5, 3.0, 4.0]);
        assert_eq!(a.min(F32x4::splat(2.5)).to_array(), [1.0, 2.0, 2.5, 2.5]);
        assert_eq!(a.sum(), 10.0);
        let mut out = [0f32; 6];
        a.store(&mut out, 2);
        assert_eq!(out, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
