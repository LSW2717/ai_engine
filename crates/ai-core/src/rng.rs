//! 시드 고정 PRNG — getrandom 의존 없이 네이티브/wasm에서 동일한 테스트 입력을 재현.

pub struct XorShift32(u32);

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x9e37_79b9 } else { seed })
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// [-1, 1) 균등 f32
    pub fn next_f32(&mut self) -> f32 {
        // 상위 24비트 → [0,1) → [-1,1)
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }

    pub fn fill_f32(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.next_f32();
        }
    }

    pub fn vec_f32(&mut self, len: usize) -> Vec<f32> {
        let mut v = vec![0.0; len];
        self.fill_f32(&mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = XorShift32::new(42);
        let mut b = XorShift32::new(42);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((-1.0..1.0).contains(&x));
        }
    }
}
