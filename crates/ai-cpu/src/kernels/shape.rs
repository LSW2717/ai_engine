//! 형태 연산 — Concat(채널 연결)·Chcopy(채널 슬라이스 실체화).
//! 둘 다 뷰 → 밀집 복사가 전부다.

use crate::view::View;

/// 뷰를 밀집 out의 채널 구간 [c_off_out, c_off_out+view.c)에 복사.
/// Concat = 파트마다 이 함수, Chcopy·alias 실체화 = c_off_out 0으로 한 번.
pub fn copy_view_into(view: View, px: usize, out: &mut [f32], out_stride: usize, c_off_out: usize) {
    debug_assert!(c_off_out + view.c <= out_stride);
    debug_assert!(out.len() >= px * out_stride);
    for p in 0..px {
        let b = view.base(p);
        out[p * out_stride + c_off_out..p * out_stride + c_off_out + view.c]
            .copy_from_slice(&view.data[b..b + view.c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;

    #[test]
    fn concat_two_parts() {
        let mut rng = XorShift32::new(1);
        let px = 4usize;
        let a = rng.vec_f32(px * 2);
        let b = rng.vec_f32(px * 3);
        let mut out = vec![0f32; px * 5];
        copy_view_into(View::dense(&a, 2), px, &mut out, 5, 0);
        copy_view_into(View::dense(&b, 3), px, &mut out, 5, 2);
        for p in 0..px {
            assert_eq!(out[p * 5..p * 5 + 2], a[p * 2..(p + 1) * 2]);
            assert_eq!(out[p * 5 + 2..(p + 1) * 5], b[p * 3..(p + 1) * 3]);
        }
    }

    #[test]
    fn chcopy_channel_slice() {
        let mut rng = XorShift32::new(2);
        let px = 3usize;
        let x = rng.vec_f32(px * 6);
        // 채널 2..5 슬라이스
        let mut out = vec![0f32; px * 3];
        copy_view_into(View { data: &x, c_off: 2, stride: 6, c: 3 }, px, &mut out, 3, 0);
        for p in 0..px {
            assert_eq!(out[p * 3..(p + 1) * 3], x[p * 6 + 2..p * 6 + 5]);
        }
    }
}
