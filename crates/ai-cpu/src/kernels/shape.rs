//! 형태 연산 — Concat(채널 연결)·Chcopy(채널 슬라이스 실체화).
//! 둘 다 뷰 → 밀집 복사가 전부다.

use crate::simd::F32x4;
use crate::view::View;

/// 뷰를 밀집 out의 채널 구간 [c_off_out, c_off_out+view.c)에 복사.
/// Concat = 파트마다 이 함수, Chcopy·alias 실체화 = c_off_out 0으로 한 번.
/// 픽셀당 c가 작아(수십) copy_from_slice의 memcpy 호출 오버헤드가 지배한다 —
/// 벡터 루프 + 스칼라 꼬리가 wasm에서 1.5배 빠르다.
pub fn copy_view_into(view: View, px: usize, out: &mut [f32], out_stride: usize, c_off_out: usize) {
    debug_assert!(c_off_out + view.c <= out_stride);
    debug_assert!(out.len() >= px * out_stride);
    let c = view.c;
    let cv = c / 4 * 4;
    for p in 0..px {
        let b = view.base(p);
        let o = p * out_stride + c_off_out;
        let mut cc = 0usize;
        while cc < cv {
            F32x4::load(view.data, b + cc).store(out, o + cc);
            cc += 4;
        }
        for ch in cv..c {
            out[o + ch] = view.data[b + ch];
        }
    }
}

/// (1,w,c) → (1,c,w) 2D 전치 (h=1 전용 — MLP-Mixer 토큰↔채널, face_blendshapes).
/// out은 밀집 (c픽셀 × w채널). 소형(97×64급)이라 스칼라 이중 루프로 충분 —
/// 읽기 순차·쓰기 스트라이드(w) 방향이 캐시에 유리하다.
pub fn transpose_wc(view: View, w: usize, c: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= w * c);
    for x in 0..w {
        let b = view.base(x);
        for j in 0..c {
            out[j * w + x] = view.data[b + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::rng::XorShift32;

    #[test]
    fn transpose_wc_roundtrip() {
        let (w, c) = (5usize, 3usize);
        let data: Vec<f32> = (0..w * c).map(|i| i as f32).collect();
        let v = View { data: &data, c_off: 0, stride: c, c };
        let mut out = vec![0f32; w * c];
        transpose_wc(v, w, c, &mut out);
        for x in 0..w {
            for j in 0..c {
                assert_eq!(out[j * w + x], data[x * c + j]);
            }
        }
    }

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
