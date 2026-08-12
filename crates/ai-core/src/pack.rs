//! 텐서·가중치 패킹 — CPU(논리 레이아웃) ↔ GPU(NHWC-C4 vec4) 변환.
//!
//! 이 모듈이 곧 변환기(ai-convert)의 출력 계약이다: 여기서 정의한 바이트 배열이
//! 그대로 GPU 버퍼에 업로드되므로, 런타임 로드는 memcpy가 된다.
//!
//! 가중치 레이아웃 (모두 vec4 단위, j = 출력 서브채널, comp = 입력 서브채널):
//! - 일반/pointwise conv: `[tap][kg][ng][j]` — tap-major라 implicit GEMM의 B 진행이 선형이고,
//!   고정 kg에서 ng가 연속이라 GEMM B-타일 협조 로드가 coalesced된다.
//!   KG는 `kg_align` 그룹 배수로 제로패딩(커널이 K 나머지를 처리하지 않도록).
//! - depthwise conv: `[cg][tap]` — 채널그룹의 탭 가중치가 연속(워크그룹 공유메모리 적재 단위).
//! - bias: `[ng]` vec4, cout 4패딩.

use crate::tensor::{DType, TensorDesc};
use half::f16;

fn f32s_to_bytes(v: Vec<f32>, dt: DType) -> Vec<u8> {
    match dt {
        DType::F32 => bytemuck::cast_slice(&v).to_vec(),
        DType::F16 => {
            let h: Vec<f16> = v.into_iter().map(f16::from_f32).collect();
            bytemuck::cast_slice(&h).to_vec()
        }
    }
}

fn bytes_to_f32s(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(b).to_vec(),
        DType::F16 => bytemuck::cast_slice::<u8, f16>(b).iter().map(|h| h.to_f32()).collect(),
    }
}

/// 논리 NHWC(`[h][w][c]`, 패딩 없음) → NHWC-C4 패킹 바이트
pub fn pack_nhwc(data: &[f32], d: &TensorDesc) -> Vec<u8> {
    assert_eq!(data.len(), d.elems(), "pack_nhwc: 입력 길이 불일치");
    let mut out = vec![0f32; (d.vec4_len() * 4) as usize];
    for h in 0..d.h {
        for w in 0..d.w {
            for c in 0..d.c {
                let src = ((h * d.w + w) * d.c + c) as usize;
                let dst = (d.idx(h, w, c / 4) * 4 + (c % 4) as u64) as usize;
                out[dst] = data[src];
            }
        }
    }
    f32s_to_bytes(out, d.dt)
}

/// NHWC-C4 패킹 바이트 → 논리 NHWC (패딩 채널 제거)
pub fn unpack_nhwc(bytes: &[u8], d: &TensorDesc) -> Vec<f32> {
    let flat = bytes_to_f32s(bytes, d.dt);
    assert_eq!(flat.len(), (d.vec4_len() * 4) as usize, "unpack_nhwc: 바이트 길이 불일치");
    let mut out = vec![0f32; d.elems()];
    for h in 0..d.h {
        for w in 0..d.w {
            for c in 0..d.c {
                let src = (d.idx(h, w, c / 4) * 4 + (c % 4) as u64) as usize;
                let dst = ((h * d.w + w) * d.c + c) as usize;
                out[dst] = flat[src];
            }
        }
    }
    out
}

/// 일반/pointwise conv 가중치: OIHW(`[cout][cin][kh][kw]`) → `[tap][kg][ng][j]` vec4.
/// 반환: (bytes, kg_padded) — kg_padded는 kg_align 배수로 패딩된 입력 채널그룹 수.
pub fn pack_weights_conv(
    w_oihw: &[f32],
    cout: u32,
    cin: u32,
    kh: u32,
    kw: u32,
    kg_align: u32,
    dt: DType,
) -> (Vec<u8>, u32) {
    assert_eq!(w_oihw.len(), (cout * cin * kh * kw) as usize);
    let kg = cin.div_ceil(4).next_multiple_of(kg_align.max(1));
    let ng = cout.div_ceil(4);
    let taps = kh * kw;
    let mut out = vec![0f32; (taps * kg * ng * 16) as usize];
    for t in 0..taps {
        let (ky, kx) = (t / kw, t % kw);
        for kgi in 0..kg {
            for ngi in 0..ng {
                for j in 0..4u32 {
                    for comp in 0..4u32 {
                        let oc = ngi * 4 + j;
                        let ic = kgi * 4 + comp;
                        let v = if oc < cout && ic < cin {
                            w_oihw[((((oc * cin) + ic) * kh + ky) * kw + kx) as usize]
                        } else {
                            0.0
                        };
                        out[(((((t * kg + kgi) * ng + ngi) * 4) + j) * 4 + comp) as usize] = v;
                    }
                }
            }
        }
    }
    (f32s_to_bytes(out, dt), kg)
}

/// depthwise conv 가중치: `[c][kh][kw]` → `[cg][tap]` vec4 (comp = 그룹 내 채널)
pub fn pack_weights_dw(w: &[f32], c: u32, kh: u32, kw: u32, dt: DType) -> Vec<u8> {
    assert_eq!(w.len(), (c * kh * kw) as usize);
    let cg = c.div_ceil(4);
    let taps = kh * kw;
    let mut out = vec![0f32; (cg * taps * 4) as usize];
    for cgi in 0..cg {
        for t in 0..taps {
            for comp in 0..4u32 {
                let ch = cgi * 4 + comp;
                let v = if ch < c { w[(ch * taps + t) as usize] } else { 0.0 };
                out[((cgi * taps + t) * 4 + comp) as usize] = v;
            }
        }
    }
    f32s_to_bytes(out, dt)
}

/// 역패킹: `[tap][kg][ng][j]` → OIHW (verify/테스트용 — pack의 역함수)
pub fn unpack_weights_conv(
    bytes: &[u8],
    cout: u32,
    cin: u32,
    kh: u32,
    kw: u32,
    kg_pad: u32,
    dt: DType,
) -> Vec<f32> {
    let flat = bytes_to_f32s(bytes, dt);
    let ng = cout.div_ceil(4);
    let mut out = vec![0f32; (cout * cin * kh * kw) as usize];
    for t in 0..kh * kw {
        let (ky, kx) = (t / kw, t % kw);
        for kgi in 0..cin.div_ceil(4) {
            for ngi in 0..ng {
                for j in 0..4u32 {
                    for comp in 0..4u32 {
                        let oc = ngi * 4 + j;
                        let ic = kgi * 4 + comp;
                        if oc < cout && ic < cin {
                            out[((((oc * cin) + ic) * kh + ky) * kw + kx) as usize] =
                                flat[(((((t * kg_pad + kgi) * ng + ngi) * 4) + j) * 4 + comp)
                                    as usize];
                        }
                    }
                }
            }
        }
    }
    out
}

/// 역패킹: `[cg][tap]` → `[c][kh][kw]`
pub fn unpack_weights_dw(bytes: &[u8], c: u32, kh: u32, kw: u32, dt: DType) -> Vec<f32> {
    let flat = bytes_to_f32s(bytes, dt);
    let taps = kh * kw;
    let mut out = vec![0f32; (c * taps) as usize];
    for ch in 0..c {
        for t in 0..taps {
            out[(ch * taps + t) as usize] = flat[(((ch / 4) * taps + t) * 4 + ch % 4) as usize];
        }
    }
    out
}

/// 역패킹: `[ng]` vec4 → `[cout]`
pub fn unpack_bias(bytes: &[u8], cout: u32, dt: DType) -> Vec<f32> {
    bytes_to_f32s(bytes, dt)[..cout as usize].to_vec()
}

/// bias: `[cout]` → `[ng]` vec4 (4패딩). bias가 없는 conv는 0 벡터를 바인딩한다.
pub fn pack_bias(b: &[f32], cout: u32, dt: DType) -> Vec<u8> {
    assert_eq!(b.len(), cout as usize);
    let ng = cout.div_ceil(4);
    let mut out = vec![0f32; (ng * 4) as usize];
    out[..cout as usize].copy_from_slice(b);
    f32s_to_bytes(out, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::XorShift32;

    #[test]
    fn nhwc_roundtrip_f32_odd_shapes() {
        // W 홀수, C%4≠0 — 구 엔진 제약이 없는지 레이아웃 수준에서 확인
        let d = TensorDesc::new(3, 5, 6, DType::F32);
        let data = XorShift32::new(1).vec_f32(d.elems());
        let packed = pack_nhwc(&data, &d);
        assert_eq!(packed.len() as u64, d.size_bytes());
        assert_eq!(unpack_nhwc(&packed, &d), data);
    }

    #[test]
    fn nhwc_roundtrip_f16_tolerance() {
        let d = TensorDesc::new(2, 3, 5, DType::F16);
        let data = XorShift32::new(2).vec_f32(d.elems());
        let rt = unpack_nhwc(&pack_nhwc(&data, &d), &d);
        for (a, b) in data.iter().zip(&rt) {
            assert!((a - b).abs() < 1e-3, "f16 왕복 오차 초과: {a} vs {b}");
        }
    }

    #[test]
    fn pack_padding_channels_are_zero() {
        // c=3 → 4번째 컴포넌트는 0이어야 함
        let d = TensorDesc::new(1, 2, 3, DType::F32);
        let data = vec![1.0; d.elems()];
        let packed = pack_nhwc(&data, &d);
        let flat: &[f32] = bytemuck::cast_slice(&packed);
        assert_eq!(flat[3], 0.0);
        assert_eq!(flat[7], 0.0);
    }

    #[test]
    fn conv_weight_layout_pointwise_identity() {
        // cin=cout=4 항등 행렬 → 단일 tap, 단일 kg/ng, w[j][comp] = I
        let mut w = vec![0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let (packed, kg) = pack_weights_conv(&w, 4, 4, 1, 1, 1, DType::F32);
        assert_eq!(kg, 1);
        let flat: &[f32] = bytemuck::cast_slice(&packed);
        // [tap=0][kg=0][ng=0][j] 의 comp들 = 항등
        for j in 0..4 {
            for comp in 0..4 {
                assert_eq!(flat[j * 4 + comp], if j == comp { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn conv_weight_kg_alignment_pads_zero() {
        // cin=6 → kg=2, align 4 → kg_padded=4, 패딩 그룹은 전부 0
        let w = XorShift32::new(3).vec_f32(8 * 6);
        let (packed, kg) = pack_weights_conv(&w, 8, 6, 1, 1, 4, DType::F32);
        assert_eq!(kg, 4);
        let flat: &[f32] = bytemuck::cast_slice(&packed);
        let ng = 2;
        // kg 2..4 영역은 모두 0
        for kgi in 2..4 {
            for i in 0..ng * 16 {
                assert_eq!(flat[(kgi * ng * 16 + i) as usize], 0.0);
            }
        }
    }

    #[test]
    fn pack_unpack_roundtrip_conv_and_dw() {
        let mut rng = XorShift32::new(11);
        let (cout, cin, k) = (10u32, 6u32, 3u32);
        let w = rng.vec_f32((cout * cin * k * k) as usize);
        let (packed, kg_pad) = pack_weights_conv(&w, cout, cin, k, k, 4, DType::F32);
        assert_eq!(unpack_weights_conv(&packed, cout, cin, k, k, kg_pad, DType::F32), w);

        let c = 7u32;
        let wd = rng.vec_f32((c * 9) as usize);
        let packed = pack_weights_dw(&wd, c, 3, 3, DType::F32);
        assert_eq!(unpack_weights_dw(&packed, c, 3, 3, DType::F32), wd);

        let b = rng.vec_f32(cout as usize);
        assert_eq!(unpack_bias(&pack_bias(&b, cout, DType::F32), cout, DType::F32), b);
    }

    #[test]
    fn dw_weight_layout() {
        // c=5, k=3: cg=2, [cg][tap] — 채널 4(두 번째 그룹 comp 0)의 tap 순서 확인
        let c = 5u32;
        let taps = 9u32;
        let w: Vec<f32> = (0..c * taps).map(|i| i as f32).collect();
        let packed = pack_weights_dw(&w, c, 3, 3, DType::F32);
        let flat: &[f32] = bytemuck::cast_slice(&packed);
        for t in 0..taps {
            // cg=1, comp=0 → 채널 4의 t번째 탭 = 4*9 + t
            assert_eq!(flat[((1 * taps + t) * 4) as usize], (4 * taps + t) as f32);
            // comp=1..3은 패딩 → 0
            assert_eq!(flat[((1 * taps + t) * 4 + 1) as usize], 0.0);
        }
    }
}
