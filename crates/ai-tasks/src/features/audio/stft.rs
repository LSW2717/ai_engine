//! 스트리밍 STFT / iSTFT — vcxrust_ai vcx-noise fast-enhancer/stft.rs 이식.
//!
//! 원본 FastEnhancer wav2wav 수출의 `ONNXSTFT` 재현:
//!   - n_fft 슬라이딩 윈도우, hop 전진 (torch.stft center 패딩 아님)
//!   - periodic Hann 분석 윈도우 (win_size == n_fft)
//!   - iSTFT = irfft(spec) × (window / Σwindow²) + overlap-add
//!
//! irfft는 conj 대칭 확장(Nyquist=0, DC는 그대로) — 원본 ONNX의
//! zero-pad c2c 트릭(2·Re(ifft) − Re(S₀)/N)과 수학적으로 동일하다
//! (tools/prep_fastenhancer.py --verify가 원본과 1e-9 일치로 검증한 식).

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

type C32 = Complex<f32>;

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    /// n_fft / 2 + 1
    pub bins: usize,
    window: Vec<f32>,
    /// window / Σwindow² (COLA 정규화)
    window_istft: Vec<f32>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        assert!(n_fft % 2 == 0 && hop > 0 && hop <= n_fft);
        // torch.hann_window(n_fft, periodic=True)
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / n_fft as f32).cos())
            .collect();
        let k = n_fft.div_ceil(hop);
        let l = hop * (2 * k - 1) + (n_fft - hop);
        let mut ws = vec![0f32; l];
        for j in 0..(2 * k - 1) {
            for i in 0..n_fft {
                ws[j * hop + i] += window[i] * window[i];
            }
        }
        let off = (k - 1) * hop;
        let window_istft: Vec<f32> = (0..n_fft)
            .map(|i| {
                let s = ws[off + i];
                if s > 1e-12 { window[i] / s } else { 0.0 }
            })
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        Self { n_fft, hop, bins: n_fft / 2 + 1, window, window_istft, fwd, inv }
    }

    /// frame(len n_fft) → out(len bins·2, [re,im] 인터리브)
    pub fn rfft_into(&self, frame: &[f32], out: &mut [f32], scratch: &mut Vec<C32>) {
        debug_assert_eq!(frame.len(), self.n_fft);
        debug_assert_eq!(out.len(), self.bins * 2);
        scratch.clear();
        scratch.extend(frame.iter().zip(&self.window).map(|(x, w)| C32::new(x * w, 0.0)));
        self.fwd.process(scratch);
        for i in 0..self.bins {
            out[i * 2] = scratch[i].re;
            out[i * 2 + 1] = scratch[i].im;
        }
    }

    /// spec(len bins·2 인터리브) → 원시 irfft 샘플(len n_fft, window_istft 곱 완료).
    /// overlap-add는 호출자 몫.
    pub fn irfft_into(&self, spec: &[f32], out: &mut [f32], scratch: &mut Vec<C32>) {
        debug_assert_eq!(spec.len(), self.bins * 2);
        debug_assert_eq!(out.len(), self.n_fft);
        scratch.clear();
        scratch.resize(self.n_fft, C32::new(0.0, 0.0));
        for i in 0..self.bins {
            scratch[i] = C32::new(spec[i * 2], spec[i * 2 + 1]);
        }
        for i in 1..(self.n_fft / 2) {
            scratch[self.n_fft - i] = scratch[i].conj();
        }
        self.inv.process(scratch);
        let scale = 1.0 / self.n_fft as f32;
        for i in 0..self.n_fft {
            out[i] = scratch[i].re * scale * self.window_istft[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_istft_roundtrip_is_identity() {
        // COLA 정규화 검증: 정상상태(연속 hop)에서 완전 재구성이어야 한다
        let (n_fft, hop) = (1024usize, 512usize);
        let stft = Stft::new(n_fft, hop);
        let sig: Vec<f32> = (0..hop * 6)
            .map(|i| (i as f32 * 0.013).sin() * 0.7 + (i as f32 * 0.037).cos() * 0.2)
            .collect();
        let mut in_cache = vec![0f32; n_fft - hop];
        let mut ola = vec![0f32; n_fft - hop];
        let mut spec = vec![0f32; stft.bins * 2];
        let mut frame = vec![0f32; n_fft];
        let mut irf = vec![0f32; n_fft];
        let mut scratch = Vec::new();
        let mut out_all = Vec::new();
        for h in 0..6 {
            let seg = &sig[h * hop..(h + 1) * hop];
            frame[..n_fft - hop].copy_from_slice(&in_cache);
            frame[n_fft - hop..].copy_from_slice(seg);
            in_cache.copy_from_slice(&frame[hop..]);
            stft.rfft_into(&frame, &mut spec, &mut scratch);
            stft.irfft_into(&spec, &mut irf, &mut scratch);
            for i in 0..(n_fft - hop) {
                irf[i] += ola[i];
            }
            out_all.extend_from_slice(&irf[..hop]);
            ola.copy_from_slice(&irf[hop..]);
        }
        // 출력은 n_fft-hop 지연 — 정상상태 구간 비교
        let delay = n_fft - hop;
        let mut max_err = 0f32;
        for i in 0..hop * 4 {
            max_err = max_err.max((out_all[delay + i] - sig[i]).abs());
        }
        assert!(max_err < 1e-4, "재구성 오차 {max_err}");
    }
}
