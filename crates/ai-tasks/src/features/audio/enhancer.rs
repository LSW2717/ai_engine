//! FastEnhancer 스트리밍 노이즈 제거 — vcx-noise engine.rs의 ai_engine판
//! (ncnn → 오디오 미니 실행기, STFT·전/후처리는 Rust).
//!
//! hop(512@48k / 256@16k) 샘플 mono f32 [-1,1]를 넣으면 hop 샘플이 나온다.
//! 출력은 입력 대비 n_fft−hop 샘플 지연 (STFT 스트리밍 공통).
//!
//! 전/후처리는 tools/prep_fastenhancer.py --verify가 원본 wav2wav ONNX와
//! 1e-9로 대조한 수식 그대로 (상수 α/β/clip은 manifest에서 옴):
//!   압축   comp = X · max(|X|, clip)^(α−1)      (Nyquist 빈은 드랍)
//!   마스크 곱 Y = M ⊙ comp (복소)
//!   역압축 Y·|Y|^(β−1) → Nyquist 0으로 conj-irfft → COLA OLA

use rustfft::num_complex::Complex;

use crate::error::TaskError;
use super::graph::FeGraph;
use super::ops::Tens;
use super::stft::Stft;

type C32 = Complex<f32>;

pub struct Enhancer {
    graph: FeGraph,
    stft: Stft,
    /// n_fft/2 — 넷이 소비하는 빈 수 (Nyquist 드랍)
    bins_used: usize,
    in_cache: Vec<f32>,
    ola: Vec<f32>,
    caches: Vec<Tens>,
    // scratch
    frame: Vec<f32>,
    spec: Vec<f32>,
    spec_out: Vec<f32>,
    comp: Vec<f32>,
    irf: Vec<f32>,
    fft_scratch: Vec<C32>,
}

impl Enhancer {
    /// graph.json + weights.bin (tools/prep_fastenhancer.py --export 산출물)
    pub fn new(graph_json: &[u8], weights: &[u8]) -> Result<Self, TaskError> {
        let graph = FeGraph::load(graph_json, weights)?;
        let pp = graph.pre_post;
        let stft = Stft::new(pp.n_fft, pp.hop);
        let caches: Vec<Tens> =
            graph.inputs[1..].iter().map(|(_, s)| Tens::zeros(s.clone())).collect();
        if caches.len() != 3 {
            return Err(TaskError::Other(format!("GRU 캐시 3개 기대, {}개", caches.len())));
        }
        let cache_len = pp.n_fft - pp.hop;
        Ok(Enhancer {
            bins_used: pp.n_fft / 2,
            in_cache: vec![0.0; cache_len],
            ola: vec![0.0; cache_len],
            caches,
            frame: vec![0.0; pp.n_fft],
            spec: vec![0.0; stft.bins * 2],
            spec_out: vec![0.0; stft.bins * 2],
            comp: vec![0.0; pp.n_fft],
            irf: vec![0.0; pp.n_fft],
            fft_scratch: Vec::with_capacity(pp.n_fft),
            graph,
            stft,
        })
    }

    /// process_frame 호출당 기대 샘플 수 (= hop)
    pub fn frame_len(&self) -> usize {
        self.stft.hop
    }

    /// 스트리밍 상태 초기화 (캐시·OLA — 스트림 파기 시)
    pub fn reset(&mut self) {
        self.in_cache.fill(0.0);
        self.ola.fill(0.0);
        for c in &mut self.caches {
            c.data.fill(0.0);
        }
    }

    /// input/output 각 frame_len() 샘플. 부족하면 0 패딩/절단.
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), TaskError> {
        let n_fft = self.stft.n_fft;
        let hop = self.stft.hop;
        let cache_len = n_fft - hop;
        let pp = self.graph.pre_post;

        // 분석 프레임 = [in_cache, 새 hop]
        let n = input.len().min(hop);
        self.frame[..cache_len].copy_from_slice(&self.in_cache);
        self.frame[cache_len..cache_len + n].copy_from_slice(&input[..n]);
        self.frame[cache_len + n..].fill(0.0);
        self.in_cache.copy_from_slice(&self.frame[hop..]);

        // STFT → 압축 (comp = X · max(|X|,clip)^(α−1), 인터리브 [re,im])
        self.stft.rfft_into(&self.frame, &mut self.spec, &mut self.fft_scratch);
        for i in 0..self.bins_used {
            let (re, im) = (self.spec[i * 2], self.spec[i * 2 + 1]);
            let mag = (re * re + im * im).sqrt().max(pp.clip_min);
            let f = mag.powf(pp.alpha - 1.0);
            self.comp[i * 2] = re * f;
            self.comp[i * 2 + 1] = im * f;
        }

        // 그래프 실행: [comp [1,F,1,2], 캐시×3] → [mask [1,2,F] (plane-major), 캐시×3]
        let mut inputs = Vec::with_capacity(4);
        inputs.push(Tens::new(
            self.graph.inputs[0].1.clone(),
            self.comp[..self.bins_used * 2].to_vec(),
        ));
        for c in &self.caches {
            inputs.push(c.clone());
        }
        let mut outs = self.graph.run(inputs)?;
        for (i, c) in outs.drain(1..).enumerate() {
            // 캐시 출력은 [1,Fr,C]일 수 있음 — 입력 shape로 재해석 (데이터 동일)
            self.caches[i] = Tens::new(self.graph.inputs[1 + i].1.clone(), c.data);
        }
        let mask = outs.remove(0);
        let f = self.bins_used;
        debug_assert_eq!(mask.numel(), f * 2);

        // 복소 마스크 곱 + 역압축 → spec_out (Nyquist 0)
        for i in 0..f {
            let (mre, mim) = (mask.data[i], mask.data[f + i]);
            let (cre, cim) = (self.comp[i * 2], self.comp[i * 2 + 1]);
            let mut yre = mre * cre - mim * cim;
            let mut yim = mre * cim + mim * cre;
            let ymag = (yre * yre + yim * yim).sqrt();
            if ymag > 0.0 {
                let g = ymag.powf(pp.beta - 1.0);
                yre *= g;
                yim *= g;
            }
            self.spec_out[i * 2] = yre;
            self.spec_out[i * 2 + 1] = yim;
        }
        self.spec_out[f * 2] = 0.0; // Nyquist
        self.spec_out[f * 2 + 1] = 0.0;

        // iSTFT + OLA
        self.stft.irfft_into(&self.spec_out, &mut self.irf, &mut self.fft_scratch);
        for i in 0..cache_len {
            self.irf[i] += self.ola[i];
        }
        let out_n = output.len().min(hop);
        output[..out_n].copy_from_slice(&self.irf[..out_n]);
        output[out_n..].fill(0.0);
        self.ola.copy_from_slice(&self.irf[hop..]);
        Ok(())
    }
}
