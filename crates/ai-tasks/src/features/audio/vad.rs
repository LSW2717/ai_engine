//! Silero VAD (v6.2, 16k) — 음성 확률. 「마이크가 꺼진 채 말하고 있나요?」 감지용.
//!
//! 그래프는 tools/prep_silero.py 가 원본 ONNX 가중치로 짠 **정적 1-청크 그래프**
//! (input [1,576] + h,c [1,128] → prob [1,1] + h_out,c_out). 공식 래퍼(OnnxWrapper)가 하는
//! 일 — 앞 청크 꼬리 64 샘플을 이어 붙이고 LSTM 상태를 청크마다 넘기는 것 — 은 여기서 한다.
//!
//! 입력 레이트: 캡처는 보통 48k(평소)·16k(BT HFP)다. 16k 는 그대로, 그 외는 창 sinc
//! 저역통과 리샘플러로 16k 로 내린다(48k 는 위상이 늘 0 인 정수 1/3 경로). 브라우저가
//! 44.1k 를 주는 경우도 같은 리샘플러가 받는다.

use super::graph::FeGraph;
use super::ops::Tens;
use crate::error::TaskError;

/// 모델 레이트.
pub const VAD_RATE: u32 = 16_000;
/// 청크 길이(16k 샘플) — 32ms.
pub const CHUNK: usize = 512;
const CONTEXT: usize = 64;
const HIDDEN: usize = 128;

pub struct Vad {
    graph: FeGraph,
    resampler: Option<Resampler>,
    /// 16k 로 내린, 아직 청크가 안 된 샘플
    pending: Vec<f32>,
    /// [앞 청크 꼬리 64 | 이번 청크 512]
    window: Vec<f32>,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl Vad {
    /// graph.json + weights.bin (tools/prep_silero.py --export 산출물). sample_rate = 입력 레이트.
    pub fn new(graph_json: &[u8], weights: &[u8], sample_rate: u32) -> Result<Self, TaskError> {
        if sample_rate < 8_000 {
            return Err(TaskError::Other(format!(
                "vad: 입력 레이트 {sample_rate} 미지원"
            )));
        }
        let graph = FeGraph::load(graph_json, weights)?;
        let names: Vec<&str> = graph.inputs.iter().map(|(n, _)| n.as_str()).collect();
        if names != ["input", "h", "c"] || graph.outputs != ["prob", "h_out", "c_out"] {
            return Err(TaskError::Other(format!(
                "vad: 그래프 입출력 불일치 {names:?} → {:?}",
                graph.outputs
            )));
        }
        Ok(Vad {
            graph,
            resampler: (sample_rate != VAD_RATE).then(|| Resampler::new(sample_rate, VAD_RATE)),
            pending: Vec::with_capacity(CHUNK * 2),
            window: vec![0.0; CONTEXT + CHUNK],
            h: vec![0.0; HIDDEN],
            c: vec![0.0; HIDDEN],
        })
    }

    /// 상태 초기화 — 듣기를 새로 시작할 때(마이크를 다시 껐을 때 등).
    pub fn reset(&mut self) {
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        self.pending.clear();
        self.window.fill(0.0);
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// mono f32 [-1,1] 를 아무 길이로 넣는다 — 이번 호출에서 끝난 청크들의 음성 확률 평균.
    /// 청크가 하나도 안 끝났으면 None(32ms 마다 하나 끝난다).
    pub fn process(&mut self, input: &[f32]) -> Result<Option<f32>, TaskError> {
        match &mut self.resampler {
            Some(r) => r.push(input, &mut self.pending),
            None => self.pending.extend_from_slice(input),
        }
        let (mut sum, mut count, mut start) = (0f32, 0usize, 0usize);
        while self.pending.len() - start >= CHUNK {
            // 앞 청크의 꼬리 64 를 머리로 옮기고 이번 청크를 잇는다 (래퍼의 _context)
            self.window.copy_within(CHUNK.., 0);
            self.window[CONTEXT..].copy_from_slice(&self.pending[start..start + CHUNK]);
            start += CHUNK;
            let outs = self.graph.run(vec![
                Tens::new(vec![1, CONTEXT + CHUNK], self.window.clone()),
                Tens::new(vec![1, HIDDEN], std::mem::take(&mut self.h)),
                Tens::new(vec![1, HIDDEN], std::mem::take(&mut self.c)),
            ])?;
            let mut outs = outs.into_iter();
            let prob = outs.next().map(|t| t.data[0]).unwrap_or(0.0);
            self.h = outs
                .next()
                .map(|t| t.data)
                .unwrap_or_else(|| vec![0.0; HIDDEN]);
            self.c = outs
                .next()
                .map(|t| t.data)
                .unwrap_or_else(|| vec![0.0; HIDDEN]);
            if !prob.is_finite() || self.c.iter().any(|v| !v.is_finite()) {
                // NaN 입력 한 번에 상태가 영구 오염되지 않게 — 처음부터 다시 듣는다
                self.reset();
                return Ok(Some(0.0));
            }
            sum += prob;
            count += 1;
        }
        self.pending.drain(..start);
        Ok((count > 0).then(|| sum / count as f32))
    }
}

/// 스트리밍 리샘플러 — 블랙만 창 sinc 저역통과(차단 = 낮은 쪽 나이퀴스트의 92%) 커널을
/// 위상 PHASES 개로 미리 깔아 둔다. 정수비(48k→16k)는 위상이 늘 0 이라 커널 한 벌만 쓴다.
struct Resampler {
    /// 출력 1샘플당 입력 진행량 (in/out)
    step: f64,
    /// 다음 출력의 중심 — hist 기준 입력 좌표
    pos: f64,
    hist: Vec<f32>,
    half: usize,
    /// [PHASES+1][2·half]
    table: Vec<f32>,
}

const PHASES: usize = 64;
/// 커널 반폭에 담을 sinc 영점 수
const ZERO_CROSSINGS: f64 = 8.0;

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        let step = from as f64 / to as f64;
        // 입력 샘플 기준 차단 주파수(cycles/sample)
        let fc = 0.5 * (to as f64 / from as f64).min(1.0) * 0.92;
        let half = (ZERO_CROSSINGS / (2.0 * fc)).ceil() as usize;
        let taps = 2 * half;
        let mut table = vec![0f32; (PHASES + 1) * taps];
        for p in 0..=PHASES {
            let frac = p as f64 / PHASES as f64;
            for j in 0..taps {
                // 탭 j 의 입력 위치 = floor(pos) − half + 1 + j → 중심과의 거리 t
                let t = (j as f64 - half as f64 + 1.0) - frac;
                let x = 2.0 * fc * t;
                let sinc = if x.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let u = (t / half as f64).clamp(-1.0, 1.0); // 창 [-1,1]
                let a = std::f64::consts::PI * (u + 1.0);
                let blackman = 0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos();
                table[p * taps + j] = (2.0 * fc * sinc * blackman) as f32;
            }
        }
        Resampler {
            step,
            pos: 0.0,
            hist: Vec::with_capacity(4096),
            half,
            table,
        }
    }

    fn reset(&mut self) {
        self.pos = 0.0;
        self.hist.clear();
    }

    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.hist.extend_from_slice(input);
        let taps = 2 * self.half;
        loop {
            let base = self.pos.floor();
            // 이 출력이 쓰는 마지막 입력이 아직 안 왔으면 다음 호출로
            if base as usize + self.half >= self.hist.len() {
                break;
            }
            let phase = ((self.pos - base) * PHASES as f64).round() as usize;
            let coeffs = &self.table[phase * taps..(phase + 1) * taps];
            let k0 = base as isize - self.half as isize + 1;
            let mut acc = 0f32;
            for (j, &w) in coeffs.iter().enumerate() {
                let k = k0 + j as isize;
                if k >= 0 {
                    acc += self.hist[k as usize] * w;
                }
            }
            out.push(acc);
            self.pos += self.step;
        }
        // 다음 출력이 더는 안 볼 앞부분을 버린다
        let keep_from = (self.pos.floor() as isize - self.half as isize + 1).max(0) as usize;
        if keep_from > 0 {
            self.hist.drain(..keep_from.min(self.hist.len()));
            self.pos -= keep_from as f64;
        }
    }
}
