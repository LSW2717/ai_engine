//! RNNoise 스트리밍 노이즈 제거 — nnnoiseless(RNNoise 2017 가중치의 순수 Rust 포트).
//!
//! 기본 잡음제거 모드용. 강력 모드(Enhancer)보다 훨씬 가볍다 (네이티브 프레임당 ~0.05ms).
//! 48kHz 전용 — 가중치가 그 레이트로 학습됐다. 다른 레이트는 바인딩이 passthrough 로 거른다.
//!
//! 480 샘플(10ms) mono f32 [-1,1]를 넣으면 480 샘플이 나온다. 잡음 제거 전용 —
//! 라이브러리가 함께 내는 음성 확률은 버린다(음성 감지는 Silero VAD, vad.rs).
//! nnnoiseless 는 i16 범위 float 를 받으므로 스케일만 여기서 맞춘다 — 바인딩에는
//! Enhancer 와 같은 [-1,1] 계약을 노출한다.

use nnnoiseless::DenoiseState;

/// 처리 가능한 유일한 레이트
pub const SAMPLE_RATE: u32 = 48_000;

const FRAME: usize = DenoiseState::FRAME_SIZE;
const I16_SCALE: f32 = 32768.0;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    // scratch (i16 범위)
    inp: [f32; FRAME],
    out: [f32; FRAME],
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser {
    /// 가중치는 크레이트에 내장 — 모델 파일이 없다.
    pub fn new() -> Self {
        Denoiser { state: DenoiseState::new(), inp: [0.0; FRAME], out: [0.0; FRAME] }
    }

    /// process_frame 호출당 기대 샘플 수 (480 = 10ms@48k)
    pub fn frame_len(&self) -> usize {
        FRAME
    }

    /// 스트리밍 상태 초기화 (GRU·피치 버퍼 — 스트림 파기 시)
    pub fn reset(&mut self) {
        self.state = DenoiseState::new();
    }

    /// input/output 각 frame_len() 샘플. 부족하면 0 패딩/절단.
    /// 첫 프레임은 페이드인 잔향이 섞인다 (라이브러리 특성 — 스트림 시작 10ms).
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(FRAME);
        for (d, s) in self.inp[..n].iter_mut().zip(input) {
            *d = s * I16_SCALE;
        }
        self.inp[n..].fill(0.0);
        self.state.process_frame(&mut self.out, &self.inp);
        let out_n = output.len().min(FRAME);
        for (d, s) in output[..out_n].iter_mut().zip(&self.out) {
            *d = s / I16_SCALE;
        }
        output[out_n..].fill(0.0);
    }
}
