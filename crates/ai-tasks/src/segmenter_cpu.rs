//! CPU 세그멘테이션 태스크 — GPU 불가·프레임 미달 시 폴백 티어.
//!
//! GPU `Segmenter`와 같은 순서(입력 → 추론 → 통계)를 노출하되, CPU 추론은
//! 동기라 upload/infer/finish_frame 3단이 아니라 `infer()` 한 번이다.
//! 어떤 티어를 쓸지는 호스트가 정한다 (모델 바이트 조달이 호스트 몫이라 —
//! 폴백은 더 가벼운 모델(R11 등)을 CPU로 싣는 것까지가 한 묶음).

use ai_cpu::CpuModel;

use crate::clock::{FrameClock, Stats};
use crate::error::TaskError;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub struct CpuSegmenter {
    model: CpuModel,
    clock: FrameClock,
}

impl CpuSegmenter {
    pub fn load(bytes: &[u8]) -> Result<Self, TaskError> {
        let model = CpuModel::load(bytes).map_err(|e| TaskError::Cpu(e.to_string()))?;
        Ok(Self { model, clock: FrameClock::new() })
    }

    pub fn model(&self) -> &CpuModel {
        &self.model
    }

    /// 그래프의 첫 입력 이름
    pub fn input_name(&self) -> &str {
        let sw = self.model.sw();
        &sw.tensors[sw.inputs[0] as usize].name
    }

    /// 스레드 폭 설정 (네이티브 전용 — 웹은 no-op, wasm 스레드는 별도 단계)
    pub fn set_threads(&mut self, n: usize) -> Result<(), TaskError> {
        self.model.set_threads(n).map_err(|e| TaskError::Cpu(e.to_string()))
    }

    /// 프레임 1장: 논리 NHWC 입력 → 추론 → 프레임타임 기록 (동기)
    pub fn infer_frame(&mut self, rgb: &[f32]) -> Result<(), TaskError> {
        let name = self.input_name().to_string();
        self.model
            .set_input(&name, rgb)
            .map_err(|e| TaskError::Cpu(e.to_string()))?;
        let t0 = Instant::now();
        self.model.infer().map_err(|e| TaskError::Cpu(e.to_string()))?;
        self.clock.record(t0.elapsed().as_secs_f64() as f32 * 1e3);
        Ok(())
    }

    pub fn read_output(&self, name: &str) -> Result<Vec<f32>, TaskError> {
        self.model.read_output(name).map_err(|e| TaskError::Cpu(e.to_string()))
    }

    pub fn stats(&self) -> Stats {
        self.clock.stats()
    }
}
