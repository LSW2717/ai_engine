//! CPU에 로드된 모델 인스턴스 — GPU 불가·강등 시의 백엔드 짝 (GpuSession과 대칭).
//! 세그·디텍터·랜드마크 구분 없이 이 타입 하나다.
//!
//! GPU와 같은 순서(입력 → 추론 → 통계)를 노출하되, CPU 추론은 동기라
//! upload/infer/finish_frame 3단이 아니라 `infer_frame()` 한 번이다.
//! 어떤 티어를 쓸지는 호스트가 정한다 (모델 바이트 조달이 호스트 몫이라 —
//! 폴백은 더 가벼운 모델(R11 등)을 CPU로 싣는 것까지가 한 묶음).

use crate::session::clock::{FrameClock, Stats};
use crate::detect::{Detection, DetectorPost};
use crate::error::TaskError;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub struct CpuSession {
    model: ai_cpu::Model,
    clock: FrameClock,
}

impl CpuSession {
    pub fn load(bytes: &[u8]) -> Result<Self, TaskError> {
        let model = ai_cpu::Model::load(bytes).map_err(|e| TaskError::Cpu(e.to_string()))?;
        Ok(Self { model, clock: FrameClock::new() })
    }

    pub fn model(&self) -> &ai_cpu::Model {
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

    /// 디텍터 프레임 1장: 레터박스된 입력(모델 크기, [-1,1]) → 추론 → 전 출력
    /// 디코드+NMS → **원본 프레임(src_w×src_h) 정규화 좌표** 검출로 반환.
    /// 레터박스 픽셀 채우기는 호스트 몫 — 기하는 `detect::letterbox`와 같아야 한다.
    pub fn detect(
        &mut self,
        post: &DetectorPost,
        rgb: &[f32],
        src_w: u32,
        src_h: u32,
    ) -> Result<Vec<Detection>, TaskError> {
        self.infer_frame(rgb)?;
        let sw = self.model.sw();
        let names: Vec<String> =
            sw.outputs.iter().map(|&o| sw.tensors[o as usize].name.clone()).collect();
        let outs: Vec<Vec<f32>> =
            names.iter().map(|n| self.read_output(n)).collect::<Result<_, _>>()?;
        let refs: Vec<&[f32]> = outs.iter().map(|v| v.as_slice()).collect();
        post.run_projected(&refs, src_w as f32, src_h as f32)
    }

    /// 재사용 버퍼로 출력 복사 — 프레임 루프에서 JS 이중 복사 제거용
    pub fn read_output_into(&self, name: &str, out: &mut Vec<f32>) -> Result<(), TaskError> {
        self.model.read_output_into(name, out).map_err(|e| TaskError::Cpu(e.to_string()))
    }

    pub fn stats(&self) -> Stats {
        self.clock.stats()
    }

    /// 스텝별 반복 벤치 passthrough — (라벨, 1회 평균 ms). 진단용 (cpu-ab 프로파일).
    pub fn bench_steps(&mut self, reps: usize) -> Vec<(String, f64)> {
        self.model.bench_steps(reps).into_iter().map(|r| (r.label, r.ms)).collect()
    }
}
