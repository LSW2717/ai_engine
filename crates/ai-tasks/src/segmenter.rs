//! 세그멘테이션 태스크 — 모델 수명·프레임 루프·프레임타임 통계를 소유한다.
//!
//! 바인딩(`ai-wasm`/`ai-ffi`)은 이 타입을 감싸기만 한다. 프레임 한 장의 순서
//! (업로드 → 추론 제출 → (호스트가 합성) → 완료 대기 → 기록)가 여기 한 곳에만
//! 있어야 웹과 모바일이 같은 동작을 한다.

use ai_core::TensorDesc;
use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

use crate::clock::{FrameClock, Stats};
use crate::error::TaskError;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub struct Segmenter {
    model: Model,
    clock: FrameClock,
    /// `infer()` 제출 시각 — `finish_frame()`이 여기서부터 잰다
    mark: Option<Instant>,
}

impl Segmenter {
    pub async fn load(ctx: &GpuContext, bytes: &[u8]) -> Result<Self, TaskError> {
        let model = Model::load(ctx, bytes).await?;
        Ok(Self { model, clock: FrameClock::new(), mark: None })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// 그래프의 첫 입력 이름 (대부분 모델이 입력 1개다)
    pub fn input_name(&self) -> &str {
        &self.model.sw.tensors[self.model.sw.inputs[0] as usize].name
    }

    /// 디바이스 유실이면 DeviceLost — 호스트 강등 판정의 근거라 프레임 경로
    /// 진입마다 확인한다 (정상 경로 비용은 원자 로드 1회).
    fn ensure_device(ctx: &GpuContext) -> Result<(), TaskError> {
        match ctx.lost_reason() {
            Some(msg) => Err(TaskError::DeviceLost(msg)),
            None => Ok(()),
        }
    }

    /// 논리 NHWC f32 프레임 업로드
    pub fn upload(&self, ctx: &GpuContext, rgb: &[f32]) -> Result<(), TaskError> {
        Self::ensure_device(ctx)?;
        let name = self.input_name().to_string();
        self.model.upload_input(ctx, &name, rgb)?;
        Ok(())
    }

    /// 추론 **제출**만 한다 (대기 없음). 합성은 호출자가 이 뒤에 붙인다.
    pub async fn infer(&mut self, ctx: &GpuContext) -> Result<(), TaskError> {
        Self::ensure_device(ctx)?;
        self.mark = Some(Instant::now());
        self.model.infer(ctx).await?;
        Ok(())
    }

    /// 이 프레임의 GPU 작업이 끝날 때까지 기다리고 프레임타임을 기록한다.
    ///
    /// ⚠ **프레임마다 반드시 불러야 한다.** 제출만 하고 안 기다리면 큐가 무한정
    /// 쌓여 화면이 과거 프레임을 보여주고(마스크가 뒤처짐), 뒤에 도는 벤치의
    /// 대기가 밀린 큐 전체를 기다려 측정값이 계속 커진다. 사파리에서 실제로
    /// 그랬다 (추론 3.13 → 10.00ms 단조 증가).
    pub async fn finish_frame(&mut self, ctx: &GpuContext) -> Result<(), TaskError> {
        Self::ensure_device(ctx)?;
        ai_gpu::readback::wait_idle(ctx).await.map_err(TaskError::Gpu)?;
        if let Some(t0) = self.mark.take() {
            self.clock.record(t0.elapsed().as_secs_f64() as f32 * 1e3);
        }
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        self.clock.stats()
    }

    /// 출력 텐서가 실제로 들어있는 스토리지 버퍼 + desc (리드백 없이 합성용)
    pub fn output_storage(&self, name: &str) -> Option<(&wgpu::Buffer, TensorDesc)> {
        self.model.output_storage(name)
    }

    /// 출력을 CPU로 읽는다 (진단·폴백 경로용 — 프레임 루프에서는 쓰지 말 것)
    pub async fn read_output(
        &self,
        ctx: &GpuContext,
        name: &str,
    ) -> Result<Vec<f32>, TaskError> {
        Ok(self.model.read_output(ctx, name).await?)
    }
}
