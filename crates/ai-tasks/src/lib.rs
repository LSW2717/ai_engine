//! ai-tasks — **공개 API 본체**. 태스크 단위(세그멘테이션/랜드마크/노이즈 억제)와
//! 그 전·후처리, 합성, 폴백 정책이 여기 산다.
//!
//! `ai-wasm`(웹)과 `ai-ffi`(모바일)는 이 크레이트를 **타입 변환만** 해서 노출한다.
//! 바인딩 크레이트에 분기(`if`)가 생겼다면 그건 로직이고 여기로 내려와야 한다 —
//! 그 규칙이 깨지는 순간 웹과 모바일 동작이 갈라진다. 실제로 지금 랜드마크가
//! 웹(mediapipe-wasm)과 모바일(vcxrust_ai/ncnn) 두 벌로 갈라져 있고, 그 실수를
//! 한 층 아래에서 반복하지 않으려고 이 크레이트를 둔다.
//!
//! 플랫폼마다 **진짜** 다른 것만 바인딩에 남긴다:
//! 서피스 획득, 프레임 임포트, 스레드 모델, 모델 바이트 조달.

// 폴더 = 제품 기능(소비하는 데이터 기준), 파일 = 개념 하나 (사용자 확정 규칙):
//   session/ 모델 인스턴스 · detect/ 얼굴·손 공용 디텍터 수학
//   features/vb/ 가상배경(마스크 소비: 배경·블러·조명·프레이밍·밝기)
//   features/face/ 얼굴(랜드마크 소비: 화장·터치업·3D·아바타 예정)
//   (예정) hand/ gaze/ audio/
pub mod detect;
pub mod error;
pub mod features;
pub mod session;

// 공개 표면은 재수출로 고정 — 내부 재배치가 바인딩·테스트를 깨지 않는다
pub use detect::{Detection, DetectorPost};
pub use error::TaskError;
pub use features::face::{FaceResult, FaceTask};
pub use session::clock::Stats;
pub use session::cpu::CpuSession;
pub use session::gpu::GpuSession;
pub use session::pool::Pool;
pub use features::vb::compositor::{CompositeOpts, Compositor};
