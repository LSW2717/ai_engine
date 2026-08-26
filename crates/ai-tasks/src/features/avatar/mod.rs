//! avatar — VRM 아바타 모드: 카메라 프레임을 **통째로 대체**하는 3D 아바타.
//! 얼굴 피팅(머리 회전)과 EAR(깜빡임)만 추적하는 의도적 최소 표정 스코프 —
//! 표정 블렌드셰이프 전체 추적은 하지 않는다 (설계 결정, 2026-08-26).
//!
//! 구성: vrm(파서) → rig(스켈레톤/blink 솔버) → renderer(스키닝+MToon 렌더).
//! 손 리타겟(21pt→손가락/팔 IK)과 스프링본 시뮬은 후속 단계 — rig 가 자리를
//! 갖고 있고 vrm 파서는 스프링본을 이미 읽는다.

pub mod renderer;
pub mod rig;
pub mod vrm;

pub use renderer::{AvatarBg, AvatarOverlay};
