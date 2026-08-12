//! 패스 프레임워크 — 패스당 파일 하나, 고정 순서 실행, 리라이트 수 보고.

pub mod canon;
pub mod check_support;
pub mod dce;
pub mod detect_cvec;
pub mod fold_bn;
pub mod fold_constants;
pub mod fuse_act;
pub mod fuse_concat;
pub mod fuse_mix;
pub mod fuse_residual;
pub mod fuse_se;
pub mod resolve_static;

use crate::error::ConvertError;
use crate::ir::Graph;

#[derive(Default, Debug)]
pub struct PassReport {
    pub rewrites: usize,
}

/// 변환 옵션 (CLI에서 채움)
#[derive(Default, Clone, Debug)]
pub struct Ctx {
    /// (w, h) — 이미지 입력 강제 크기. None이면 모델의 정적 shape 사용.
    pub size: Option<(u32, u32)>,
    /// 스칼라 입력 고정 (downsample_ratio=1.0)
    pub set_inputs: Vec<(String, f32)>,
    /// 순환 상태 쌍 (입력명, 출력명)
    pub states: Vec<(String, String)>,
    pub fp16: bool,
    pub strip_refiner: bool,
}

pub type Pass = fn(&mut Graph, &Ctx) -> Result<PassReport, ConvertError>;

/// 정규화까지의 파이프라인
pub fn run_to_canon(g: &mut Graph, ctx: &Ctx) -> Result<(), ConvertError> {
    resolve_static::run(g, ctx)?;
    dce::run(g, ctx)?;
    fold_constants::run(g, ctx)?;
    fold_bn::run(g, ctx)?;
    canon::run_all(g, ctx)?;
    dce::run(g, ctx)?;
    Ok(())
}

/// 전체 파이프라인 (canonicalize + 융합 + 지원성 검사)
pub fn run_full(g: &mut Graph, ctx: &Ctx) -> Result<(), ConvertError> {
    run_to_canon(g, ctx)?;
    fuse_act::run(g, ctx)?;
    fuse_residual::run(g, ctx)?;
    detect_cvec::run(g, ctx)?; // scalar/cvec attr 정규화 — fuse_mix가 (1-z) 인식에 의존
    fuse_mix::run(g, ctx)?;
    fuse_concat::run(g, ctx)?;
    // fuse_se는 비활성 — 실측: segate 1워크그룹 직렬(평균 256µs/op)이 비융합 체인
    // (~35µs, gpool+pw+pw 3디스패치)의 6배 퇴행. 디스패치 절약(~16µs)보다 병렬성
    // 손실이 압도적. webgl2도 SE를 개별 드로우로 남긴 이유. 커널/포맷은 보존.
    dce::run(g, ctx)?;
    check_support::run(g, ctx)?;
    Ok(())
}
