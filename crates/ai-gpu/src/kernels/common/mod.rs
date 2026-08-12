//! 커널 codegen 공통 헬퍼 — 템플릿 슬롯 치환, 활성화 표, 에필로그 방출기.

pub mod activation;
pub mod epilogue;
pub mod gemm_tile;
pub mod writer;

use ai_core::DType;

/// `//@TYPES` 슬롯 — 스토리지 vec4 타입 별칭. f16이어도 누산은 항상 f32:
/// 로드는 `vec4f(X[...])`, 스토어는 `sv4(v)`로 변환한다.
pub fn sv4_alias(dt: DType) -> String {
    match dt {
        DType::F32 => "alias sv4 = vec4f;".to_string(),
        DType::F16 => "enable f16;\n\nalias sv4 = vec4<f16>;".to_string(),
    }
}
