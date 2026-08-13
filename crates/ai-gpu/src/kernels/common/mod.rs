//! 커널 codegen 공통 헬퍼 — 템플릿 슬롯 치환, 활성화 표, 에필로그 방출기.

pub mod activation;
pub mod epilogue;
pub mod gemm_tile;
pub mod source;
pub mod writer;

use ai_core::DType;

fn vec4_ty(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "vec4f",
        DType::F16 => "vec4<f16>",
    }
}

/// `//@TYPES` 슬롯 — 활성화(`sv4`)와 가중치(`wv4`) 스토리지 vec4 별칭.
/// 어느 쪽이 f16이든 누산은 항상 f32: 로드는 `vec4f(X[...])`, 스토어는 `sv4(v)`.
///
/// **정밀도가 두 축인 이유**: 저해상 심층 conv은 가중치 재사용이 픽셀 수 M으로 묶여
/// (9×16이면 M=144뿐) 가중치 페치가 대역을 지배한다. 가중치만 f16으로 낮추면
/// 트래픽이 절반이 되면서 활성화는 f32라 정확도 손실이 거의 없다.
/// (webgl2 엔진 실측: 전체 f16 rel 8.55e-3 게이트 초과 vs 가중치만 f16 1.72e-3 통과.
/// dw 가중치와 bias는 BN 접힘 다이내믹 레인지 때문에 f32로 남겨야 한다.)
pub fn type_aliases(act: DType, weights: DType) -> String {
    let mut s = String::new();
    if act == DType::F16 || weights == DType::F16 {
        s.push_str("enable f16;\n\n");
    }
    s.push_str(&format!("alias sv4 = {};\n", vec4_ty(act)));
    s.push_str(&format!("alias wv4 = {};", vec4_ty(weights)));
    s
}

/// 가중치를 따로 안 두는 커널(elementwise/resize/pool/gather)용 축약
pub fn sv4_alias(dt: DType) -> String {
    type_aliases(dt, dt)
}
