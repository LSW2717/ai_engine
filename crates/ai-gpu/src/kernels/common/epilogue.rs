//! conv 계열 커널의 에필로그 방출기 — bias → activation → residual(활성화 후) 순.
//!
//! 이 순서는 CPU 레퍼런스(ai_core::reference::conv)와 WebGL2 엔진의 conv-tail
//! 융합 규약과 일치한다. 모든 conv 커널이 이 함수를 공유해 규약 불일치를 차단한다.

use ai_core::Activation;

use super::activation::act_expr;

/// `var`(vec4f 누산 변수)에 에필로그를 적용하는 문장들을 방출.
/// - `bias_expr`: 예: `"BIAS[ng]"` — None이면 bias 없음
/// - `residual_expr`: 예: `"RES[out_idx]"` — None이면 residual 없음
pub fn emit(
    var: &str,
    bias_expr: Option<&str>,
    act: Activation,
    residual_expr: Option<&str>,
) -> String {
    let mut s = String::new();
    if let Some(b) = bias_expr {
        s.push_str(&format!("{var} = {var} + {b};\n"));
    }
    if act != Activation::None {
        s.push_str(&format!("{var} = {};\n", act_expr(act, var)));
    }
    if let Some(r) = residual_expr {
        s.push_str(&format!("{var} = {var} + {r};\n"));
    }
    s
}

/// 캐시 키 조각: `bias`/`nobias`, act 태그, `res`/`nores`
pub fn key_fragment(bias: bool, act: Activation, residual: bool) -> String {
    format!(
        "{} act={} {}",
        if bias { "bias" } else { "nobias" },
        act.tag(),
        if residual { "res" } else { "nores" }
    )
}
