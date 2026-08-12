//! 활성화 함수의 WGSL vec4 표현식 표.
//!
//! CPU 레퍼런스(ai_core::Activation::apply)와 수식이 정확히 일치해야 한다
//! (hardswish/hardsigmoid는 PyTorch/MNv3 규약: x/6 + 0.5).

use ai_core::Activation;

/// `var`(vec4f 변수명)에 활성화를 적용하는 표현식을 반환
pub fn act_expr(act: Activation, var: &str) -> String {
    match act {
        Activation::None => var.to_string(),
        Activation::Relu => format!("max({var}, vec4f(0.0))"),
        Activation::Sigmoid => format!("(vec4f(1.0) / (vec4f(1.0) + exp(-{var})))"),
        Activation::Tanh => format!("tanh({var})"),
        Activation::Hardswish => {
            format!("({var} * clamp({var} / 6.0 + 0.5, vec4f(0.0), vec4f(1.0)))")
        }
        Activation::Hardsigmoid => {
            format!("clamp({var} / 6.0 + 0.5, vec4f(0.0), vec4f(1.0))")
        }
        Activation::Clamp01 => format!("clamp({var}, vec4f(0.0), vec4f(1.0))"),
    }
}

/// 캐시 키·테스트 그리드용 전체 활성화 목록
pub const ALL: [Activation; 7] = [
    Activation::None,
    Activation::Relu,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Hardswish,
    Activation::Hardsigmoid,
    Activation::Clamp01,
];
