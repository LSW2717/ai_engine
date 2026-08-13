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
        // exp(-x)는 x가 크게 음수여도 inf로 포화할 뿐이라 1/(1+inf)=0으로 안전하다.
        Activation::Sigmoid => format!("(vec4f(1.0) / (vec4f(1.0) + exp(-{var})))"),
        // ⚠ 입력 클램프 필수. f32에서 |x|>9면 tanh는 이미 ±1로 포화하는데,
        // 클램프 없이 백엔드 builtin에 큰 값을 넣으면 내부 exp(2x)가 오버플로해
        // 0이나 NaN이 나온다 (Metal 실측: tanh(43.78) → 0.0).
        // RVM ConvGRU의 candidate conv이 실제 이미지에서 이 구간에 들어가
        // 마스크가 무너졌다 — 랜덤 입력 테스트로는 활성화 전 값이 작아 안 드러난다.
        Activation::Tanh => format!("tanh(clamp({var}, vec4f(-9.0), vec4f(9.0)))"),
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
