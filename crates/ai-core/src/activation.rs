//! 활성화 함수 정의 — CPU 레퍼런스 적용과 WGSL codegen이 공유하는 열거형.
//!
//! hardswish/hardsigmoid는 PyTorch/MobileNetV3 규약(x/6 + 0.5)을 따른다.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    None,
    Relu,
    Sigmoid,
    Tanh,
    Hardswish,
    Hardsigmoid,
    /// clamp(x, 0, 1) — RVM 출력 등
    Clamp01,
    /// clamp(x, 0, 6) — MobileNetV2 계열 (MediaPipe hand_landmarks)
    Relu6,
    /// sqrt(max(x, 0)) — L2 norm·layernorm 분산 경로 (face_blendshapes)
    Sqrt,
    /// -x (blendshapes layernorm의 tf 분해 산물)
    Neg,
    /// 1/x — Div는 canon이 Recip+Mul로 분해한다
    Recip,
}

impl Activation {
    /// CPU 레퍼런스용 스칼라 적용
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Hardswish => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
            Activation::Hardsigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
            Activation::Clamp01 => x.clamp(0.0, 1.0),
            Activation::Relu6 => x.clamp(0.0, 6.0),
            Activation::Sqrt => x.max(0.0).sqrt(),
            Activation::Neg => -x,
            Activation::Recip => 1.0 / x,
        }
    }

    /// 캐시 키/라벨용 짧은 이름
    pub fn tag(self) -> &'static str {
        match self {
            Activation::None => "none",
            Activation::Relu => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Hardswish => "hswish",
            Activation::Hardsigmoid => "hsigmoid",
            Activation::Clamp01 => "clamp01",
            Activation::Relu6 => "relu6",
            Activation::Sqrt => "sqrt",
            Activation::Neg => "neg",
            Activation::Recip => "recip",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_activations_match_mnv3_convention() {
        // hardsigmoid(±3) = 0/1 경계, hardsigmoid(0) = 0.5
        assert_eq!(Activation::Hardsigmoid.apply(3.0), 1.0);
        assert_eq!(Activation::Hardsigmoid.apply(-3.0), 0.0);
        assert_eq!(Activation::Hardsigmoid.apply(0.0), 0.5);
        // hardswish(3) = 3, hardswish(-3) = 0, hardswish(1) = 1 * (1/6+0.5) = 2/3
        assert_eq!(Activation::Hardswish.apply(3.0), 3.0);
        assert_eq!(Activation::Hardswish.apply(-3.0), -0.0);
        assert!((Activation::Hardswish.apply(1.0) - 2.0 / 3.0).abs() < 1e-6);
    }
}
