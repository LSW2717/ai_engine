//! 원소별 이항 연산. Phase 2 체인 융합(GRU mix 등)의 이음새.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// b는 채널별 slope — MediaPipe 랜드마크 계열 (a>0 ? a : a*b)
    Prelu,
}

impl BinaryOp {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Prelu => {
                if a > 0.0 {
                    a
                } else {
                    a * b
                }
            }
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Prelu => "prelu",
        }
    }

    /// WGSL 표현식 — a/b는 vec4f로 캐스팅된 조각 (infix가 아닌 op이 있어 함수형)
    pub fn wgsl_expr(self, a: &str, b: &str) -> String {
        match self {
            BinaryOp::Add => format!("{a} + {b}"),
            BinaryOp::Sub => format!("{a} - {b}"),
            BinaryOp::Mul => format!("{a} * {b}"),
            // max(a,0) + b*min(a,0) — 분기 없는 prelu
            BinaryOp::Prelu => {
                format!("max({a}, vec4f(0.0)) + {b} * min({a}, vec4f(0.0))")
            }
        }
    }
}
