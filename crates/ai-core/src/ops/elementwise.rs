//! 원소별 이항 연산. Phase 2 체인 융합(GRU mix 등)의 이음새.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
}

impl BinaryOp {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
        }
    }

    /// WGSL 이항 연산자
    pub fn wgsl_op(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
        }
    }
}
