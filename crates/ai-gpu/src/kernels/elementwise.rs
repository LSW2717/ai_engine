//! elementwise binary 커널 — add/sub/mul, tensor∘tensor 또는 tensor∘scalar, 활성화 융합.
//!
//! NHWC-C4 패킹 위에서 순수 vec4 스트림 연산이라 레이아웃 개념이 없다.
//! 패딩 채널 lane은 입력이 0이라 결과가 무의미해도 unpack에서 버려진다.
//! Phase 2의 elementwise 체인 융합(GRU mix 등)은 이 Spec의 BODY 슬롯 확장으로 들어온다.

use ai_core::ops::BinaryOp;
use ai_core::{Activation, DType};

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::activation::act_expr;
use crate::kernels::common::writer::fill;
use crate::kernels::common::sv4_alias;

const TEMPLATE: &str = include_str!("shaders/elementwise.wgsl");
pub const WORKGROUP: u32 = 256;

/// 두 번째 피연산자의 형태
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EwOperand {
    /// 같은 크기 tensor∘tensor
    Tensor,
    /// tensor∘scalar (값은 params.scalar — 값이 바뀌어도 재컴파일 없음)
    Scalar { scalar_first: bool },
    /// tensor∘vector([1,1,C] 채널 브로드캐스트, B 인덱스 = i % P.cg) — SE 스케일 경로
    ChannelVector,
    /// 단항 (op 무시, v = A[i] → act) — 단독 활성화 op의 lowering 대상
    Unary,
    /// GRU 갱신 mix(A,B,Z) = (1-Z)·A + Z·B (op 무시) — sub/mul/mul/add 4패스 융합
    Mix,
}

#[derive(Clone, Copy, Debug)]
pub struct ElementwiseSpec {
    pub op: BinaryOp,
    pub operand: EwOperand,
    pub act: Activation,
    /// 총 vec4 수 — 디스패치 계산에만 쓰이고 캐시 키/WGSL에는 안 들어감
    pub len_vec4: u32,
    pub dt: DType,
}

impl KernelSpec for ElementwiseSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        let mode = match self.operand {
            EwOperand::Tensor => "tt",
            EwOperand::Scalar { scalar_first: false } => "ts",
            EwOperand::Scalar { scalar_first: true } => "st",
            EwOperand::ChannelVector => "tv",
            EwOperand::Unary => "u",
            EwOperand::Mix => "mix",
        };
        format!("ew op={} mode={mode} act={} dt={}", self.op.tag(), self.act.tag(), self.dt.tag())
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let extra = match self.operand {
            EwOperand::Scalar { .. } | EwOperand::Unary => {
                "@group(0) @binding(2) var<storage, read_write> O: array<sv4>;".to_string()
            }
            EwOperand::Mix => "@group(0) @binding(2) var<storage, read> B: array<sv4>;\n\
                 @group(0) @binding(3) var<storage, read> Z: array<sv4>;\n\
                 @group(0) @binding(4) var<storage, read_write> O: array<sv4>;"
                .to_string(),
            _ => "@group(0) @binding(2) var<storage, read> B: array<sv4>;\n\
                 @group(0) @binding(3) var<storage, read_write> O: array<sv4>;"
                .to_string(),
        };
        let o = self.op.wgsl_op();
        let mut body = match self.operand {
            EwOperand::Tensor => format!("var v = vec4f(A[i]) {o} vec4f(B[i]);\n"),
            EwOperand::Scalar { scalar_first } => {
                let s = "vec4f(P.scalar)";
                if scalar_first {
                    format!("var v = {s} {o} vec4f(A[i]);\n")
                } else {
                    format!("var v = vec4f(A[i]) {o} {s};\n")
                }
            }
            EwOperand::ChannelVector => format!("var v = vec4f(A[i]) {o} vec4f(B[i % P.cg]);\n"),
            EwOperand::Unary => "var v = vec4f(A[i]);\n".to_string(),
            EwOperand::Mix => "var v = mix(vec4f(A[i]), vec4f(B[i]), vec4f(Z[i]));\n".to_string(),
        };
        if self.act != Activation::None {
            body.push_str(&format!("v = {};\n", act_expr(self.act, "v")));
        }
        fill(TEMPLATE, &[("TYPES", sv4_alias(self.dt)), ("EXTRA_BINDINGS", extra), ("BODY", body)])
    }

    fn bindings(&self) -> Vec<StorageDir> {
        match self.operand {
            EwOperand::Scalar { .. } | EwOperand::Unary => {
                vec![StorageDir::Read, StorageDir::ReadWrite]
            }
            EwOperand::Mix => {
                vec![StorageDir::Read, StorageDir::Read, StorageDir::Read, StorageDir::ReadWrite]
            }
            _ => vec![StorageDir::Read, StorageDir::Read, StorageDir::ReadWrite],
        }
    }

    fn workgroups(&self) -> [u32; 3] {
        let groups = self.len_vec4.div_ceil(WORKGROUP);
        assert!(groups < 65536, "elementwise: 텐서가 너무 큼 (groups={groups})");
        [groups, 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;
    use crate::kernels::common::activation::ALL;

    /// 전 변형 그리드의 WGSL이 naga 검증을 통과하는지 (GPU 불필요)
    #[test]
    fn naga_validates_all_variants() {
        let caps = crate::test_util::fake_caps();
        let operands = [
            EwOperand::Tensor,
            EwOperand::Scalar { scalar_first: false },
            EwOperand::Scalar { scalar_first: true },
            EwOperand::ChannelVector,
            EwOperand::Unary,
            EwOperand::Mix,
        ];
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for operand in operands {
                for act in ALL {
                    let spec =
                        ElementwiseSpec { op, operand, act, len_vec4: 1024, dt: DType::F32 };
                    validate_wgsl(&spec.wgsl(&caps));
                }
            }
        }
    }
}
