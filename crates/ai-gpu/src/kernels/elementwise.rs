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
use crate::kernels::common::source::{self, SrcView};
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
    /// 피연산자 A/B/Z가 백킹 텐서의 채널 구간(뷰)일 때의 재매핑. 전부 NONE이면
    /// 평범한 스트림 연산 그대로 (`A[i]`), 뷰가 있으면 그 피연산자만 인덱스를 푼다.
    pub views: [SrcView; 3],
    /// 출력 텐서의 채널그룹 수 — 뷰가 있을 때만 인덱스 분해에 쓰인다
    pub out_cg: u32,
}

impl ElementwiseSpec {
    /// 뷰 없는 평범한 스트림 연산
    pub fn plain(op: BinaryOp, operand: EwOperand, act: Activation, len_vec4: u32, dt: DType) -> Self {
        Self { op, operand, act, len_vec4, dt, views: [SrcView::NONE; 3], out_cg: 0 }
    }

    /// 피연산자 슬롯 하나의 읽기 식. 뷰가 아니면 `NAME[i]`,
    /// 뷰면 행(pix)을 백킹 stride로 다시 잡는다.
    fn read(&self, name: &str, slot: usize) -> String {
        let v = self.views[slot];
        if !v.is_offset() {
            return format!("{name}[i]");
        }
        let (cg, stride, off) = (self.out_cg, v.stride_cg(), v.off_cg());
        debug_assert!(cg > 0, "뷰 피연산자는 out_cg가 필요하다");
        format!("{name}[(i / {cg}u) * {stride}u + {off}u + i % {cg}u]")
    }
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
        format!(
            "ew op={} mode={mode} act={} dt={}{}",
            self.op.tag(),
            self.act.tag(),
            self.dt.tag(),
            // 뷰가 있으면 인덱스 식이 달라지므로 out_cg까지 키에 들어가야 한다
            if self.views.iter().any(|v| v.is_offset()) {
                format!("{} cg{}", source::key_of(&self.views), self.out_cg)
            } else {
                String::new()
            }
        )
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
        let (a, b, z) = (self.read("A", 0), self.read("B", 1), self.read("Z", 2));
        let av = format!("vec4f({a})");
        let mut body = match self.operand {
            EwOperand::Tensor => {
                format!("var v = {};\n", self.op.wgsl_expr(&av, &format!("vec4f({b})")))
            }
            EwOperand::Scalar { scalar_first } => {
                let s = "vec4f(P.scalar)";
                if scalar_first {
                    format!("var v = {};\n", self.op.wgsl_expr(s, &av))
                } else {
                    format!("var v = {};\n", self.op.wgsl_expr(&av, s))
                }
            }
            // 채널 벡터 B는 [1,1,C]라 뷰 대상이 아니다 (재매핑 없음)
            EwOperand::ChannelVector => {
                format!("var v = {};\n", self.op.wgsl_expr(&av, "vec4f(B[i % P.cg])"))
            }
            EwOperand::Unary => format!("var v = {av};\n"),
            EwOperand::Mix => format!("var v = mix({av}, vec4f({b}), vec4f({z}));\n"),
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
                    let spec = ElementwiseSpec::plain(op, operand, act, 1024, DType::F32);
                    validate_wgsl(&spec.wgsl(&caps));
                    // 뷰 피연산자 변형: A/B/Z 전부 백킹의 채널 구간
                    let viewed = ElementwiseSpec {
                        views: [SrcView::view(16, 48, 16); 3],
                        out_cg: 4,
                        ..spec
                    };
                    validate_wgsl(&viewed.wgsl(&caps));
                }
            }
        }
    }
}
