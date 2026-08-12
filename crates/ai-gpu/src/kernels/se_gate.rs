//! SE 게이트 커널 — gpool→FC1[→FC2] 융합 (fuse_se 패스의 lowering 대상).
//!
//! 워크그룹 하나(256스레드)가 채널 평균 → FC 체인을 공유메모리로 잇는다.
//! FC 가중치는 pack_weights_conv(kg_align=4)의 [kg][ng][j] — 1×1 conv과 동일.

use ai_core::{Activation, DType};

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::activation::act_expr;
use crate::kernels::common::sv4_alias;
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/se_gate.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct SeGateSpec {
    /// 입력 픽셀 수 (h*w)
    pub hw: u32,
    pub c_in: u32,
    pub c_mid: u32,
    pub act1: Activation,
    /// (c_out, act2) — None이면 FC1 출력이 곧 게이트
    pub fc2: Option<(u32, Activation)>,
    pub dt: DType,
}

impl SeGateSpec {
    fn cg_in(&self) -> u32 {
        self.c_in.div_ceil(4)
    }
    fn cg_mid(&self) -> u32 {
        self.c_mid.div_ceil(4)
    }
}

impl KernelSpec for SeGateSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "segate hw{} {}->{}{} act1={}{} dt={}",
            self.hw,
            self.c_in,
            self.c_mid,
            self.fc2.map(|(c, _)| format!("->{c}")).unwrap_or_default(),
            self.act1.tag(),
            self.fc2.map(|(_, a)| format!(" act2={}", a.tag())).unwrap_or_default(),
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let (cg_in, cg_mid) = (self.cg_in(), self.cg_mid());
        let mut binds = String::from(
            "@group(0) @binding(1) var<storage, read> IN: array<sv4>;\n\
             @group(0) @binding(2) var<storage, read> W1: array<sv4>;\n\
             @group(0) @binding(3) var<storage, read> B1: array<sv4>;\n",
        );
        let mut consts = format!(
            "const HW: u32 = {}u;\nconst CG_IN: u32 = {cg_in}u;\nconst CG_MID: u32 = {cg_mid}u;\nconst INV_HW: f32 = 1.0 / {}.0;",
            self.hw, self.hw
        );
        let mut sh = format!("var<workgroup> mean_sh: array<vec4f, {cg_in}>;");
        let act1 = if self.act1 != Activation::None {
            format!("v = {};", act_expr(self.act1, "v"))
        } else {
            String::new()
        };

        let (fc1_store, fc2_code) = match self.fc2 {
            None => {
                binds.push_str(
                    "@group(0) @binding(4) var<storage, read_write> OUT: array<sv4>;",
                );
                ("OUT[ng] = sv4(v);".to_string(), String::new())
            }
            Some((c_out, act2)) => {
                let cg_out = c_out.div_ceil(4);
                binds.push_str(
                    "@group(0) @binding(4) var<storage, read> W2: array<sv4>;\n\
                     @group(0) @binding(5) var<storage, read> B2: array<sv4>;\n\
                     @group(0) @binding(6) var<storage, read_write> OUT: array<sv4>;",
                );
                consts.push_str(&format!("\nconst CG_OUT: u32 = {cg_out}u;"));
                sh.push_str(&format!("\nvar<workgroup> mid_sh: array<vec4f, {cg_mid}>;"));
                let act2_line = if act2 != Activation::None {
                    format!("v = {};", act_expr(act2, "v"))
                } else {
                    String::new()
                };
                let fc2 = format!(
                    "workgroupBarrier();\n    for (var ng = t; ng < CG_OUT; ng = ng + 256u) {{\n        var acc = vec4f(0.0);\n        for (var kg = 0u; kg < CG_MID; kg = kg + 1u) {{\n            let a = mid_sh[kg];\n            let wb = (kg * CG_OUT + ng) * 4u;\n            acc = acc + vec4f(dot(vec4f(W2[wb]), a), dot(vec4f(W2[wb + 1u]), a), dot(vec4f(W2[wb + 2u]), a), dot(vec4f(W2[wb + 3u]), a));\n        }}\n        var v = acc + vec4f(B2[ng]);\n        {act2_line}\n        OUT[ng] = sv4(v);\n    }}"
                );
                ("mid_sh[ng] = v;".to_string(), fc2)
            }
        };

        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("BINDINGS", binds),
                ("CONSTS", consts),
                ("SH_DECL", sh),
                ("ACT1", act1),
                ("FC1_STORE", fc1_store),
                ("FC2", fc2_code),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        if self.fc2.is_some() {
            vec![
                StorageDir::Read,
                StorageDir::Read,
                StorageDir::Read,
                StorageDir::Read,
                StorageDir::Read,
                StorageDir::ReadWrite,
            ]
        } else {
            vec![StorageDir::Read, StorageDir::Read, StorageDir::Read, StorageDir::ReadWrite]
        }
    }

    fn workgroups(&self) -> [u32; 3] {
        [1, 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for dt in [DType::F32, DType::F16] {
            // RVM 대표 shape: 960→240→960, 단일 FC 128
            let two = SeGateSpec {
                hw: 144,
                c_in: 960,
                c_mid: 240,
                act1: Activation::Relu,
                fc2: Some((960, Activation::Hardsigmoid)),
                dt,
            };
            validate_wgsl(&two.wgsl(&caps));
            let one = SeGateSpec {
                hw: 144,
                c_in: 960,
                c_mid: 128,
                act1: Activation::Sigmoid,
                fc2: None,
                dt,
            };
            validate_wgsl(&one.wgsl(&caps));
        }
    }
}
