//! channel_gather 커널 — 채널 축 재배열의 만능 복사.
//!
//! parts = [(입력 채널 수, 입력 내 시작 채널)] 목록이 출력 채널 공간을 순서대로 채운다:
//! - Concat: parts = [(c0, 0), (c1, 0), ...] (각 입력의 전체 채널)
//! - Chcopy/비정렬 뷰: parts = [(n, src_c)] (한 입력의 부분 채널)
//! 출력 텍셀의 4개 lane을 한 스레드가 전부 계산하므로 파트 경계가 텍셀 중간에
//! 걸려도 RMW 해저드가 없다. 입력 최대 3개(RVM concat 최대 파트 수).

use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::{fill, W};
use crate::kernels::common::sv4_alias;

const TEMPLATE: &str = include_str!("shaders/channel_gather.wgsl");

#[derive(Clone, Debug)]
pub struct GatherPart {
    /// 이 파트가 출력에 기여하는 채널 수
    pub c: u32,
    /// 입력 텐서 내 시작 채널
    pub src_c: u32,
    /// 입력 텐서의 전체 채널 수 (스트라이드 계산용)
    pub in_c: u32,
}

#[derive(Clone, Debug)]
pub struct ChannelGatherSpec {
    /// 출력 픽셀 수 (h*w)
    pub px: u32,
    /// 출력 채널 수
    pub c_out: u32,
    pub parts: Vec<GatherPart>,
    pub dt: DType,
}

impl ChannelGatherSpec {
    fn cgo(&self) -> u32 {
        self.c_out.div_ceil(4)
    }
}

impl KernelSpec for ChannelGatherSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        let parts: Vec<String> = self
            .parts
            .iter()
            .map(|p| format!("{}@{}/{}", p.c, p.src_c, p.in_c))
            .collect();
        format!(
            "chgather px{} c{} [{}] dt={}",
            self.px,
            self.c_out,
            parts.join(","),
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        assert!(self.parts.len() <= 3, "channel_gather 파트 3개 초과");
        let mut ins = W::new();
        for (i, _) in self.parts.iter().enumerate() {
            ins.line(format!(
                "@group(0) @binding({}) var<storage, read> IN{i}: array<sv4>;",
                i + 1
            ));
        }
        let out_b = format!(
            "@group(0) @binding({}) var<storage, read_write> OUT: array<sv4>;",
            self.parts.len() + 1
        );
        let consts = format!(
            "const TOTAL: u32 = {}u;\nconst CGO: u32 = {}u;",
            self.px * self.cgo(),
            self.cgo()
        );

        // lane별: 출력 채널 oc → 파트 경계 if-체인 (경계는 codegen 상수)
        let mut lanes = W::new();
        for lane in 0..4u32 {
            lanes.line(format!("var l{lane} = 0.0;"));
            lanes.line(format!("{{ let oc = oc0 + {lane}u;"));
            let mut base = 0u32;
            for (i, p) in self.parts.iter().enumerate() {
                let end = base + p.c;
                let kw = if i == 0 { "if" } else { "else if" };
                let cg_in = p.in_c.div_ceil(4);
                lanes.line(format!("  {kw} (oc < {end}u) {{"));
                lanes.line(format!("    let ic = oc - {base}u + {}u;", p.src_c));
                lanes.line(format!(
                    "    l{lane} = vec4f(IN{i}[px * {cg_in}u + ic / 4u])[ic % 4u];"
                ));
                lanes.line("  }");
                base = end;
            }
            lanes.line("}");
        }

        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("IN_BINDINGS", ins.done()),
                ("OUT_BINDING", out_b),
                ("CONSTS", consts),
                ("LANES", lanes.done()),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        let mut b = vec![StorageDir::Read; self.parts.len()];
        b.push(StorageDir::ReadWrite);
        b
    }

    fn workgroups(&self) -> [u32; 3] {
        [(self.px * self.cgo()).div_ceil(256), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        // concat [3,1] (텍셀 중간 경계), 3파트, chcopy 형태
        let cases = vec![
            ChannelGatherSpec {
                px: 144 * 256,
                c_out: 4,
                parts: vec![
                    GatherPart { c: 3, src_c: 0, in_c: 3 },
                    GatherPart { c: 1, src_c: 0, in_c: 1 },
                ],
                dt: DType::F32,
            },
            ChannelGatherSpec {
                px: 100,
                c_out: 128,
                parts: vec![
                    GatherPart { c: 80, src_c: 0, in_c: 80 },
                    GatherPart { c: 40, src_c: 0, in_c: 40 },
                    GatherPart { c: 8, src_c: 0, in_c: 8 },
                ],
                dt: DType::F32,
            },
            ChannelGatherSpec {
                px: 37,
                c_out: 1,
                parts: vec![GatherPart { c: 1, src_c: 3, in_c: 4 }],
                dt: DType::F16,
            },
        ];
        for spec in cases {
            validate_wgsl(&spec.wgsl(&caps));
        }
    }
}
