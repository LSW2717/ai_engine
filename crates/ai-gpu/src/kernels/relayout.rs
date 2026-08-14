//! 논리 순서 보존 desc 재배치 커널 — C4 레인 재패킹 (flatten의 일반화,
//! transpose.rs와 같은 골격). tf2onnx 언팩/팩 전치(face_blendshapes)가 표적.

use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::sv4_alias;
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/relayout.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct RelayoutSpec {
    pub px_in: u32,
    pub c_in: u32,
    pub px_out: u32,
    pub c_out: u32,
    pub dt: DType,
}

impl RelayoutSpec {
    fn groups(&self) -> u32 {
        self.px_out * self.c_out.div_ceil(4)
    }
}

impl KernelSpec for RelayoutSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "relayout {}x{}->{}x{} dt={}",
            self.px_in,
            self.c_in,
            self.px_out,
            self.c_out,
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let consts = format!(
            "const CI: u32 = {}u;\nconst CO: u32 = {}u;\nconst CGI: u32 = {}u;\nconst CGO: u32 = {}u;\nconst PXO: u32 = {}u;",
            self.c_in,
            self.c_out,
            self.c_in.div_ceil(4),
            self.c_out.div_ceil(4),
            self.px_out
        );
        fill(TEMPLATE, &[("TYPES", sv4_alias(self.dt)), ("CONSTS", consts)])
    }

    fn bindings(&self) -> Vec<StorageDir> {
        vec![StorageDir::Read, StorageDir::ReadWrite]
    }

    fn workgroups(&self) -> [u32; 3] {
        [self.groups().div_ceil(256), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for (pi, ci, po, co) in [(97u32, 64u32, 97 * 64, 1u32), (97 * 64, 1, 97, 64)] {
            validate_wgsl(
                &RelayoutSpec { px_in: pi, c_in: ci, px_out: po, c_out: co, dt: DType::F32 }
                    .wgsl(&caps),
            );
        }
    }
}
