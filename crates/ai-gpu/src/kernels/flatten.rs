//! flatten 실체화 커널 — (px, c) → (1, 1, px*c). chcopy의 지오메트리 변경판
//! (레인 재배치). MediaPipe 디텍터 헤드의 Reshape 경로가 표적.

use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::sv4_alias;
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/flatten.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct FlattenSpec {
    pub px: u32,
    pub c_in: u32,
    pub dt: DType,
}

impl FlattenSpec {
    fn n(&self) -> u32 {
        self.px * self.c_in
    }
}

impl KernelSpec for FlattenSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!("flatten px{} c{} dt={}", self.px, self.c_in, self.dt.tag())
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let consts = format!(
            "const N: u32 = {}u;\nconst CIN: u32 = {}u;\nconst CGIN: u32 = {}u;",
            self.n(),
            self.c_in,
            self.c_in.div_ceil(4)
        );
        fill(TEMPLATE, &[("TYPES", sv4_alias(self.dt)), ("CONSTS", consts)])
    }

    fn bindings(&self) -> Vec<StorageDir> {
        vec![StorageDir::Read, StorageDir::ReadWrite]
    }

    fn workgroups(&self) -> [u32; 3] {
        [self.n().div_ceil(4).div_ceil(256), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for (px, c) in [(256u32, 2u32), (256, 32), (1, 1434)] {
            validate_wgsl(&FlattenSpec { px, c_in: c, dt: DType::F32 }.wgsl(&caps));
        }
    }
}
