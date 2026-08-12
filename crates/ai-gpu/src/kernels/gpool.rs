//! global average pool 커널 — SE 블록 입구, [H,W,C] → [1,1,C] 벡터 텐서.

use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/gpool.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct GpoolSpec {
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub dt: DType,
}

impl GpoolSpec {
    fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }
}

impl KernelSpec for GpoolSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!("gpool {}x{} c{} dt={}", self.h, self.w, self.c, self.dt.tag())
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let hw = self.h * self.w;
        let consts = format!(
            "const HW: u32 = {hw}u;\nconst CG: u32 = {}u;\nconst INV_HW: f32 = 1.0 / {hw}.0;",
            self.cg()
        );
        fill(
            TEMPLATE,
            &[
                ("CONSTS", consts),
                (
                    "OUT_BINDING",
                    "@group(0) @binding(2) var<storage, read_write> OUT: array<vec4f>;".to_string(),
                ),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        vec![StorageDir::Read, StorageDir::ReadWrite]
    }

    fn workgroups(&self) -> [u32; 3] {
        [self.cg(), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for (h, w, c) in [(9, 16, 576), (72, 128, 16), (3, 5, 6)] {
            validate_wgsl(&GpoolSpec { h, w, c, dt: DType::F32 }.wgsl(&caps));
        }
    }
}
