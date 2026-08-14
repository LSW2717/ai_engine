//! (h=1) W↔C 2D 전치 커널 — in(1,w,c) → out(1,c,w). C4 레인 재배치라 실복사
//! (MLP-Mixer 토큰↔채널, face_blendshapes). flatten.rs와 같은 골격.

use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::sv4_alias;
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/transpose.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct TransposeSpec {
    /// 입력 w (= 출력 채널 수)
    pub w: u32,
    /// 입력 c (= 출력 픽셀 수)
    pub c: u32,
    pub dt: DType,
}

impl TransposeSpec {
    fn groups(&self) -> u32 {
        self.c * self.w.div_ceil(4)
    }
}

impl KernelSpec for TransposeSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!("transpose w{} c{} dt={}", self.w, self.c, self.dt.tag())
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let consts = format!(
            "const W: u32 = {}u;\nconst C: u32 = {}u;\nconst CGIN: u32 = {}u;\nconst CGOUT: u32 = {}u;",
            self.w,
            self.c,
            self.c.div_ceil(4),
            self.w.div_ceil(4)
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
        for (w, c) in [(97u32, 64u32), (64, 97), (2, 146), (5, 3)] {
            validate_wgsl(&TransposeSpec { w, c, dt: DType::F32 }.wgsl(&caps));
        }
    }
}
