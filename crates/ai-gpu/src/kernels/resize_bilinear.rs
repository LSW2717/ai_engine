//! bilinear 리사이즈 커널 — half_pixel/asymmetric 좌표 변환은 codegen 플래그.

use ai_core::ops::CoordMode;
use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::fill;

const TEMPLATE: &str = include_str!("shaders/resize_bilinear.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct ResizeBilinearSpec {
    pub ih: u32,
    pub iw: u32,
    pub c: u32,
    pub oh: u32,
    pub ow: u32,
    pub mode: CoordMode,
    pub dt: DType,
}

impl ResizeBilinearSpec {
    fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }
}

impl KernelSpec for ResizeBilinearSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "resize {}x{}->{}x{} c{} {} dt={}",
            self.ih,
            self.iw,
            self.oh,
            self.ow,
            self.c,
            self.mode.tag(),
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {}u;\nconst OW: u32 = {}u;\nconst CG: u32 = {}u;\nconst SY: f32 = {}.0 / {}.0;\nconst SX: f32 = {}.0 / {}.0;",
            self.ih, self.iw, self.oh, self.ow, self.cg(), self.ih, self.oh, self.iw, self.ow
        );
        let coord = match self.mode {
            CoordMode::HalfPixel => {
                "let fy = (f32(oy) + 0.5) * SY - 0.5;\nlet fx = (f32(ox) + 0.5) * SX - 0.5;"
            }
            CoordMode::Asymmetric => "let fy = f32(oy) * SY;\nlet fx = f32(ox) * SX;",
        };
        fill(
            TEMPLATE,
            &[
                ("CONSTS", consts),
                ("COORD", coord.to_string()),
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
        [self.ow.div_ceil(8), self.oh.div_ceil(8), self.cg()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for mode in [CoordMode::HalfPixel, CoordMode::Asymmetric] {
            let spec = ResizeBilinearSpec {
                ih: 18,
                iw: 32,
                c: 40,
                oh: 36,
                ow: 64,
                mode,
                dt: DType::F32,
            };
            validate_wgsl(&spec.wgsl(&caps));
        }
    }
}
