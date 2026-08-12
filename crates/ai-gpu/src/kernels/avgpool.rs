//! average pool 커널 — RVM 디코더의 다운샘플 경로 (k==s, pad=0가 주 사용처).

use ai_core::ops::AvgPool2d;
use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::{fill, W};

const TEMPLATE: &str = include_str!("shaders/avgpool.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct AvgPoolSpec {
    pub ih: u32,
    pub iw: u32,
    pub c: u32,
    pub k: u32,
    pub s: u32,
    pub pad: [u32; 4],
    pub dt: DType,
}

impl AvgPoolSpec {
    pub fn from_op(op: &AvgPool2d, ih: u32, iw: u32, c: u32, dt: DType) -> Self {
        assert_eq!(op.kh, op.kw, "정방 커널만 지원");
        assert_eq!(op.sh, op.sw, "등방 stride만 지원");
        Self { ih, iw, c, k: op.kh, s: op.sh, pad: op.pad, dt }
    }

    pub fn out_hw(&self) -> (u32, u32) {
        let oh = (self.ih + self.pad[0] + self.pad[2] - self.k) / self.s + 1;
        let ow = (self.iw + self.pad[1] + self.pad[3] - self.k) / self.s + 1;
        (oh, ow)
    }

    fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }
}

impl KernelSpec for AvgPoolSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "avgpool {}x{} c{} k{} s{} p{:?} dt={}",
            self.ih, self.iw, self.c, self.k, self.s, self.pad, self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let (oh, ow) = self.out_hw();
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {oh}u;\nconst OW: u32 = {ow}u;\nconst CG: u32 = {}u;\nconst INV_K: f32 = 1.0 / {}.0;",
            self.ih,
            self.iw,
            self.cg(),
            self.k * self.k
        );
        let mut body = W::new();
        for ky in 0..self.k {
            for kx in 0..self.k {
                body.line(format!(
                    "{{ let iy = i32(oy) * {} + {} - {};",
                    self.s, ky, self.pad[0]
                ));
                body.line(format!(
                    "  let ix = i32(ox) * {} + {} - {};",
                    self.s, kx, self.pad[1]
                ));
                body.line("  let m = select(0.0, 1.0, iy >= 0 && iy < IH && ix >= 0 && ix < IW);");
                body.line("  let cy = u32(clamp(iy, 0, IH - 1));");
                body.line("  let cx = u32(clamp(ix, 0, IW - 1));");
                body.line("  acc = acc + IN[(cy * u32(IW) + cx) * CG + cg] * m; }");
            }
        }
        fill(
            TEMPLATE,
            &[
                ("CONSTS", consts),
                ("TAPS_UNROLLED", body.done()),
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
        let (oh, ow) = self.out_hw();
        [ow.div_ceil(8), oh.div_ceil(8), self.cg()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for (k, s) in [(2u32, 2u32), (3, 3), (2, 1)] {
            let spec = AvgPoolSpec {
                ih: 18,
                iw: 32,
                c: 40,
                k,
                s,
                pad: [0; 4],
                dt: DType::F32,
            };
            validate_wgsl(&spec.wgsl(&caps));
        }
    }
}
