//! max pool 커널 — MediaPipe 랜드마크/디텍터 계열 (k2 s2가 주 사용처).
//! pad_c = 출력 채널 끝 제로패딩 (BlazeFace "MaxPool→Pad(C)→Add"의 Pad 접기).

use ai_core::ops::MaxPool2d;
use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::sv4_alias;
use crate::kernels::common::writer::{fill, W};

const TEMPLATE: &str = include_str!("shaders/maxpool.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct MaxPoolSpec {
    pub ih: u32,
    pub iw: u32,
    pub c: u32,
    pub k: u32,
    pub s: u32,
    pub pad: [u32; 4],
    pub pad_c: u32,
    pub dt: DType,
}

impl MaxPoolSpec {
    pub fn from_op(op: &MaxPool2d, ih: u32, iw: u32, c: u32, dt: DType) -> Self {
        assert_eq!(op.kh, op.kw, "정방 커널만 지원");
        assert_eq!(op.sh, op.sw, "등방 stride만 지원");
        Self { ih, iw, c, k: op.kh, s: op.sh, pad: op.pad, pad_c: op.pad_c, dt }
    }

    pub fn out_hw(&self) -> (u32, u32) {
        let oh = (self.ih + self.pad[0] + self.pad[2] - self.k) / self.s + 1;
        let ow = (self.iw + self.pad[1] + self.pad[3] - self.k) / self.s + 1;
        (oh, ow)
    }

    fn cg_in(&self) -> u32 {
        self.c.div_ceil(4)
    }

    fn ocg(&self) -> u32 {
        (self.c + self.pad_c).div_ceil(4)
    }
}

impl KernelSpec for MaxPoolSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "maxpool {}x{} c{} k{} s{} p{:?} pc{} dt={}",
            self.ih, self.iw, self.c, self.k, self.s, self.pad, self.pad_c,
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let (oh, ow) = self.out_hw();
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {oh}u;\nconst OW: u32 = {ow}u;\nconst OCG: u32 = {}u;\nconst CGIN: u32 = {}u;\nconst CF: f32 = {}.0;",
            self.ih,
            self.iw,
            self.ocg(),
            self.cg_in(),
            self.c
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
                body.line("  let ok = iy >= 0 && iy < IH && ix >= 0 && ix < IW;");
                body.line("  let cy = u32(clamp(iy, 0, IH - 1));");
                body.line("  let cx = u32(clamp(ix, 0, IW - 1));");
                body.line(
                    "  let v = select(vec4f(-3.0e38), vec4f(IN[(cy * u32(IW) + cx) * CGIN + cg]), ok);",
                );
                body.line("  acc = max(acc, v); }");
            }
        }
        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("CONSTS", consts),
                ("TAPS_UNROLLED", body.done()),
                (
                    "OUT_BINDING",
                    "@group(0) @binding(2) var<storage, read_write> OUT: array<sv4>;".to_string(),
                ),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        vec![StorageDir::Read, StorageDir::ReadWrite]
    }

    fn workgroups(&self) -> [u32; 3] {
        let (oh, ow) = self.out_hw();
        [(oh * ow * self.ocg()).div_ceil(256), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        // (k, s, pad_c) — pad_c는 비4배수 경계 포함
        for (k, s, pad_c) in [(2u32, 2u32, 0u32), (2, 2, 4), (3, 1, 2)] {
            let spec = MaxPoolSpec {
                ih: 16,
                iw: 16,
                c: 42,
                k,
                s,
                pad: [0; 4],
                pad_c,
                dt: DType::F32,
            };
            validate_wgsl(&spec.wgsl(&caps));
        }
    }
}
