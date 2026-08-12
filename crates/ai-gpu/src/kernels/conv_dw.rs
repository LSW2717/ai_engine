//! depthwise conv 커널 — k3/k5, s1/s2, 임의 pad.
//!
//! 채널당 독립 연산이라 GEMM 구조가 무의미하다: 슬라이딩 윈도우 + vec4(4채널 동시).
//! 가중치는 `pack_weights_dw`의 `[cg][tap]` 레이아웃 — 한 채널그룹의 탭이 연속이라
//! 공유메모리 1회 적재 단위와 일치한다.
//!
//! (2×2px/스레드 halo 타일 변형은 벤치에서 dw가 병목으로 나오면 추가 — RVM 기준
//! dw는 전체 FLOPs의 ~3%라 direct 버전이 우선이다.)

use ai_core::ops::Conv2d;
use ai_core::{Activation, DType};

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::{fill, W};
use crate::kernels::common::{epilogue, sv4_alias};

const TEMPLATE: &str = include_str!("shaders/conv_dw.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct ConvDwSpec {
    pub ih: u32,
    pub iw: u32,
    /// 채널 수 (cg = ceil(c/4))
    pub c: u32,
    pub k: u32,
    pub s: u32,
    /// dilation (RVM 인코더 마지막 스테이지 d=2)
    pub d: u32,
    /// [top, left, bottom, right]
    pub pad: [u32; 4],
    pub act: Activation,
    pub residual: bool,
    pub dt: DType,
}

impl ConvDwSpec {
    pub fn from_op(op: &Conv2d, ih: u32, iw: u32, residual: bool, dt: DType) -> Self {
        assert!(op.is_depthwise(), "depthwise op이 아님");
        assert_eq!(op.kh, op.kw, "정방 커널만 지원");
        assert_eq!(op.sh, op.sw, "등방 stride만 지원");
        Self { ih, iw, c: op.cin, k: op.kh, s: op.sh, d: op.dil, pad: op.pad, act: op.act, residual, dt }
    }

    pub fn out_hw(&self) -> (u32, u32) {
        let ek = self.d * (self.k - 1) + 1;
        let oh = (self.ih + self.pad[0] + self.pad[2] - ek) / self.s + 1;
        let ow = (self.iw + self.pad[1] + self.pad[3] - ek) / self.s + 1;
        (oh, ow)
    }

    fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }

    fn out_binding(&self) -> u32 {
        if self.residual { 5 } else { 4 }
    }
}

impl KernelSpec for ConvDwSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "conv_dw {}x{} c{} k{} s{} d{} p{:?} {} dt={}",
            self.ih,
            self.iw,
            self.c,
            self.k,
            self.s,
            self.d,
            self.pad,
            epilogue::key_fragment(true, self.act, self.residual),
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let (oh, ow) = self.out_hw();
        let taps = self.k * self.k;
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {}u;\nconst OW: u32 = {}u;\nconst CG: u32 = {}u;\nconst TAPS: u32 = {}u;",
            self.ih, self.iw, oh, ow, self.cg(), taps
        );
        let wsh = format!("var<workgroup> Wsh: array<vec4f, {taps}>;");

        // 탭 완전 언롤 — stride/pad는 리터럴로 박는다
        let mut body = W::new();
        for ky in 0..self.k {
            for kx in 0..self.k {
                let tap = ky * self.k + kx;
                body.line(format!(
                    "{{ let iy = i32(oy) * {} + {} - {};",
                    self.s,
                    ky * self.d,
                    self.pad[0]
                ));
                body.line(format!(
                    "  let ix = i32(ox) * {} + {} - {};",
                    self.s,
                    kx * self.d,
                    self.pad[1]
                ));
                body.line("  let m = select(0.0, 1.0, iy >= 0 && iy < IH && ix >= 0 && ix < IW);");
                body.line("  let cy = u32(clamp(iy, 0, IH - 1));");
                body.line("  let cx = u32(clamp(ix, 0, IW - 1));");
                body.line(format!(
                    "  acc = acc + Wsh[{tap}u] * vec4f(IN[(cy * u32(IW) + cx) * CG + cg]) * m; }}"
                ));
            }
        }

        let (res_b, out_b) = crate::kernels::common::gemm_tile::binding_slots(self.residual);

        let epi = epilogue::emit(
            "acc",
            Some("vec4f(BIAS[cg])"),
            self.act,
            self.residual.then_some("vec4f(RES[out_idx])"),
        );

        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("CONSTS", consts),
                ("WSH_DECL", wsh),
                ("RES_BINDING", res_b),
                ("OUT_BINDING", out_b),
                ("TAPS_UNROLLED", body.done()),
                ("EPILOGUE", epi),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        let mut b = vec![StorageDir::Read, StorageDir::Read, StorageDir::Read];
        if self.residual {
            b.push(StorageDir::Read);
        }
        b.push(StorageDir::ReadWrite);
        b
    }

    fn workgroups(&self) -> [u32; 3] {
        let (oh, ow) = self.out_hw();
        [ow.div_ceil(8), oh.div_ceil(8), self.cg()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::common::activation::ALL;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates_all_variants() {
        let caps = crate::test_util::fake_caps();
        for (k, s) in [(3u32, 1u32), (3, 2), (5, 1), (5, 2)] {
            let p = (k - 1) / 2;
            for act in ALL {
                for residual in [false, true] {
                    for dt in [DType::F32, DType::F16] {
                        let spec = ConvDwSpec {
                            ih: 36,
                            iw: 64,
                            c: 96,
                            k,
                            s,
                            d: 1,
                            pad: [p; 4],
                            act,
                            residual,
                            dt,
                        };
                        validate_wgsl(&spec.wgsl(&caps));
                    }
                }
            }
        }
    }
}
