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

    /// 픽셀 블록 폭 — k5(탭 25)만 4px 행 재사용이 이긴다 (실측: k3는 스레드
    /// 감소+CG소형 비합체로 퇴행, k5는 c960 53.8→38.5µs / c120 22→16.8µs).
    fn pb(&self) -> u32 {
        if self.k >= 5 { 4 } else { 1 }
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
        let pb = self.pb();
        let bpr = ow.div_ceil(pb);
        let nblk = oh * bpr;
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {}u;\nconst OW: u32 = {}u;\nconst CG: u32 = {}u;\nconst TAPS: u32 = {}u;\nconst NBLK: u32 = {}u;\nconst BPR: u32 = {}u;\nconst PB: u32 = {}u;",
            self.ih, self.iw, oh, ow, self.cg(), taps, nblk, bpr, pb
        );

        // direct4식 행 재사용: 행당 유니크 열 {p*S + kx*D} 로드, 탭은 완전 언롤
        let mut used: Vec<u32> = Vec::new();
        for p in 0..pb {
            for kx in 0..self.k {
                let c = p * self.s + kx * self.d;
                if !used.contains(&c) {
                    used.push(c);
                }
            }
        }
        used.sort_unstable();
        let pl = self.pad[1] as i32;
        let mut body = W::new();
        for ky in 0..self.k {
            let dy = ky * self.d;
            body.line(format!("{{ // row {ky}"));
            body.line(format!("  let iy = i32(oy) * {} + {} - {};", self.s, dy, self.pad[0]));
            body.line("  if (iy >= 0 && iy < IH) {");
            body.line("    let row = u32(iy) * u32(IW);");
            for &c in &used {
                body.line(format!("    let x{c} = i32(ox0) * {} + {};", self.s, c as i32 - pl));
                body.line(format!(
                    "    let a{c} = vec4f(IN[(row + u32(clamp(x{c}, 0, IW - 1))) * CG + cg]) * f32(x{c} >= 0 && x{c} < IW);"
                ));
            }
            for kx in 0..self.k {
                let tap = ky * self.k + kx;
                body.line(format!("    {{ let wv = vec4f(W[{tap}u * CG + cg]);"));
                for p in 0..pb {
                    let c = p * self.s + kx * self.d;
                    body.line(format!("      acc[{p}] = acc[{p}] + wv * a{c};"));
                }
                body.line("    }");
            }
            body.line("  }");
            body.line("}");
        }

        // 스토어: 4픽셀 언롤 + OW 에지 가드 + 에필로그
        let mut store = W::new();
        for p in 0..pb {
            store.line(format!("if (ox0 + {p}u < OW) {{"));
            store.line(format!("  let out_idx = (oy * OW + ox0 + {p}u) * CG + cg;"));
            store.line(format!("  var v_{p} = acc[{p}];"));
            let epi = epilogue::emit(
                &format!("v_{p}"),
                Some("vec4f(BIAS[cg])"),
                self.act,
                self.residual.then_some(&*format!("vec4f(RES[out_idx])")),
            );
            for line in epi.lines() {
                store.line(format!("  {line}"));
            }
            store.line(format!("  OUT[out_idx] = sv4(v_{p});"));
            store.line("}");
        }

        let (res_b, out_b) = crate::kernels::common::gemm_tile::binding_slots(self.residual);

        let acc_init = (0..pb).map(|_| "vec4f(0.0)").collect::<Vec<_>>().join(", ");
        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("CONSTS", consts),
                ("RES_BINDING", res_b),
                ("OUT_BINDING", out_b),
                ("ACC_DECL", format!("var acc = array<vec4f, {pb}>({acc_init});")),
                ("ROWS", body.done()),
                ("STORE4", store.done()),
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
        let nblk = oh * ow.div_ceil(self.pb());
        [(nblk * self.cg()).div_ceil(256), 1, 1]
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
