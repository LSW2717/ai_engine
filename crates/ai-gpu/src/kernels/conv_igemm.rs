//! 일반 conv (k>1, groups=1) = implicit GEMM 커널.
//!
//! im2col 버퍼를 실체화하지 않는다 — tiled 변형은 A-타일 로드에서 소스 좌표를
//! 즉석 계산하고, 가중치가 tap-major(`[tap][kgp][ng][j]`, `pack_weights_conv
//! kg_align=4`)라 B 진행은 순수 선형이다. MAC/스토어는 gemm_pw와 동일 코어 공유.
//!
//! 변형 정책(순수 함수):
//! - `Direct`: M < 512(출력 기아) 이거나 taps×kg ≤ 9(스템 cin=3처럼 K 극소) —
//!   스레드당 1셀, 탭 언롤 + kg 런타임 루프
//! - `Tiled`: 그 외 — 32×32×16 타일 implicit GEMM

use ai_core::ops::Conv2d;
use ai_core::{Activation, DType};

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::{fill, W};
use crate::kernels::common::{epilogue, gemm_tile, sv4_alias};
use crate::kernels::gemm_pw::{TM, TN_NG};

const TEMPLATE_TILED: &str = include_str!("shaders/conv_igemm_tiled.wgsl");
const TEMPLATE_DIRECT: &str = include_str!("shaders/conv_igemm_direct.wgsl");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IgemmVariant {
    Tiled,
    Direct,
}

#[derive(Clone, Copy, Debug)]
pub struct ConvIgemmSpec {
    pub ih: u32,
    pub iw: u32,
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub s: u32,
    /// dilation
    pub d: u32,
    /// [top, left, bottom, right]
    pub pad: [u32; 4],
    pub act: Activation,
    pub residual: bool,
    pub dt: DType,
}

impl ConvIgemmSpec {
    pub fn from_op(op: &Conv2d, ih: u32, iw: u32, residual: bool, dt: DType) -> Self {
        assert_eq!(op.groups, 1, "일반 conv만 (dw는 conv_dw, pw는 gemm_pw)");
        assert_eq!(op.kh, op.kw, "정방 커널만 지원");
        assert_eq!(op.sh, op.sw, "등방 stride만 지원");
        assert!(op.kh > 1, "1×1은 gemm_pw를 사용");
        Self {
            ih,
            iw,
            cin: op.cin,
            cout: op.cout,
            k: op.kh,
            s: op.sh,
            d: op.dil,
            pad: op.pad,
            act: op.act,
            residual,
            dt,
        }
    }

    pub fn out_hw(&self) -> (u32, u32) {
        let ek = self.d * (self.k - 1) + 1;
        let oh = (self.ih + self.pad[0] + self.pad[2] - ek) / self.s + 1;
        let ow = (self.iw + self.pad[1] + self.pad[3] - ek) / self.s + 1;
        (oh, ow)
    }

    fn m(&self) -> u32 {
        let (oh, ow) = self.out_hw();
        oh * ow
    }

    fn kg(&self) -> u32 {
        self.cin.div_ceil(4)
    }

    /// 패커(kg_align=4)와 일치해야 하는 패딩된 kg
    fn kgp(&self) -> u32 {
        self.kg().next_multiple_of(4)
    }

    fn ng(&self) -> u32 {
        self.cout.div_ceil(4)
    }

    pub fn variant(&self) -> IgemmVariant {
        // Direct4(4픽셀 블로킹)가 작은 cout(타일 스레드 낭비)과 작은 K(스템)에서 승리.
        // RVM/MNv 계열 디코더는 전부 NG ≤ 20이라 사실상 Direct4가 주력이고,
        // Tiled는 cout·K 모두 큰 미래 shape용으로 유지한다.
        if self.ng() <= 20 || self.m() < 512 || self.k * self.k * self.kg() <= 9 {
            IgemmVariant::Direct
        } else {
            IgemmVariant::Tiled
        }
    }
}

impl KernelSpec for ConvIgemmSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "conv_igemm v={:?} {}x{} {}->{} k{} s{} d{} p{:?} {} dt={}",
            self.variant(),
            self.ih,
            self.iw,
            self.cin,
            self.cout,
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
        let m = oh * ow;
        let kb = self.kgp() / 4; // KGP는 4의 배수 (패커 보장)
        let consts = format!(
            "const M: u32 = {m}u;\nconst OW: u32 = {ow}u;\nconst KG: u32 = {}u;\nconst KGP: u32 = {}u;\nconst NG: u32 = {}u;\nconst KB: u32 = {kb}u;\nconst IH: i32 = {};\nconst IW: i32 = {};\nconst S: i32 = {};\nconst D: i32 = {};\nconst PT: i32 = {};\nconst PL: i32 = {};",
            self.kg(),
            self.kgp(),
            self.ng(),
            self.ih,
            self.iw,
            self.s,
            self.d,
            self.pad[0],
            self.pad[1]
        );
        let (res_b, out_b) = gemm_tile::binding_slots(self.residual);
        let types = sv4_alias(self.dt);

        match self.variant() {
            IgemmVariant::Tiled => {
                // 탭 외부 언롤: 좌표 오프셋 리터럴 + 사전 계산된 픽셀 디코드 사용
                let mac = gemm_tile::emit_mac_unrolled();
                let mut taps_code = W::new();
                for ky in 0..self.k {
                    for kx in 0..self.k {
                        let tap = ky * self.k + kx;
                        let (dy, dx) = (ky * self.d, kx * self.d);
                        taps_code.line(format!("{{ // tap {tap} (dy={dy}, dx={dx})"));
                        taps_code.line(format!("  let iya = oya * S + {dy} - PT;"));
                        taps_code.line(format!("  let ixa = oxa * S + {dx} - PL;"));
                        taps_code.line(format!("  let iyb = oyb * S + {dy} - PT;"));
                        taps_code.line(format!("  let ixb = oxb * S + {dx} - PL;"));
                        taps_code.line(
                            "  let a_ok = va_ok && iya >= 0 && iya < IH && ixa >= 0 && ixa < IW;",
                        );
                        taps_code.line(
                            "  let b_ok = vb_ok && iyb >= 0 && iyb < IH && ixb >= 0 && ixb < IW;",
                        );
                        taps_code.line("  let abase = (u32(clamp(iya, 0, IH - 1)) * u32(IW) + u32(clamp(ixa, 0, IW - 1))) * KG;");
                        taps_code.line("  let bbase = (u32(clamp(iyb, 0, IH - 1)) * u32(IW) + u32(clamp(ixb, 0, IW - 1))) * KG;");
                        taps_code.line("  for (var kb = 0u; kb < KB; kb = kb + 1u) {");
                        taps_code.line("    let kga = kb * 4u + kka;");
                        taps_code.line("    let in_k = kga < KG;");
                        taps_code.line("    { var v = vec4f(0.0);");
                        taps_code.line("      if (a_ok && in_k) { v = vec4f(IN[abase + kga]); }");
                        taps_code.line("      Ash[la] = v; }");
                        taps_code.line("    { var v = vec4f(0.0);");
                        taps_code.line("      if (b_ok && in_k) { v = vec4f(IN[bbase + kga]); }");
                        taps_code.line("      Ash[lb] = v; }");
                        taps_code.line(format!(
                            "    {{ var v = vec4f(0.0);\n      if (bng_a < NG) {{ v = vec4f(W[(({tap}u * KGP + kb * 4u + bkk_a) * NG + bng_a) * 4u + bj_a]); }}\n      Bsh[la] = v; }}"
                        ));
                        taps_code.line(format!(
                            "    {{ var v = vec4f(0.0);\n      if (bng_b < NG) {{ v = vec4f(W[(({tap}u * KGP + kb * 4u + bkk_b) * NG + bng_b) * 4u + bj_b]); }}\n      Bsh[lb] = v; }}"
                        ));
                        taps_code.line("    workgroupBarrier();");
                        for line in mac.lines() {
                            taps_code.line(format!("    {line}"));
                        }
                        taps_code.line("    workgroupBarrier();");
                        taps_code.line("  }");
                        taps_code.line("}");
                    }
                }
                fill(
                    TEMPLATE_TILED,
                    &[
                        ("TYPES", types),
                        ("CONSTS", consts),
                        ("RES_BINDING", res_b),
                        ("OUT_BINDING", out_b),
                        ("TAP_LOOPS", taps_code.done()),
                        (
                            "STORE_UNROLLED",
                            gemm_tile::emit_store_unrolled(self.act, self.residual),
                        ),
                    ],
                )
            }
            IgemmVariant::Direct => {
                // direct4: 가로 4픽셀 × 1그룹, 탭 외부 언롤, (tap,kg)당 가중치 4페치 공유
                let bpr = ow.div_ceil(4);
                let nblk = oh * bpr;
                let extra = format!("const NBLK: u32 = {nblk}u;\nconst BPR: u32 = {bpr}u;");
                let mut body = W::new();
                for ky in 0..self.k {
                    for kx in 0..self.k {
                        let tap = ky * self.k + kx;
                        let (dy, dx) = (ky * self.d, kx * self.d);
                        body.line(format!("{{ // tap {tap}"));
                        body.line(format!("  let iy = i32(oy) * S + {dy} - PT;"));
                        body.line("  let y_ok = iy >= 0 && iy < IH;");
                        body.line("  let row = u32(clamp(iy, 0, IH - 1)) * u32(IW);");
                        for p in 0..4 {
                            body.line(format!(
                                "  let ix{p} = i32(ox0 + {p}u) * S + {dx} - PL;"
                            ));
                            body.line(format!(
                                "  let ok{p} = y_ok && ix{p} >= 0 && ix{p} < IW && ox0 + {p}u < OW;"
                            ));
                            body.line(format!(
                                "  let base{p} = (row + u32(clamp(ix{p}, 0, IW - 1))) * KG;"
                            ));
                        }
                        body.line("  for (var kg = 0u; kg < KG; kg = kg + 1u) {");
                        body.line(format!("    let wb = (({tap}u * KGP + kg) * NG + ng) * 4u;"));
                        body.line("    let w0 = vec4f(W[wb]); let w1 = vec4f(W[wb + 1u]);");
                        body.line("    let w2 = vec4f(W[wb + 2u]); let w3 = vec4f(W[wb + 3u]);");
                        for p in 0..4 {
                            body.line(format!(
                                "    if (ok{p}) {{ let a = vec4f(IN[base{p} + kg]); \
                                 acc[{p}] = acc[{p}] + vec4f(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }}"
                            ));
                        }
                        body.line("  }");
                        body.line("}");
                    }
                }
                // 스토어: 4픽셀 언롤 + OW 에지 가드 + 에필로그
                let mut store = W::new();
                for p in 0..4 {
                    store.line(format!("if (ox0 + {p}u < OW) {{"));
                    store.line(format!("  let out_idx = (oy * OW + ox0 + {p}u) * NG + ng;"));
                    store.line(format!("  var v_{p} = acc[{p}];"));
                    let epi = epilogue::emit(
                        &format!("v_{p}"),
                        Some("vec4f(BIAS[ng])"),
                        self.act,
                        self.residual.then_some(&*format!("vec4f(RES[out_idx])")),
                    );
                    for line in epi.lines() {
                        store.line(format!("  {line}"));
                    }
                    store.line(format!("  OUT[out_idx] = sv4(v_{p});"));
                    store.line("}");
                }
                fill(
                    TEMPLATE_DIRECT,
                    &[
                        ("TYPES", types),
                        ("CONSTS", format!("{consts}\n{extra}")),
                        ("RES_BINDING", res_b),
                        ("OUT_BINDING", out_b),
                        ("TAP_LOOPS", body.done()),
                        ("STORE4", store.done()),
                    ],
                )
            }
        }
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
        match self.variant() {
            IgemmVariant::Direct => {
                let (oh, ow) = self.out_hw();
                let nblk = oh * ow.div_ceil(4);
                [(nblk * self.ng()).div_ceil(64), 1, 1]
            }
            IgemmVariant::Tiled => [self.m().div_ceil(TM), self.ng().div_ceil(TN_NG), 1],
        }
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
        // (ih, iw, cin, cout, k, s): 스템(direct 대-M), tiled, direct 소-M, k5
        for (ih, iw, cin, cout, k, s) in [
            (72u32, 128u32, 3u32, 16u32, 3u32, 2u32),
            (24, 32, 16, 24, 3, 1),
            (17, 23, 6, 8, 3, 2),
            (33, 45, 8, 12, 5, 2),
        ] {
            let p = (k - 1) / 2;
            for act in ALL {
                for residual in [false, true] {
                    for dt in [DType::F32, DType::F16] {
                        let spec = ConvIgemmSpec {
                            ih,
                            iw,
                            cin,
                            cout,
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

    #[test]
    fn variant_policy() {
        let mk = |ih, iw, cin, k| ConvIgemmSpec {
            ih,
            iw,
            cin,
            cout: 16,
            k,
            s: 1,
            d: 1,
            pad: [(k - 1) / 2; 4],
            act: Activation::None,
            residual: false,
            dt: DType::F32,
        };
        // 스템: M 크지만 K 극소 → Direct
        assert_eq!(mk(72, 128, 3, 3).variant(), IgemmVariant::Direct);
        // 작은 cout(NG≤20) → Direct4 (RVM/MNv 디코더 전역)
        assert_eq!(mk(24, 32, 16, 3).variant(), IgemmVariant::Direct);
        // 작은 M → Direct
        assert_eq!(mk(9, 16, 64, 3).variant(), IgemmVariant::Direct);
        // cout·K·M 모두 큼 → Tiled
        let big = ConvIgemmSpec {
            ih: 24,
            iw: 32,
            cin: 64,
            cout: 128,
            k: 3,
            s: 1,
            d: 1,
            pad: [1; 4],
            act: Activation::None,
            residual: false,
            dt: DType::F32,
        };
        assert_eq!(big.variant(), IgemmVariant::Tiled);
    }
}
