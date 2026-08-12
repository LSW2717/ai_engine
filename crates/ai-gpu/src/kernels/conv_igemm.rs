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
const TEMPLATE_SPLITK: &str = include_str!("shaders/conv_igemm_splitk.wgsl");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IgemmVariant {
    Tiled,
    Direct,
    /// 셀(1픽셀×1출력그룹) × k-청크 분할 + 워크그룹 리덕션 — direct4가
    /// 스레드 기아인 저해상 심층 k3 (9×16 128->128 등)
    Splitk,
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
    /// concat-into-conv 융합 파트별 채널 수 (0 = 미사용, 전부 0 = 단일 입력).
    /// 파트 시작이 채널그룹 정렬임은 변환기(fuse_concat)가 보장.
    pub srcs: [u32; 3],
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
            srcs: [0; 3],
        }
    }

    fn nsrcs(&self) -> u32 {
        self.srcs.iter().filter(|c| **c > 0).count() as u32
    }

    /// 입력 파트 목록: (버퍼명, 파트 kg 폭, 전역 kg 오프셋). 단일 입력 = 1파트 "IN".
    fn parts(&self) -> Vec<(String, u32, u32)> {
        if self.nsrcs() <= 1 {
            return vec![("IN".into(), self.kg(), 0)];
        }
        let mut v = Vec::new();
        let mut off = 0;
        for (i, c) in self.srcs.iter().enumerate() {
            if *c == 0 {
                break;
            }
            let cg = c.div_ceil(4);
            v.push((format!("IN{i}"), cg, off));
            off += cg;
        }
        debug_assert_eq!(off, self.kg(), "파트 kg 합이 cin과 불일치");
        v
    }

    /// 바인딩 선언 블록 — 파트 수·residual에 따라 번호가 밀리므로 통짜 생성
    fn emit_bindings(&self) -> String {
        let mut w = W::new();
        let mut b = 1u32;
        if self.nsrcs() <= 1 {
            w.line(format!("@group(0) @binding({b}) var<storage, read> IN: array<sv4>;"));
            b += 1;
        } else {
            for i in 0..self.nsrcs() {
                w.line(format!("@group(0) @binding({b}) var<storage, read> IN{i}: array<sv4>;"));
                b += 1;
            }
        }
        w.line(format!(
            "@group(0) @binding({b}) var<storage, read> W: array<sv4>; // [tap][kgp][ng][j]"
        ));
        b += 1;
        w.line(format!("@group(0) @binding({b}) var<storage, read> BIAS: array<sv4>;"));
        b += 1;
        if self.residual {
            w.line(format!("@group(0) @binding({b}) var<storage, read> RES: array<sv4>;"));
            b += 1;
        }
        w.line(format!("@group(0) @binding({b}) var<storage, read_write> OUT: array<sv4>;"));
        w.done()
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
        // (실험 기록: 가중치 공유메모리 프리로드(DirectShared)는 Apple에서 완패 —
        // 워크그룹당 8~16KB 공유메모리가 점유율을 붕괴시키고, 동일 주소 전역 로드는
        // 이미 SIMD 브로드캐스트+캐시로 최적이다. webgl2 챔피언도 공유메모리 없음.)
        // 멀티소스(concat 융합)는 Direct/Splitk만 지원 — Tiled 후보라도 Direct로.
        if self.nsrcs() > 1
            || self.ng() <= 20
            || self.m() < 512
            || self.k * self.k * self.kg() <= 9
        {
            // direct4 스레드 수(M/4 블록 × NG)가 기아이고 **나눌 K가 실제로 클 때만**
            // K분할. 실측: 9×16 KG32(1152스레드)·18×32 KG43(2880)은 Splitk 승
            // (171->80: 287→194µs), 36×64(5760)부터는 direct4의 4px 공유·행 재사용이
            // 이긴다(226 vs 266µs). KG<16(segm s050 소채널 conv)은 청크가 비어
            // 리덕션 오버헤드만 남으므로 스레드가 적어도 Direct 유지.
            let (oh, ow) = self.out_hw();
            let nblk = oh * ow.div_ceil(4);
            if nblk * self.ng() < 4096 && self.kg() >= 16 {
                IgemmVariant::Splitk
            } else {
                IgemmVariant::Direct
            }
        } else {
            IgemmVariant::Tiled
        }
    }

    /// Direct의 픽셀 블록 폭. (실험 기록: 8픽셀 블로킹은 dot당 로드가 38% 줄어도
    /// 레지스터 압박(a 10개+acc 8개)으로 전 shape 20~60% 퇴행 — webgl2의 "ig 언롤
    /// 스필" 교훈과 동일한 벽. 4가 최적.)
    fn pb(&self) -> u32 {
        4
    }

    /// direct 본문 — 행(ky) 단위 처리: 같은 행의 kx 탭들이 겹치는 입력 열을
    /// 공유하므로 행당 유니크 열만 로드 (k3 s1: 탭별 12로드 → 행당 6로드). 경계는
    /// 분기 대신 clamp 인덱스 + 마스크 곱, 행이 통째로 밖이면 스킵.
    fn emit_direct_body(&self) -> String {
        let pb = self.pb();
        // 사용되는 열 오프셋 집합: {p*S + kx*D | p∈0..pb, kx∈0..k}
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
        let parts = self.parts();
        let mut body = W::new();
        for ky in 0..self.k {
            let dy = ky * self.d;
            body.line(format!("{{ // row {ky}"));
            body.line(format!("  let iy = i32(oy) * S + {dy} - PT;"));
            body.line("  if (iy >= 0 && iy < IH) {");
            body.line("    let row = u32(iy) * u32(IW);");
            for &c in &used {
                body.line(format!("    let x{c} = i32(ox0) * S + {};", c as i32 - pl));
                body.line(format!("    let pos{c} = row + u32(clamp(x{c}, 0, IW - 1));"));
                body.line(format!("    let m{c} = f32(x{c} >= 0 && x{c} < IW);"));
            }
            for (pi, (buf, cg, off)) in parts.iter().enumerate() {
                body.line(format!("    {{ // part {pi} (kg {off}..{})", off + cg));
                for &c in &used {
                    body.line(format!("      let q{c} = pos{c} * {cg}u;"));
                }
                body.line(format!("      for (var kg = 0u; kg < {cg}u; kg = kg + 1u) {{"));
                for &c in &used {
                    body.line(format!("        let a{c} = vec4f({buf}[q{c} + kg]) * m{c};"));
                }
                for kx in 0..self.k {
                    let tap = ky * self.k + kx;
                    body.line(format!(
                        "        {{ let wb = (({tap}u * KGP + kg + {off}u) * NG + ng) * 4u;"
                    ));
                    body.line("          let w0 = vec4f(W[wb]); let w1 = vec4f(W[wb + 1u]);");
                    body.line("          let w2 = vec4f(W[wb + 2u]); let w3 = vec4f(W[wb + 3u]);");
                    for p in 0..pb {
                        let c = p * self.s + kx * self.d;
                        body.line(format!(
                            "          acc[{p}] = acc[{p}] + vec4f(dot(w0, a{c}), dot(w1, a{c}), dot(w2, a{c}), dot(w3, a{c}));"
                        ));
                    }
                    body.line("        }");
                }
                body.line("      }");
                body.line("    }");
            }
            body.line("  }");
            body.line("}");
        }
        body.done()
    }

    /// direct 공용 스토어: 픽셀 블록 언롤 + OW 에지 가드 + 에필로그
    fn emit_direct_store(&self) -> String {
        let mut store = W::new();
        for p in 0..self.pb() {
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
        store.done()
    }
}

impl KernelSpec for ConvIgemmSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "conv_igemm v={:?}{}{} {}x{} {}->{} k{} s{} d{} p{:?} {} dt={}",
            self.variant(),
            if self.variant() == IgemmVariant::Direct { format!("/pb{}", self.pb()) } else { String::new() },
            if self.nsrcs() > 1 { format!(" srcs{:?}", self.srcs) } else { String::new() },
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
                let pb = self.pb();
                let bpr = ow.div_ceil(pb);
                let nblk = oh * bpr;
                let extra =
                    format!("const NBLK: u32 = {nblk}u;\nconst BPR: u32 = {bpr}u;\nconst PB: u32 = {pb}u;");
                let acc_init = (0..pb).map(|_| "vec4f(0.0)").collect::<Vec<_>>().join(", ");
                let acc_decl = format!("var acc = array<vec4f, {pb}>({acc_init});");
                fill(
                    TEMPLATE_DIRECT,
                    &[
                        ("TYPES", types),
                        ("BINDINGS", self.emit_bindings()),
                        ("CONSTS", format!("{consts}\n{extra}")),
                        ("ACC_DECL", acc_decl),
                        ("TAP_LOOPS", self.emit_direct_body()),
                        ("STORE4", self.emit_direct_store()),
                    ],
                )
            }
            IgemmVariant::Splitk => {
                let cells = m * self.ng();
                let (split, cpw) = crate::kernels::gemm_pw::splitk_geometry(cells);
                let (pxt, ngt) = crate::kernels::gemm_pw::splitk_tile(self.ng(), cpw);
                let ngtc = self.ng().div_ceil(ngt);
                let kc = self.kg().div_ceil(split);
                let extra = format!(
                    "const CPW: u32 = {cpw}u;\nconst SPLIT: u32 = {split}u;\nconst KC: u32 = {kc}u;\nconst PXT: u32 = {pxt}u;\nconst NGT: u32 = {ngt}u;\nconst NGTC: u32 = {ngtc}u;"
                );
                let parts = self.parts();
                let mut taps = W::new();
                for ky in 0..self.k {
                    for kx in 0..self.k {
                        let tap = ky * self.k + kx;
                        let (dy, dx) = (ky * self.d, kx * self.d);
                        taps.line(format!("{{ // tap {tap}"));
                        taps.line(format!("  let iy = oy * S + {dy} - PT;"));
                        taps.line(format!("  let ix = ox * S + {dx} - PL;"));
                        taps.line("  if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {");
                        taps.line("    let pix = u32(iy) * u32(IW) + u32(ix);");
                        for (pi, (buf, cg, off)) in parts.iter().enumerate() {
                            let end = off + cg;
                            taps.line(format!("    {{ // part {pi}"));
                            taps.line(format!("      let base = pix * {cg}u;"));
                            // 청크 [k0,k1)과 파트 [off,end)의 교집합만 순회
                            if parts.len() == 1 {
                                taps.line("      for (var kg = k0; kg < k1; kg = kg + 1u) {");
                            } else {
                                taps.line(format!(
                                    "      for (var kg = max(k0, {off}u); kg < min(k1, {end}u); kg = kg + 1u) {{"
                                ));
                            }
                            taps.line(format!("        let a = vec4f({buf}[base + kg - {off}u]);"));
                            taps.line(format!(
                                "        let wb = (({tap}u * KGP + kg) * NG + ng) * 4u;"
                            ));
                            taps.line(
                                "        acc = acc + vec4f(dot(vec4f(W[wb]), a), dot(vec4f(W[wb + 1u]), a), dot(vec4f(W[wb + 2u]), a), dot(vec4f(W[wb + 3u]), a));",
                            );
                            taps.line("      }");
                            taps.line("    }");
                        }
                        taps.line("  }");
                        taps.line("}");
                    }
                }
                let epi = epilogue::emit(
                    "acc2",
                    Some("vec4f(BIAS[ng])"),
                    self.act,
                    self.residual.then_some("vec4f(RES[out_idx])"),
                );
                fill(
                    TEMPLATE_SPLITK,
                    &[
                        ("TYPES", types),
                        ("BINDINGS", self.emit_bindings()),
                        ("CONSTS", format!("{consts}\n{extra}")),
                        ("TAPS", taps.done()),
                        ("EPILOGUE", epi),
                    ],
                )
            }
        }
    }

    fn bindings(&self) -> Vec<StorageDir> {
        let n_in = self.nsrcs().max(1) as usize;
        let mut b = vec![StorageDir::Read; n_in + 2]; // IN들 + W + BIAS
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
                let nblk = oh * ow.div_ceil(self.pb());
                [(nblk * self.ng()).div_ceil(256), 1, 1]
            }
            IgemmVariant::Splitk => {
                let cells = self.m() * self.ng();
                let (_, cpw) = crate::kernels::gemm_pw::splitk_geometry(cells);
                let (pxt, ngt) = crate::kernels::gemm_pw::splitk_tile(self.ng(), cpw);
                [self.m().div_ceil(pxt) * self.ng().div_ceil(ngt), 1, 1]
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
                            srcs: [0; 3],
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
            srcs: [0; 3],
        };
        // 스템: M 크지만 K 극소 → Direct (스레드 충분)
        assert_eq!(mk(72, 128, 3, 3).variant(), IgemmVariant::Direct);
        // 작은 그리드라도 KG<16이면 Direct (나눌 K가 없음 — segm s050 소채널)
        assert_eq!(mk(24, 32, 16, 3).variant(), IgemmVariant::Direct);
        // 큰 그리드는 KG가 커도 Direct (4px 공유·행 재사용이 유리)
        assert_eq!(mk(72, 128, 64, 3).variant(), IgemmVariant::Direct);
        // 작은 M + K 충분 → Splitk (9×16 심층)
        assert_eq!(mk(9, 16, 64, 3).variant(), IgemmVariant::Splitk);
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
            srcs: [0; 3],
        };
        assert_eq!(big.variant(), IgemmVariant::Tiled);
    }
}
