//! pointwise(1×1) conv = GEMM 커널 — MobileNet 계열 연산량의 주력.
//!
//! `out[M, Ng] = epilogue(in[M, Kg] × W[Kg, Ng])`, M = H·W 픽셀.
//! NHWC-C4 텐서는 row-major `[M, Kg]` 행렬 그 자체라 재배열이 없다.
//!
//! 변형 정책(`pick_variant`, 순수 함수 — 캐시 키에 포함):
//! - `Tiled` (M ≥ 512): 32×32×16 타일, 공유메모리 급전, 스레드당 4px×1그룹 레지스터 블록
//! - `Small` (M < 512): 스레드당 1셀, 전체 K 직독 — 심층 저해상도/SE 경로의 기아 해법
//!
//! bias는 항상 바인딩(없으면 0 벡터) — 변형 공간을 줄인다. residual만 BGL을 바꾼다.
//! 가중치는 `pack_weights_conv(kg_align=4)` 레이아웃(`[kg][ng][j]`, kg 4그룹 패딩) 필수.

use ai_core::{Activation, DType};

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::writer::fill;
use crate::kernels::common::{epilogue, gemm_tile, type_aliases};

const TEMPLATE_TILED: &str = include_str!("shaders/gemm_pw_tiled.wgsl");
const TEMPLATE_SMALL: &str = include_str!("shaders/gemm_pw_small.wgsl");
const TEMPLATE_GEMV: &str = include_str!("shaders/gemm_pw_gemv.wgsl");
const TEMPLATE_SPLITK: &str = include_str!("shaders/gemm_pw_splitk.wgsl");

/// 타일 파라미터 (tiled 변형)
pub const TM: u32 = 32; // 픽셀 타일
pub const TN_NG: u32 = 8; // 출력 채널그룹 타일 (32ch)
pub const TK: u32 = 4; // k-step 채널그룹 (16ch)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmVariant {
    Tiled,
    Small,
    /// 워크그룹 협조 리덕션 GEMV — 셀 수(M×NG)가 극소하고 K가 큰 SE 경로.
    /// 워크그룹(256스레드)이 셀 하나의 K 합을 나눠 든다 (스레드 기아 해소).
    Gemv,
    /// 워크그룹 내 split-K — M 중간(심층 9×16 등) + K 큼.
    /// 워크그룹 = CPW셀 × SPLIT청크, 공유메모리 청크 리덕션 (단일 디스패치).
    Splitk,
}

/// Gemv 워크그룹 폭 — 리덕션 길이(KG)에 맞춘다.
///
/// 256 고정이었을 때 SE 확장 FC(240→960, KG=60)는 레인 76%가 놀면서도 리덕션 트리는
/// 8라운드를 다 돌았다. 같은 460KB를 읽는 압축 FC(960→240, KG=240)가 48GB/s인데
/// 확장 쪽만 27GB/s로 떨어지던 이유다. 최소 32는 유지한다 (Apple SIMD 폭 = 32,
/// 그 아래로는 워크그룹만 늘고 이득이 없다).
fn gemv_width(kg: u32) -> u32 {
    kg.next_power_of_two().clamp(32, 256)
}

/// Splitk의 (SPLIT, CPW) — 셀이 적을수록 K를 더 쪼개 스레드를 만든다.
/// (conv_igemm의 Splitk 변형도 같은 기하를 쓴다. SPLIT=16/CPW=16 실험은
/// 셀<4096에서도 이득 없음 — 리덕션 비용이 점유율 이득을 상쇄.)
pub(crate) fn splitk_geometry(cells: u32) -> (u32, u32) {
    if cells < 8192 { (8, 32) } else { (4, 64) }
}

/// Splitk 워크그룹의 px×ng 2D 타일 (PXT, NGT). NGT 스레드가 입력을 공유하고
/// PXT 픽셀이 가중치 라인을 공유한다 — 1px×CPW로 펴면 워크그룹마다 가중치
/// 슬라이스를 L2에서 재독해 심층 conv이 L2 대역 병목이 된다.
/// NGT는 NG를 나눠떨어지게 고른다 — NG가 4의 배수가 아닌 모델(예: cout 24 →
/// NG 6)에서 고정 NGT=4는 레인 낭비로 퇴행한다 (NG 크면 낭비 비중이 작아 4 허용).
pub(crate) fn splitk_tile(ng: u32, cpw: u32) -> (u32, u32) {
    let ngt = if ng % 4 == 0 || ng >= 8 {
        4.min(ng)
    } else if ng % 2 == 0 {
        2
    } else {
        1
    };
    (cpw / ngt, ngt)
}

/// (기각 기록) **Splitk 픽셀 블로킹(스레드당 PB픽셀)은 이득이 없다** — 2026-08-13 실측.
/// 동기는 타당했다: Splitk는 스레드당 1픽셀이라 vec4 로드 5개(A 1 + W 4)당 16 MAC =
/// 3.2 MAC/load뿐이고, 이미 빠른 Direct(144×256, 920 GMAC/s)는 4픽셀 블로킹으로
/// ~11 MAC/load다. 그런데 PB=2/4 어느 쪽도 프로덕션 shape에서 노이즈(±15µs)를 못 넘었다
/// (영향받는 7개 op 합 −15µs, 같은 런에서 무관한 conv_igemm op이 ±16µs 흔들림).
/// → 9×16 심층 pw의 175~614 GMAC/s는 **로드 처리량 병목이 아니다**. 다른 데를 봐라.
/// 부수 교훈: 에지 가드를 k-루프 안 분기로 넣으면 PB=1에서도 260µs 퇴행한다.
///
/// 변형 선택 — spec의 순수 함수여야 캐시가 결정적이다.
/// - 셀 극소(M×NG < 512) + K 큼 → Gemv (SE의 M=1: 스레드 36개→256×NG개)
/// - 셀 중간 + K 큼(타일 그리드가 기아) → Splitk (960→160 실측 120 GFLOP/s 해소)
/// - K 깊고 그리드 충분 → Tiled (공유메모리 급전이 긴 K에서만 고정비를 회수)
/// - 그 외 → Small(4px 블로킹: 가중치 페치를 4픽셀이 공유)
///
/// KG<16에서 Tiled는 K루프 1~2회에 타일 적재+배리어 고정비만 내는 꼴 — 예산표
/// 실측: 36×64 24→72(KG6)가 Tiled 83µs로 하한(6µs)의 13배. 소형 K는 Small이 정답.
pub fn pick_variant(m: u32, kg: u32, ng: u32) -> GemmVariant {
    let tiles = m.div_ceil(TM) * ng.div_ceil(TN_NG);
    if m * ng < 512 && kg >= 16 {
        GemmVariant::Gemv
    } else if kg >= 16 && tiles < 192 && ng >= splitk_ng_gate() && m >= 64 {
        GemmVariant::Splitk
    } else if ng >= 16 && kg >= 16 && m >= 64 {
        GemmVariant::Tiled
    } else {
        GemmVariant::Small
    }
}

/// Splitk를 허용하는 최소 NG.
///
/// 예전 값 16은 근거 없이 Tiled 조건을 복사한 것이었다. Small은 스레드당 4픽셀이라
/// NG가 작으면 워크그룹이 몇 개 안 나온다 — 18×32 120→40은 nblk 144 × NG 10 / 256 =
/// **6 워크그룹**으로 19코어 GPU를 3분의 1도 못 채웠다. Splitk로 보내면 216개.
/// 실측(격리 min-of-3, RVM 141op): 18×32 120→40 ×2 41.5·44.6→12.1µs,
/// 18×32 72→40 23.0→12.9, 36×64 →24 ×2 20.2·28.7→12.8·14.2. 합 −96µs.
/// 진단용 env 오버라이드 — 프로세스 내 상수라 캐시 키는 여전히 결정적이다.
fn splitk_ng_gate() -> u32 {
    use std::sync::OnceLock;
    static G: OnceLock<u32> = OnceLock::new();
    *G.get_or_init(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(v) = std::env::var("AI_PW_SPLITK_NG") {
                if let Ok(n) = v.parse() {
                    return n;
                }
            }
        }
        4
    })
}

#[derive(Clone, Copy, Debug)]
pub struct GemmPwSpec {
    /// 픽셀 수 (H*W)
    pub m: u32,
    /// 입력 채널그룹 수 (실제 ceil(cin/4) — 가중치는 4그룹 배수로 패딩돼 있음)
    pub kg: u32,
    /// 출력 채널그룹 수
    pub ng: u32,
    pub act: Activation,
    pub residual: bool,
    /// 활성화 스토리지 dtype
    pub dt: DType,
    /// 가중치 스토리지 dtype — 활성화와 독립 (conv_igemm과 같은 규약)
    pub wdt: DType,
}

impl GemmPwSpec {
    fn variant(&self) -> GemmVariant {
        pick_variant(self.m, self.kg, self.ng)
    }

}

impl KernelSpec for GemmPwSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "gemm_pw v={:?}{} M{} KG{} NG{} {} dt={}/w{}",
            self.variant(),
            match self.variant() {
                GemmVariant::Gemv => format!("/w{}", gemv_width(self.kg)),
                _ => String::new(),
            },
            self.m,
            self.kg,
            self.ng,
            epilogue::key_fragment(true, self.act, self.residual),
            self.dt.tag(),
            self.wdt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let kt = self.kg.div_ceil(TK);
        let consts = format!(
            "const M: u32 = {}u;\nconst KG: u32 = {}u;\nconst NG: u32 = {}u;\nconst KT: u32 = {}u;",
            self.m, self.kg, self.ng, kt
        );
        let (res_b, out_b) = gemm_tile::binding_slots(self.residual);

        let types = type_aliases(self.dt, self.wdt);

        match self.variant() {
            GemmVariant::Small => {
                let nblk = self.m.div_ceil(4);
                let consts_s = format!("{consts}\nconst NBLK: u32 = {nblk}u;");
                let mut store = crate::kernels::common::writer::W::new();
                for p in 0..4u32 {
                    if p > 0 {
                        store.line(format!("if (n{p}) {{"));
                    }
                    store.line(format!("  let oi_{p} = (px0 + {p}u) * NG + ng;"));
                    store.line(format!("  var v_{p} = acc[{p}];"));
                    let epi = epilogue::emit(
                        &format!("v_{p}"),
                        Some("vec4f(BIAS[ng])"),
                        self.act,
                        self.residual.then_some(&*format!("vec4f(RES[oi_{p}])")),
                    );
                    for line in epi.lines() {
                        store.line(format!("  {line}"));
                    }
                    store.line(format!("  OUT[oi_{p}] = sv4(v_{p});"));
                    if p > 0 {
                        store.line("}");
                    }
                }
                fill(
                    TEMPLATE_SMALL,
                    &[
                        ("TYPES", types),
                        ("CONSTS", consts_s),
                        ("RES_BINDING", res_b),
                        ("OUT_BINDING", out_b),
                        ("STORE4", store.done()),
                    ],
                )
            }
            GemmVariant::Gemv => {
                let epi = epilogue::emit(
                    "acc2",
                    Some("vec4f(BIAS[ng])"),
                    self.act,
                    self.residual.then_some("vec4f(RES[out_idx])"),
                );
                let wgt = gemv_width(self.kg);
                fill(
                    TEMPLATE_GEMV,
                    &[
                        ("TYPES", types),
                        ("CONSTS", format!("{consts}\nconst WGT: u32 = {wgt}u;")),
                        ("RES_BINDING", res_b),
                        ("OUT_BINDING", out_b),
                        ("WG_DECL", format!("var<workgroup> sh: array<vec4f, {wgt}>;")),
                        ("WG_ATTR", format!("@compute @workgroup_size({wgt})")),
                        ("EPILOGUE", epi),
                    ],
                )
            }
            GemmVariant::Splitk => {
                let cells = self.m * self.ng;
                let (split, cpw) = splitk_geometry(cells);
                let (pxt, ngt) = splitk_tile(self.ng, cpw);
                let ngtc = self.ng.div_ceil(ngt);
                let kc = self.kg.div_ceil(split);
                let consts_sk = format!(
                    "{consts}\nconst CPW: u32 = {cpw}u;\nconst SPLIT: u32 = {split}u;\nconst KC: u32 = {kc}u;\nconst PXT: u32 = {pxt}u;\nconst NGT: u32 = {ngt}u;\nconst NGTC: u32 = {ngtc}u;"
                );
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
                        ("CONSTS", consts_sk),
                        ("RES_BINDING", res_b),
                        ("OUT_BINDING", out_b),
                        ("EPILOGUE", epi),
                    ],
                )
            }
            GemmVariant::Tiled => fill(
                TEMPLATE_TILED,
                &[
                    ("TYPES", types),
                    ("CONSTS", consts),
                    ("RES_BINDING", res_b),
                    ("OUT_BINDING", out_b),
                    ("MAC_UNROLLED", gemm_tile::emit_mac_unrolled()),
                    ("STORE_UNROLLED", gemm_tile::emit_store_unrolled(self.act, self.residual)),
                ],
            ),
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
            GemmVariant::Small => [(self.m.div_ceil(4) * self.ng).div_ceil(256), 1, 1],
            GemmVariant::Gemv => [self.m * self.ng, 1, 1],
            GemmVariant::Splitk => {
                let cells = self.m * self.ng;
                let (_, cpw) = splitk_geometry(cells);
                let (pxt, ngt) = splitk_tile(self.ng, cpw);
                [self.m.div_ceil(pxt) * self.ng.div_ceil(ngt), 1, 1]
            }
            GemmVariant::Tiled => [self.m.div_ceil(TM), self.ng.div_ceil(TN_NG), 1],
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
        // 양 변형 × 전 활성화 × residual × 에지 shape (Splitk: 144×240·60, 셀 비배수 에지)
        for (m, kg, ng) in [
            (1, 144, 36),
            (144, 32, 6),
            (9216, 4, 16),
            (147456, 1, 1),
            (768, 2, 3),
            (144, 240, 40),
            (144, 40, 240),
            (100, 17, 21),
        ] {
            for act in ALL {
                for residual in [false, true] {
                    // (활성화, 가중치) 정밀도 조합 — 혼합(f32 활성/f16 가중치) 포함
                    for (dt, wdt) in [
                        (DType::F32, DType::F32),
                        (DType::F32, DType::F16),
                        (DType::F16, DType::F16),
                    ] {
                        let spec = GemmPwSpec { m, kg, ng, act, residual, dt, wdt };
                        validate_wgsl(&spec.wgsl(&caps));
                    }
                }
            }
        }
    }

    #[test]
    fn variant_policy() {
        // SE M=1 + 큰 K → Gemv
        assert_eq!(pick_variant(1, 144, 36), GemmVariant::Gemv);
        // 심층 M=144 + K 큼 → Splitk (타일 그리드 기아 해소)
        assert_eq!(pick_variant(144, 240, 40), GemmVariant::Splitk);
        assert_eq!(pick_variant(144, 40, 240), GemmVariant::Splitk);
        assert_eq!(pick_variant(144, 168, 28), GemmVariant::Splitk);
        // M 큰 전해상도 + NG 큼 → Tiled 유지
        assert_eq!(pick_variant(9216, 16, 16), GemmVariant::Tiled);
        // NG 작아도 K가 크면 Splitk — Small은 스레드당 4픽셀이라 워크그룹이 안 나온다.
        // (18×32 120→40: Small이면 6 워크그룹, Splitk면 216개. 실측 −30µs/op)
        assert_eq!(pick_variant(576, 30, 10), GemmVariant::Splitk);
        assert_eq!(pick_variant(2304, 18, 6), GemmVariant::Splitk);
        // K가 작으면(청크가 빈다) NG와 무관하게 Small
        assert_eq!(pick_variant(36864, 6, 4), GemmVariant::Small);
        assert_eq!(pick_variant(36864, 1, 4), GemmVariant::Small);
        assert_eq!(pick_variant(512, 8, 8), GemmVariant::Small);
    }

    #[test]
    fn gemv_width_tracks_kg() {
        // SE 확장 FC(240→960)는 KG=60 — 256스레드를 쓰면 레인 76%가 논다
        assert_eq!(gemv_width(60), 64);
        assert_eq!(gemv_width(30), 32);
        // 압축 FC(960→240)는 KG=240 → 상한 256 그대로
        assert_eq!(gemv_width(240), 256);
        assert_eq!(gemv_width(400), 256);
        // SIMD 폭 아래로는 안 내린다
        assert_eq!(gemv_width(9), 32);
    }
}
