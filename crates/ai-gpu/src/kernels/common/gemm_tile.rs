//! 32×32×16 GEMM 타일의 MAC 코어·스토어 codegen — gemm_pw(tiled)와
//! conv_igemm(tiled)이 공유한다. implicit GEMM은 A-타일 로드만 다르다.
//!
//! 전제(템플릿 계약): 워크그룹 64스레드, `ng_t = t % 8`, `px_t = t / 8`,
//! `Ash[[px_local(32)][kk(4)]]`, `Bsh[[kk(4)][ng_t(8)][j(4)]]`,
//! `acc: array<vec4f, 4>`, 상수 `M`/`NG`, 변수 `tile_px`/`ng`.

use ai_core::Activation;

use super::epilogue;
use super::writer::W;

/// kk(4) × p(4) 완전 언롤 MAC — 가중치 4페치당 16 dot4
pub fn emit_mac_unrolled() -> String {
    let mut mac = W::new();
    for kk in 0..4 {
        mac.line(format!("{{ let b{kk} = ({kk}u * 8u + ng_t) * 4u;"));
        mac.line(format!(
            "  let w0_{kk} = Bsh[b{kk}]; let w1_{kk} = Bsh[b{kk} + 1u]; \
             let w2_{kk} = Bsh[b{kk} + 2u]; let w3_{kk} = Bsh[b{kk} + 3u];"
        ));
        for p in 0..4 {
            mac.line(format!(
                "  {{ let a = Ash[(px_t * 4u + {p}u) * 4u + {kk}u]; \
                 acc[{p}] = acc[{p}] + vec4f(dot(w0_{kk}, a), dot(w1_{kk}, a), \
                 dot(w2_{kk}, a), dot(w3_{kk}, a)); }}"
            ));
        }
        mac.line("}");
    }
    mac.done()
}

/// p(4) 언롤 스토어 — M 가드 + 공유 에필로그(bias → act → residual).
/// bias/residual 로드는 `vec4f(...)`로 감싸고 스토어는 `sv4(...)` — f16 스토리지 대응.
pub fn emit_store_unrolled(act: Activation, residual: bool) -> String {
    let mut store = W::new();
    for p in 0..4 {
        store.line(format!("let px_{p} = tile_px + px_t * 4u + {p}u;"));
        store.line(format!("if (px_{p} < M) {{"));
        store.line(format!("  let oi_{p} = px_{p} * NG + ng;"));
        store.line(format!("  var v_{p} = acc[{p}];"));
        let epi = epilogue::emit(
            &format!("v_{p}"),
            Some("vec4f(BIAS[ng])"),
            act,
            residual.then_some(&*format!("vec4f(RES[oi_{p}])")),
        );
        for line in epi.lines() {
            store.line(format!("  {line}"));
        }
        store.line(format!("  OUT[oi_{p}] = sv4(v_{p});"));
        store.line("}");
    }
    store.done()
}

/// residual 유무에 따른 RES/OUT 바인딩 선언 (모든 conv 계열 공통, sv4 = dtype 별칭)
pub fn binding_slots(residual: bool) -> (String, String) {
    if residual {
        (
            "@group(0) @binding(4) var<storage, read> RES: array<sv4>;".to_string(),
            "@group(0) @binding(5) var<storage, read_write> OUT: array<sv4>;".to_string(),
        )
    } else {
        (
            String::new(),
            "@group(0) @binding(4) var<storage, read_write> OUT: array<sv4>;".to_string(),
        )
    }
}
