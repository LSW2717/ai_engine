// 일반 conv (k>1, groups=1) — implicit GEMM tiled 변형.
// gemm_pw_tiled와 같은 32×32×16 타일 골격. 최적화 구조:
// - 탭은 codegen 외부 언롤(좌표 오프셋 리터럴) — 루프 내 tap 디코드 나눗셈 제거
// - 픽셀 디코드(px→oy,ox)는 스레드당 1회 사전 계산 — A-로드 나눗셈 완전 제거
// - 가중치 tap-major 레이아웃이라 B 진행은 순수 선형
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS, TAP_LOOPS, STORE_UNROLLED

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<wv4>; // [tap][kgp][ng][j]
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS

var<workgroup> Ash: array<vec4f, 128>; // [px_local(32)][kk(4)]
var<workgroup> Bsh: array<vec4f, 128>; // [kk(4)][ng_t(8)][j(4)]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let tile_px = wg.x * 32u;
    let tile_ng = wg.y * 8u;
    let ng_t = t % 8u;
    let px_t = t / 8u;
    let ng = tile_ng + ng_t;
    var acc = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));

    // A-로드 2슬롯(l = t, t+64)의 픽셀 좌표 사전 계산 — 탭/kb 루프에서 나눗셈 0
    let la = t;
    let lb = t + 64u;
    let pxa = tile_px + la / 4u;
    let pxb = tile_px + lb / 4u;
    let kka = la % 4u; // (t+64)%4 == t%4
    let oya = i32(pxa / OW);
    let oxa = i32(pxa - u32(oya) * OW);
    let oyb = i32(pxb / OW);
    let oxb = i32(pxb - u32(oyb) * OW);
    let va_ok = pxa < M;
    let vb_ok = pxb < M;

    // B-로드 2슬롯의 (kk, ng, j) 사전 계산
    let bkk_a = la / 32u;
    let brem_a = la - bkk_a * 32u;
    let bng_a = tile_ng + brem_a / 4u;
    let bj_a = brem_a % 4u;
    let bkk_b = lb / 32u;
    let brem_b = lb - bkk_b * 32u;
    let bng_b = tile_ng + brem_b / 4u;
    let bj_b = brem_b % 4u;

    //@TAP_LOOPS

    if (ng < NG) {
        //@STORE_UNROLLED
    }
}
