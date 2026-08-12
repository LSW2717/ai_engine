// pointwise 1×1 conv — tiled GEMM 변형 (M ≥ 512)
// 타일: TM=32px × TN=8출력그룹(32ch), k-step TK=4그룹(16ch). 워크그룹 64스레드.
// 스레드 = 4픽셀 × 1출력그룹 → acc[4] — WebGL2 엔진의 4×4 레지스터 블록
// (가중치 4페치 → 16 dot4)을 공유메모리 급전으로 이식한 것.
// 공유메모리 4KiB — WebGPU 기본 한도 16KiB의 1/4.
// 슬롯: TYPES(sv4 별칭), RES_BINDING, OUT_BINDING, CONSTS, MAC_UNROLLED, STORE_UNROLLED

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<sv4>;
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

    for (var kt = 0u; kt < KT; kt = kt + 1u) {
        // A 타일 협조 로드 (스레드당 2 vec4)
        for (var l = t; l < 128u; l = l + 64u) {
            let px_l = l / 4u;
            let kk = l % 4u;
            let px = tile_px + px_l;
            let kg = kt * 4u + kk;
            var v = vec4f(0.0);
            if (px < M && kg < KG) {
                v = vec4f(IN[px * KG + kg]);
            }
            Ash[l] = v;
        }
        // B 타일 협조 로드 (스레드당 2 vec4) — W의 kg 차원은 패커가 4의 배수로 패딩
        for (var l = t; l < 128u; l = l + 64u) {
            let kk = l / 32u;
            let rem = l - kk * 32u;
            let ng_l = tile_ng + rem / 4u;
            let j = rem % 4u;
            var v = vec4f(0.0);
            if (ng_l < NG) {
                v = vec4f(W[((kt * 4u + kk) * NG + ng_l) * 4u + j]);
            }
            Bsh[l] = v;
        }
        workgroupBarrier();
        //@MAC_UNROLLED
        workgroupBarrier();
    }

    if (ng < NG) {
        //@STORE_UNROLLED
    }
}
