// pointwise 1×1 conv — small4 변형: 스레드 = 연속 4픽셀 × 1출력그룹.
// (kg당 가중치 4페치를 4픽셀이 공유 — WebGL2 SPAN=4의 pw 등가.
//  1×1은 공간 구조가 없어 행 경계 무관하게 연속 픽셀 4개면 된다)
// NG 퇴화(타일 낭비)·소형 K에서 tiled보다 빠르다.
// 슬롯: TYPES(sv4 별칭), RES_BINDING, OUT_BINDING, CONSTS, STORE4

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<sv4>;
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= NBLK * NG) {
        return;
    }
    let blk = i / NG;
    let ng = i - blk * NG;
    let px0 = blk * 4u;
    let n1 = px0 + 1u < M;
    let n2 = px0 + 2u < M;
    let n3 = px0 + 3u < M;
    var acc = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    for (var kg = 0u; kg < KG; kg = kg + 1u) {
        let wb = (kg * NG + ng) * 4u;
        let w0 = vec4f(W[wb]);
        let w1 = vec4f(W[wb + 1u]);
        let w2 = vec4f(W[wb + 2u]);
        let w3 = vec4f(W[wb + 3u]);
        let a0 = vec4f(IN[px0 * KG + kg]);
        acc[0] = acc[0] + vec4f(dot(w0, a0), dot(w1, a0), dot(w2, a0), dot(w3, a0));
        if (n1) {
            let a = vec4f(IN[(px0 + 1u) * KG + kg]);
            acc[1] = acc[1] + vec4f(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a));
        }
        if (n2) {
            let a = vec4f(IN[(px0 + 2u) * KG + kg]);
            acc[2] = acc[2] + vec4f(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a));
        }
        if (n3) {
            let a = vec4f(IN[(px0 + 3u) * KG + kg]);
            acc[3] = acc[3] + vec4f(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a));
        }
    }
    //@STORE4
}
