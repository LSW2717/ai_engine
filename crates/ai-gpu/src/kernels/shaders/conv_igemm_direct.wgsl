// 일반 conv — direct4 변형: 스레드 = 가로 인접 4픽셀 × 1출력그룹 (acc[4]).
// WebGL2 엔진의 승리 공식 이식: (tap, kg)당 가중치 4페치를 4픽셀이 공유 → 가중치
// 트래픽 1/4. 탭 외부 언롤(오프셋 리터럴), kg 런타임 루프, 공유메모리 없음.
// 작은 cout(타일 낭비)과 작은 K(스템)에서 tiled보다 빠르다.
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS, TAP_LOOPS, STORE4

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<sv4>; // [tap][kgp][ng][j]
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
    let oy = blk / BPR;
    let ox0 = (blk - oy * BPR) * 4u;
    var acc = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    //@TAP_LOOPS
    //@STORE4
}
