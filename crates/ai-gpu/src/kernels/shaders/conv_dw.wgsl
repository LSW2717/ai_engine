// depthwise conv k3/k5, s1/s2 — 채널당 독립이라 GEMM이 아닌 슬라이딩 윈도우.
// 스레드 = 가로 4픽셀 블록 × 1채널그룹 (direct4와 같은 행 재사용: 같은 행의 kx
// 탭들이 겹치는 열을 공유 — k3 s1이면 4px에 행당 6로드, 1px/스레드 대비 2.6배↓).
// 인덱스는 cg 최하위(합체): i = blk*CG + cg. 가중치는 tap-major [tap][cg].
// 경계는 clamp+마스크, 행 밖이면 스킵.
// 슬롯: TYPES(sv4 별칭), RES_BINDING, OUT_BINDING, CONSTS, ROWS, STORE4

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<sv4>; // [tap][cg]
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= NBLK * CG) {
        return;
    }
    let cg = i % CG;
    let blk = i / CG;
    let oy = blk / BPR;
    let ox0 = (blk - oy * BPR) * PB;
    //@ACC_DECL
    //@ROWS
    //@STORE4
}
