// depthwise conv k3/k5, s1/s2 — 채널당 독립이라 GEMM이 아닌 슬라이딩 윈도우.
// 스레드 = 1출력픽셀 × 1채널그룹, 워크그룹 (8,8,1), wg.z = 채널그룹.
// 그룹의 탭 가중치(9/25 vec4)를 공유메모리에 1회 적재 후 전원이 사용.
// 탭은 codegen이 stride/pad를 박아 완전 언롤, 경계는 branchless 마스크.
// 슬롯: TYPES(sv4 별칭), WSH_DECL, RES_BINDING, OUT_BINDING, CONSTS, TAPS_UNROLLED, EPILOGUE

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<sv4>; // [cg][tap]
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS
//@WSH_DECL

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let cg = wg.z;
    if (t < TAPS) {
        Wsh[t] = vec4f(W[cg * TAPS + t]);
    }
    workgroupBarrier();

    let ox = gid.x;
    let oy = gid.y;
    if (ox >= OW || oy >= OH) {
        return;
    }
    var acc = vec4f(0.0);
    //@TAPS_UNROLLED
    let out_idx = (oy * OW + ox) * CG + cg;
    //@EPILOGUE
    OUT[out_idx] = sv4(acc);
}
