// average pool — dw conv에서 가중치를 뺀 슬라이딩 윈도우.
// 스레드 = 1출력픽셀 × 1채널그룹, 워크그룹 (8,8,1), wg.z = 채널그룹.
// 분모는 k*k 고정(CPU 레퍼런스와 동일 규약). RVM은 k==s, pad=0 경로만 쓴다.
// 슬롯: OUT_BINDING, CONSTS, TAPS_UNROLLED

@group(0) @binding(1) var<storage, read> IN: array<vec4f>;
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let cg = wg.z;
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= OW || oy >= OH) {
        return;
    }
    var acc = vec4f(0.0);
    //@TAPS_UNROLLED
    OUT[(oy * OW + ox) * CG + cg] = acc * INV_K;
}
