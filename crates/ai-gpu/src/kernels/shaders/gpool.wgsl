// global average pool: [H,W,C] → [1,1,C]
// 채널그룹당 워크그룹 1개(256스레드): strided 적재 → 공유메모리 트리 리듀스.
// SE 블록의 입구 — 출력은 M=1 벡터 텐서로 gemm_pw(small)에 바로 들어간다.
// 슬롯: OUT_BINDING, CONSTS

@group(0) @binding(1) var<storage, read> IN: array<vec4f>;
//@OUT_BINDING
//@CONSTS

var<workgroup> sh: array<vec4f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cg = wg.x;
    var acc = vec4f(0.0);
    for (var i = t; i < HW; i = i + 256u) {
        acc = acc + IN[i * CG + cg];
    }
    sh[t] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            sh[t] = sh[t] + sh[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        OUT[cg] = sh[0] * INV_HW;
    }
}
