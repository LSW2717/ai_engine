// pointwise 1×1 conv — GEMV 변형 (M×NG 극소 + K 큼: SE squeeze/excite의 M=1)
// 워크그룹 = 출력 vec4 셀 하나. 스레드가 kg를 WGT-stride로 나눠 부분합,
// 공유메모리 트리 리덕션.
//
// ⚠ 워크그룹 폭은 **KG에 맞춘다** (256 고정 아님). SE의 확장 FC(240→960)는
// KG=60인데 256스레드를 쓰면 레인 76%가 놀고 리덕션 트리는 그대로 8라운드를
// 돈다 — 같은 460KB를 읽는 압축 FC(960→240, KG=240)가 48GB/s인데 확장 FC는
// 27GB/s로 떨어지던 원인이다.
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS, WG_DECL, WG_ATTR, EPILOGUE

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<wv4>;
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS

//@WG_DECL

//@WG_ATTR
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    // wg.x = px * NG + ng
    let px = wg.x / NG;
    let ng = wg.x - px * NG;
    var acc = vec4f(0.0);
    for (var kg = t; kg < KG; kg = kg + WGT) {
        let a = vec4f(IN[px * KG + kg]);
        let wb = (kg * NG + ng) * 4u;
        let w0 = vec4f(W[wb]);
        let w1 = vec4f(W[wb + 1u]);
        let w2 = vec4f(W[wb + 2u]);
        let w3 = vec4f(W[wb + 3u]);
        acc = acc + vec4f(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a));
    }
    sh[t] = acc;
    workgroupBarrier();
    for (var s = WGT >> 1u; s > 0u; s = s >> 1u) {
        if (t < s) {
            sh[t] = sh[t] + sh[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        var acc2 = sh[0];
        let out_idx = px * NG + ng;
        //@EPILOGUE
        OUT[out_idx] = sv4(acc2);
    }
}
