// average pool — dw conv에서 가중치를 뺀 슬라이딩 윈도우.
// 스레드 = 출력 텍셀 1개, 1D 평탄 인덱스 (cg 최하위, 합체 액세스).
// 분모는 k*k 고정(CPU 레퍼런스와 동일 규약). RVM은 k==s, pad=0 경로만 쓴다.
// 슬롯: OUT_BINDING, CONSTS, TAPS_UNROLLED

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= OH * OW * CG) {
        return;
    }
    let cg = i % CG;
    let px = i / CG;
    let ox = px % OW;
    let oy = px / OW;
    var acc = vec4f(0.0);
    //@TAPS_UNROLLED
    OUT[i] = sv4(acc * INV_K);
}
