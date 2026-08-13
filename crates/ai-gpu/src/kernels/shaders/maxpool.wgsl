// max pool — avgpool의 max 판. 패딩 영역은 -inf 항등(select 마스크),
// pad_c(채널 끝 제로패딩 — BlazeFace Pad 접기, ops/pool.rs 참조)는
// C 이상 레인을 0으로 마스킹해 만든다.
// 스레드 = 출력 텍셀 1개, 1D 평탄 인덱스 (cg 최하위, 합체 액세스).
// 슬롯: OUT_BINDING, CONSTS, TAPS_UNROLLED

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= OH * OW * OCG) {
        return;
    }
    let cg = i % OCG;
    let px = i / OCG;
    let ox = px % OW;
    let oy = px / OW;
    var acc = vec4f(-3.0e38);
    if (cg < CGIN) {
        //@TAPS_UNROLLED
    }
    // 채널 c 이상 레인(그룹 꼬리 + pad_c 구간)은 0
    let ch = vec4f(f32(cg * 4u)) + vec4f(0.0, 1.0, 2.0, 3.0);
    acc = select(vec4f(0.0), acc, ch < vec4f(CF));
    OUT[i] = sv4(acc);
}
