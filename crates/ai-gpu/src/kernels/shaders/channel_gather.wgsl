// 채널 gather — Concat(≤3입력)/Chcopy/비정렬 뷰 실체화의 통합 커널.
// 출력 텍셀(vec4)을 완전히 채우므로 read-modify-write가 없다 → 단일 디스패치로
// 파트 경계가 텍셀 중간에 걸려도 안전하다.
// 슬롯: TYPES, IN_BINDINGS, OUT_BINDING, CONSTS, LANES (lane별 gather 표현식)

//@TYPES

//@IN_BINDINGS
//@OUT_BINDING
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= TOTAL) {
        return;
    }
    let px = i / CGO;
    let cg = i - px * CGO;
    let oc0 = cg * 4u;
    //@LANES
    OUT[i] = sv4(vec4f(l0, l1, l2, l3));
}
