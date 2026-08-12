// SE 게이트 — gpool(채널 평균) → FC1(act1) [→ FC2(act2)] 를 워크그룹 하나로.
// M=1 벡터 경로는 op당 일감이 극소해 디스패치+배리어 비용이 지배적이었다
// (체인당 2~3 디스패치 → 1). 평균/중간값은 공유메모리로 전달.
// 슬롯: TYPES, BINDINGS, CONSTS, SH_DECL, FC1_STORE(중간 저장 or 최종 출력), FC2(선택)

//@TYPES

//@BINDINGS
//@CONSTS

//@SH_DECL

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    // 1) 채널그룹별 평균 — 스레드 = cg (strided)
    for (var cg = t; cg < CG_IN; cg = cg + 256u) {
        var acc = vec4f(0.0);
        for (var p = 0u; p < HW; p = p + 1u) {
            acc = acc + vec4f(IN[p * CG_IN + cg]);
        }
        mean_sh[cg] = acc * INV_HW;
    }
    workgroupBarrier();
    // 2) FC1 — 스레드 = 출력 채널그룹
    for (var ng = t; ng < CG_MID; ng = ng + 256u) {
        var acc = vec4f(0.0);
        for (var kg = 0u; kg < CG_IN; kg = kg + 1u) {
            let a = mean_sh[kg];
            let wb = (kg * CG_MID + ng) * 4u;
            acc = acc
                + vec4f(
                    dot(vec4f(W1[wb]), a),
                    dot(vec4f(W1[wb + 1u]), a),
                    dot(vec4f(W1[wb + 2u]), a),
                    dot(vec4f(W1[wb + 3u]), a),
                );
        }
        var v = acc + vec4f(B1[ng]);
        //@ACT1
        //@FC1_STORE
    }
    //@FC2
}
