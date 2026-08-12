// 일반 conv — split-K 변형 (저해상 심층 k3: direct4가 스레드 기아인 shape)
// 9×16 128->128 k3는 direct4 스레드가 1152개뿐 (M/4 × NG). 셀(픽셀×출력그룹)을
// 1픽셀 단위로 펴고 K(kg)를 SPLIT청크로 나눠 스레드를 SPLIT배로 만든다 —
// gemm_pw_splitk와 같은 워크그룹 내 리덕션, 탭 좌표는 청크 밖에서 계산.
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS, TAPS(탭 언롤 본문), EPILOGUE

//@TYPES

//@BINDINGS
//@CONSTS

var<workgroup> sh: array<vec4f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cl = t % CPW; // 타일 내 셀 (px×ng 2D 타일 — gemm_pw_splitk와 동일한 이유)
    let chunk = t / CPW; // k-청크
    let pxt = wg.x / NGTC;
    let ngt = wg.x - pxt * NGTC;
    let lpx = cl / NGT;
    let px = pxt * PXT + lpx;
    let ng = ngt * NGT + (cl - lpx * NGT);
    let live = px < M && ng < NG;
    var acc = vec4f(0.0);
    if (live) {
        let oy = i32(px / OW);
        let ox = i32(px - u32(oy) * OW);
        let k0 = chunk * KC;
        let k1 = min(k0 + KC, KG);
        //@TAPS
    }
    sh[t] = acc;
    workgroupBarrier();
    if (chunk == 0u && live) {
        var acc2 = sh[t];
        for (var s = 1u; s < SPLIT; s = s + 1u) {
            acc2 = acc2 + sh[t + s * CPW];
        }
        let out_idx = px * NG + ng;
        //@EPILOGUE
        OUT[out_idx] = sv4(acc2);
    }
}
