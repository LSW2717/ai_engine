// bilinear 리사이즈 — RVM 디코더의 2× 업샘플 경로 (일반형으로 구현).
// 스레드 = 출력 텍셀 1개, 1D 평탄 인덱스 (cg가 최하위 축) — 인접 스레드가 인접
// 주소를 읽고 쓴다. concat-into-resize 융합 시 소스가 여럿이고 ld()가 cg 범위로
// 파트를 고른다 (경계는 codegen 상수 if-체인).
// 슬롯: TYPES, BINDINGS, CONSTS, LOAD_FN, COORD

//@TYPES

//@BINDINGS
//@CONSTS

//@LOAD_FN

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
    //@COORD
    let y0f = floor(fy);
    let x0f = floor(fx);
    let y0 = u32(clamp(i32(y0f), 0, IH - 1));
    let y1 = u32(clamp(i32(y0f) + 1, 0, IH - 1));
    let x0 = u32(clamp(i32(x0f), 0, IW - 1));
    let x1 = u32(clamp(i32(x0f) + 1, 0, IW - 1));
    let ty = clamp(fy - y0f, 0.0, 1.0);
    let tx = clamp(fx - x0f, 0.0, 1.0);
    let g00 = ld(y0 * u32(IW) + x0, cg);
    let g01 = ld(y0 * u32(IW) + x1, cg);
    let g10 = ld(y1 * u32(IW) + x0, cg);
    let g11 = ld(y1 * u32(IW) + x1, cg);
    OUT[i] = sv4(mix(mix(g00, g01, tx), mix(g10, g11, tx), ty));
}
