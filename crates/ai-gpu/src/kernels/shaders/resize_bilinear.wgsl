// bilinear 리사이즈 — RVM 디코더의 2× 업샘플 경로 (일반형으로 구현).
// 스레드 = 1출력픽셀 × 1채널그룹, 워크그룹 (8,8,1), wg.z = 채널그룹.
// 좌표 변환(half_pixel/asymmetric)은 codegen 슬롯으로 박힌다.
// 슬롯: OUT_BINDING, CONSTS, COORD

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
    //@COORD
    let y0f = floor(fy);
    let x0f = floor(fx);
    let y0 = u32(clamp(i32(y0f), 0, IH - 1));
    let y1 = u32(clamp(i32(y0f) + 1, 0, IH - 1));
    let x0 = u32(clamp(i32(x0f), 0, IW - 1));
    let x1 = u32(clamp(i32(x0f) + 1, 0, IW - 1));
    let ty = clamp(fy - y0f, 0.0, 1.0);
    let tx = clamp(fx - x0f, 0.0, 1.0);
    let g00 = IN[(y0 * u32(IW) + x0) * CG + cg];
    let g01 = IN[(y0 * u32(IW) + x1) * CG + cg];
    let g10 = IN[(y1 * u32(IW) + x0) * CG + cg];
    let g11 = IN[(y1 * u32(IW) + x1) * CG + cg];
    OUT[(oy * OW + ox) * CG + cg] = mix(mix(g00, g01, tx), mix(g10, g11, tx), ty);
}
