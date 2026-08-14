// (h=1) W↔C 2D 전치 — in(1,W,C) NHWC-C4 → out(1,C,W).
// MLP-Mixer 토큰↔채널 (face_blendshapes). C4 레인 재배치라 실복사.
// 스레드 1개 = 출력 vec4 그룹 1개 (flatten.wgsl과 같은 규약).

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read_write> OUT: array<sv4>;
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= C * CGOUT) {
        return;
    }
    let px_o = g / CGOUT;  // 출력 픽셀 = 입력 채널 (0..C)
    let cg_o = g % CGOUT;
    var v = vec4f(0.0);
    for (var l = 0u; l < 4u; l++) {
        let ch_o = cg_o * 4u + l;  // 출력 채널 = 입력 픽셀 (0..W)
        if (ch_o < W) {
            v[l] = f32(IN[ch_o * CGIN + px_o / 4u][px_o % 4u]);
        }
    }
    OUT[g] = sv4(v);
}
