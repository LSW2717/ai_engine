// flatten 실체화 — (px, c) NHWC-C4 → (1, 1, px*c) 채널벡터.
// C4는 픽셀마다 채널을 4레인으로 패딩하므로 c%4≠0이면 평탄 복사가 아니라
// 레인 재배치가 필요하다 (tf2onnx 디텍터 헤드의 Reshape→chcopy, reshape canon 참조).

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read_write> OUT: array<sv4>;
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= (N + 3u) / 4u) {
        return;
    }
    var v = vec4f(0.0);
    for (var l = 0u; l < 4u; l++) {
        let n = g * 4u + l;
        if (n < N) {
            let px = n / CIN;
            let ch = n % CIN;
            v[l] = f32(IN[px * CGIN + ch / 4u][ch % 4u]);
        }
    }
    OUT[g] = sv4(v);
}
