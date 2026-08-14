// 논리 순서(row-major) 보존 desc 재배치 — C4 레인 재패킹 (flatten의 일반화).
// in(px_i × CI ch) → out(px_o × CO ch), 논리 스트림 n은 동일:
//   n = px_o*CO + ch_o = px_i*CI + ch_i
// tf2onnx의 [0,2,3,1]/[0,3,1,2] 언팩·팩 전치가 표적 (face_blendshapes).

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read_write> OUT: array<sv4>;
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= PXO * CGO) {
        return;
    }
    let px_o = g / CGO;
    let cg_o = g % CGO;
    var v = vec4f(0.0);
    for (var l = 0u; l < 4u; l++) {
        let ch_o = cg_o * 4u + l;
        if (ch_o < CO) {
            let n = px_o * CO + ch_o;
            let px_i = n / CI;
            let ch_i = n % CI;
            v[l] = f32(IN[px_i * CGI + ch_i / 4u][ch_i % 4u]);
        }
    }
    OUT[g] = sv4(v);
}
