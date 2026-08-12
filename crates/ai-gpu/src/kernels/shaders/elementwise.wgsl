// elementwise binary (+ activation)
// 슬롯: EXTRA_BINDINGS(두 번째 입력/출력 선언), BODY(연산 + 활성화)
// 길이는 P.len으로 받는다 — shape 특수화 이득이 없는 커널이라 파이프라인 캐시를 아낀다.

struct Params {
    scalar: f32,
    cg: u32, // broadcast 모드: 채널그룹 수 (B 벡터 인덱스 = i % cg)
    _p1: u32,
    len: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4f>;
//@EXTRA_BINDINGS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.len) {
        return;
    }
    //@BODY
    O[i] = v;
}
