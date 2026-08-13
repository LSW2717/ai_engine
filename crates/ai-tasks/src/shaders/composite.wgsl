// 마스크 + 카메라 프레임 → 최종 화면 합성 (풀스크린 삼각형 1패스).
//
// ⚠ 프래그먼트 스테이지에서 **스토리지 버퍼를 읽지 않는다**.
// WebGPU는 maxStorageBuffersInFragmentStage가 구현마다 다르고 0일 수 있다
// (compat 모드/일부 브라우저). 검증 오류는 비동기라 조용히 실패하고, 캔버스가
// 지워진 채 남아 "검은 마스크"가 된다 — 사파리에서 실제로 그랬다.
// 대신 출력 버퍼를 rgba32float 텍스처로 복사해 textureLoad로 읽는다.
// textureLoad는 필터링을 안 하므로 float32-filterable도 필요 없다.
//
// 합성까지 여기서 끝낸다: out = bg*(1-m) + fg*m.
// 캔버스 2D 블렌드(multiply/difference/lighter)로 하면 브라우저마다 결과가
// 달라진다 — 사파리에서 합성만 깨졌던 원인이 그것이다. 셰이더에서 하면 없다.

struct P {
    w: u32, h: u32, cg: u32, ch: u32,
    mode: u32,      // 0 = 합성, 1 = 마스크만
    bg: u32,        // 0 = 그라디언트, 1 = 검정, 2 = 프레임 블러(근사)
    _pad0: u32, _pad1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var SRC: texture_2d<f32>;   // 모델 출력 (NHWC-C4)
@group(0) @binding(2) var FRAME: texture_2d<f32>; // 카메라 프레임 (rgba8)
@group(0) @binding(3) var SAMP: sampler;

// 풀스크린 삼각형 (버텍스 버퍼 없음)
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// 마스크는 모델 해상도라 표시 해상도로 확대된다. textureLoad는 보간이 없으니
// 이웃 4텍셀을 직접 바이리니어로 섞어 경계 계단을 없앤다.
fn maskAt(uv: vec2<f32>) -> f32 {
    let fw = f32(p.w);
    let fh = f32(p.h);
    let t = vec2<f32>(uv.x * fw - 0.5, uv.y * fh - 0.5);
    let b = floor(t);
    let f = t - b;
    let cg = i32(p.cg);
    let lane = i32(p.ch & 3u);
    let grp = i32(p.ch >> 2u);
    var acc = 0.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let xi = clamp(i32(b.x) + dx, 0, i32(p.w) - 1);
            let yi = clamp(i32(b.y) + dy, 0, i32(p.h) - 1);
            let v4 = textureLoad(SRC, vec2<i32>(xi * cg + grp, yi), 0);
            let wgt = (select(1.0 - f.x, f.x, dx == 1)) * (select(1.0 - f.y, f.y, dy == 1));
            acc = acc + clamp(v4[lane], 0.0, 1.0) * wgt;
        }
    }
    return acc;
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(FRAME));
    let uv = pos.xy / dim;
    let a = maskAt(uv);
    if (p.mode == 1u) {
        return vec4<f32>(a, a, a, 1.0);   // 마스크만 보기
    }
    let fg = textureSampleLevel(FRAME, SAMP, uv, 0.0).rgb;
    var bg: vec3<f32>;
    if (p.bg == 1u) {
        bg = vec3<f32>(0.0);
    } else if (p.bg == 2u) {
        // 프레임 블러 근사 — 넓게 흩뿌린 9탭 (배경 이미지 없이도 가상배경 느낌)
        var sum = vec3<f32>(0.0);
        let r = 6.0 / dim;
        for (var i = -1; i <= 1; i = i + 1) {
            for (var j = -1; j <= 1; j = j + 1) {
                sum = sum + textureSampleLevel(
                    FRAME, SAMP, uv + vec2<f32>(f32(i), f32(j)) * r, 0.0).rgb;
            }
        }
        bg = sum / 9.0;
    } else {
        // 그라디언트
        let t = clamp((uv.x + uv.y) * 0.5, 0.0, 1.0);
        bg = mix(vec3<f32>(0.118, 0.227, 0.372), vec3<f32>(0.482, 0.176, 0.369), t);
    }
    let outc = bg * (1.0 - a) + fg * a;
    return vec4<f32>(outc, 1.0);
}
