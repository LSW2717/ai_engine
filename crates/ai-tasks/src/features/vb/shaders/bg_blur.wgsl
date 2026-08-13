// 배경 블러 — 1/5 해상도에서 7탭 분리 가우시안, 인물 픽셀 제외 가중
// (웹 buildBlurPass 등가: 인물 색이 배경 블러로 번지는 것을 마스크로 차단하고,
// 누적 가중치 부족분은 원본 색으로 채운다). h/v 방향은 uniform.

@group(0) @binding(0) var src: texture_2d<f32>;       // 첫 패스: 프레임 / 이후: 이전 블러
@group(0) @binding(1) var mask_hi: texture_2d<f32>;   // 프레임 해상도 인물 마스크
@group(0) @binding(2) var samp: sampler;
struct P { dir: vec2f, _pad: vec2f }
@group(0) @binding(3) var<uniform> p: P;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    var offs = array<f32, 7>(0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0);
    var wts = array<f32, 7>(0.25, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02);
    let center = textureSampleLevel(src, samp, in.uv, 0.0);
    let pm0 = textureSampleLevel(mask_hi, samp, in.uv, 0.0).r;
    var acc = center * (wts[0] * (1.0 - pm0));
    for (var i = 1; i < 7; i += 1) {
        let off = p.dir * offs[i];
        let c1 = in.uv + off;
        let c2 = in.uv - off;
        acc += textureSampleLevel(src, samp, c1, 0.0)
            * (wts[i] * (1.0 - textureSampleLevel(mask_hi, samp, c1, 0.0).r));
        acc += textureSampleLevel(src, samp, c2, 0.0)
            * (wts[i] * (1.0 - textureSampleLevel(mask_hi, samp, c2, 0.0).r));
    }
    return vec4f(acc.rgb + (1.0 - acc.a) * center.rgb, 1.0);
}
