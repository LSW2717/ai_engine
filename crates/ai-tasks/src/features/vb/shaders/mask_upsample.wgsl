// 마스크 업샘플 — joint bilateral filter (프레임을 가이드로 저해상 마스크를
// 프레임 해상도로): 공간 가우시안 × 색 가우시안 가중 평균. 웹/모바일 공통
// 규약(σ_space 2.0+blur·3.2, σ_color 0.1+blur·0.36), 스텝/반경은 Rust가
// σ에서 유도해 uniform으로 준다 (웹 updateSigmaSpace 등가).

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var mask_lo: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
// v-ai segmentationTexture는 NEAREST — 마스크만 최근접 (프레임 가이드는 선형)
@group(0) @binding(4) var samp_near: sampler;
struct P {
    texel: vec2f,       // 출력(프레임) 해상도 텍셀
    step: f32,
    radius: f32,
    offset: f32,
    sigma_texel: f32,
    sigma_color: f32,
    _pad: f32,
}
@group(0) @binding(3) var<uniform> p: P;

fn gauss(x: f32, sigma: f32) -> f32 {
    let c = -0.5 / (sigma * sigma * 4.0 + 1.0e-6);
    return exp(x * x * c);
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let center = textureSampleLevel(frame, samp, in.uv, 0.0).rgb;
    var num = 0.0;
    var den = 0.0;
    for (var i = -p.radius + p.offset; i <= p.radius; i += p.step) {
        for (var j = -p.radius + p.offset; j <= p.radius; j += p.step) {
            let coord = in.uv + vec2f(j, i) * p.texel;
            let fc = textureSampleLevel(frame, samp, coord, 0.0).rgb;
            let m = textureSampleLevel(mask_lo, samp_near, coord, 0.0).r;
            let w = gauss(distance(in.uv, coord), p.sigma_texel)
                * gauss(distance(center, fc), p.sigma_color);
            den += w;
            num += w * m;
        }
    }
    return vec4f(num / max(den, 1.0e-5), 0.0, 0.0, 1.0);
}
