// 마스크 정제 — 분리 5탭 블러(h/v) 후 엣지 인지 재혼합 (웹 buildMaskPostProcessStage
// 등가): 엣지가 강한 곳은 raw를 살려 윤곽 밀착, 평탄한 곳은 블러로 떨림 제거,
// feather smoothstep + 감마로 경계 톤 조정.
// fs_blur(방향 uniform)와 fs_refine 두 엔트리 — 같은 파일, 파이프라인 2개.

@group(0) @binding(0) var src: texture_2d<f32>;      // blur: 입력 마스크 / refine: blurred
@group(0) @binding(1) var raw: texture_2d<f32>;      // refine 전용 (blur 패스에선 미사용)
@group(0) @binding(2) var samp: sampler;
struct P {
    dir: vec2f,          // blur 패스 방향 텍셀 (refine에선 texel 크기)
    edge_blend: f32,
    edge_gamma: f32,
    edge_feather: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}
@group(0) @binding(3) var<uniform> p: P;

fn m(t: texture_2d<f32>, uv: vec2f) -> f32 {
    return textureSampleLevel(t, samp, uv, 0.0).r;
}

@fragment fn fs_blur(in: VsOut) -> @location(0) vec4f {
    var c = m(src, in.uv) * 0.227027;
    c += m(src, in.uv + p.dir * 1.384615) * 0.316216;
    c += m(src, in.uv - p.dir * 1.384615) * 0.316216;
    c += m(src, in.uv + p.dir * 3.230769) * 0.070270;
    c += m(src, in.uv - p.dir * 3.230769) * 0.070270;
    return vec4f(clamp(c, 0.0, 1.0), 0.0, 0.0, 1.0);
}

@fragment fn fs_refine(in: VsOut) -> @location(0) vec4f {
    let blurred = m(src, in.uv);
    let rawv = m(raw, in.uv);
    let px = m(src, in.uv + vec2f(p.dir.x, 0.0));
    let nx = m(src, in.uv - vec2f(p.dir.x, 0.0));
    let py = m(src, in.uv + vec2f(0.0, p.dir.y));
    let ny = m(src, in.uv - vec2f(0.0, p.dir.y));
    let edge = clamp(abs(px - nx) + abs(py - ny), 0.0, 1.0);
    var refined = mix(blurred, rawv, clamp(p.edge_blend * edge, 0.0, 1.0));
    refined = mix(refined, smoothstep(0.0, 1.0, refined), clamp(p.edge_feather, 0.0, 1.0));
    refined = pow(clamp(refined, 0.0, 1.0), max(0.5, p.edge_gamma));
    return vec4f(refined, 0.0, 0.0, 1.0);
}
