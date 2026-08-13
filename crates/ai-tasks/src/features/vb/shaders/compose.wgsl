// 최종 합성 — 배경 4모드(원본/블러/단색/이미지) + coverage smoothstep +
// light wrapping(이미지 배경) + spill suppression(엣지 채도 제거) +
// edge darkening + 밝기/흑백(배경만 — 웹 규약). v-ai 파리티 스펙.
//
// 확장 슬롯: 조명(relight)·프레이밍(person crop)·인물교체(replace 모드)는
// 이 셰이더에 uniform 분기로 추가된다 — INTEGRATION.md §1.6.

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var mask_hi: texture_2d<f32>;
@group(0) @binding(2) var bg_blur: texture_2d<f32>;
@group(0) @binding(3) var bg_image: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
struct P {
    bg_mode: u32,        // 0 원본 1 블러 2 단색 3 이미지
    blur_strength: f32,
    brightness: f32,
    grayscale: f32,
    cov: vec2f,
    spill: f32,
    edge_dark: f32,
    bg_color: vec4f,
    bg_scale: vec2f,     // 이미지 cover 크롭
    bg_offset: vec2f,
    light_wrap: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // 스튜디오 조명 (v-ai RELIGHT 등가): x=enabled, y=ambient, z=aspect
    relight: vec4f,
    // xy=위치, z=radius, w=intensity / rgb=색, w=target(0 인물 1 배경 2 전체)
    lights: array<vec4f, 4>,
}
@group(0) @binding(5) var<uniform> p: P;

fn apply_relight(base: vec3f, pm: f32, uv: vec2f) -> vec3f {
    if (p.relight.x < 0.5) { return base; }
    var acc = vec3f(0.0);
    for (var i = 0; i < 2; i += 1) {
        let pr = p.lights[i * 2];
        let ct = p.lights[i * 2 + 1];
        var d = uv - pr.xy;
        d.x *= p.relight.z;
        var fall = 1.0 - smoothstep(0.0, max(pr.z, 1e-3), length(d));
        fall *= fall;
        let w = select(select(1.0, 1.0 - pm, ct.w < 1.5), pm, ct.w < 0.5);
        acc += ct.rgb * (pr.w * fall * w);
    }
    let lit = base * (p.relight.y + acc);
    // 소프트 하이라이트 롤오프 — 클리핑 대신 1.0 초과분을 부드럽게 누른다
    return lit / (1.0 + max(vec3f(0.0), lit - vec3f(1.0)));
}

fn gray(c: vec3f) -> f32 {
    return dot(c, vec3f(0.299, 0.587, 0.114));
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let color = textureSampleLevel(frame, samp, in.uv, 0.0).rgb;
    let raw = textureSampleLevel(mask_hi, samp, in.uv, 0.0).r;

    var bg: vec3f;
    if (p.bg_mode == 0u) {
        bg = color;
    } else if (p.bg_mode == 1u) {
        let blurred = textureSampleLevel(bg_blur, samp, in.uv, 0.0).rgb;
        bg = mix(color, blurred, clamp(p.blur_strength, 0.0, 1.0));
    } else if (p.bg_mode == 2u) {
        bg = p.bg_color.rgb;
    } else {
        bg = textureSampleLevel(bg_image, samp, in.uv * p.bg_scale + p.bg_offset, 0.0).rgb;
    }

    var fg = color;
    if (p.bg_mode == 3u) {
        // light wrapping — 배경 빛이 인물 윤곽에 감기는 효과 (screen 블렌드)
        let lwm = 1.0 - max(0.0, raw - p.cov.y) / (1.0 - p.cov.y);
        let lw = p.light_wrap * lwm * bg;
        fg = 1.0 - (1.0 - fg) * (1.0 - lw);
    }

    let pm = smoothstep(p.cov.x, p.cov.y, raw);
    let edge = clamp(1.0 - abs(pm * 2.0 - 1.0), 0.0, 1.0);
    fg = mix(fg, vec3f(gray(fg)), edge * clamp(p.spill, 0.0, 1.0));
    bg = mix(bg, vec3f(gray(bg)), clamp(p.grayscale, 0.0, 1.0)) * p.brightness;

    var fin = mix(bg, fg, pm);
    fin *= 1.0 - edge * clamp(p.edge_dark, 0.0, 1.0) * 0.06;
    fin = apply_relight(fin, pm, in.uv);
    return vec4f(fin, 1.0);
}
