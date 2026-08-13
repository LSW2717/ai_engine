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
    use_fgr: f32,
    texel: vec2f,        // 출력(캔버스) 텍셀 — 이미지 배경 자체 블러가 소비
    // 스튜디오 조명 (v-ai RELIGHT 등가): x=enabled, y=ambient, z=aspect
    relight: vec4f,
    // xy=위치, z=radius, w=intensity / rgb=색, w=target(0 인물 1 배경 2 전체)
    lights: array<vec4f, 4>,
    // 인물 중앙 프레이밍: xyz = scale, cx, cy (scale 1 = 크롭 없음)
    framing: vec4f,
    // mirror/degree 배경 보정 (v-ai updateTransform): 2×2 행렬 열우선 [c0x,c0y,c1x,c1y]
    bg_mat: vec4f,
    // xy = aspect 보정 (v-ai updateAspectComp)
    bg_aspect: vec4f,
}
@group(0) @binding(5) var<uniform> p: P;
// RVM 전경색 (매팅) — use_fgr=0이면 미사용 (프레임 뷰가 더미로 바인딩됨)
@group(0) @binding(6) var fgr_tex: texture_2d<f32>;

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

// 이미지 배경 자체 블러 — v-ai blurBackground 등가 (radius mix(1,12,s),
// 가우시안 exp(-d²/(2r²+1e-4)), 오프셋 texel×2, s≤0.001이면 스킵)
fn blur_bg_image(uv: vec2f) -> vec3f {
    let s = clamp(p.blur_strength, 0.0, 1.0);
    if (s <= 0.001) {
        return textureSampleLevel(bg_image, samp, uv, 0.0).rgb;
    }
    let radius_f = mix(1.0, 12.0, s);
    let radius = i32(radius_f);
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var x = -12; x <= 12; x += 1) {
        for (var y = -12; y <= 12; y += 1) {
            if (abs(x) > radius || abs(y) > radius) {
                continue;
            }
            let off = vec2f(f32(x), f32(y)) * p.texel * 2.0;
            let d = f32(x * x + y * y);
            let w = exp(-d / (2.0 * radius_f * radius_f + 1.0e-4));
            sum += textureSampleLevel(bg_image, samp, uv + off, 0.0).rgb * w;
            total += w;
        }
    }
    return sum / max(total, 1.0e-4);
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    // 인물 중앙 프레이밍 크롭 좌표 — v-ai 규약:
    //  - image/단색(image 스테이지): **인물 레이어만** 크롭, 배경·조명은 화면 고정
    //  - 원본/블러(2D transform 등가): 합성 전체 크롭 — 배경·조명도 크롭 좌표
    let cuv = in.uv * p.framing.x + (p.framing.yz - p.framing.x * 0.5);
    let whole = p.bg_mode < 2u;
    let buv = select(in.uv, cuv, whole); // 배경·조명 좌표
    let color = textureSampleLevel(frame, samp, cuv, 0.0).rgb;
    let raw = textureSampleLevel(mask_hi, samp, cuv, 0.0).r;

    var bg: vec3f;
    if (p.bg_mode == 0u) {
        bg = color;
    } else if (p.bg_mode == 1u) {
        let blurred = textureSampleLevel(bg_blur, samp, buv, 0.0).rgb;
        bg = mix(color, blurred, clamp(p.blur_strength, 0.0, 1.0));
    } else if (p.bg_mode == 2u) {
        bg = p.bg_color.rgb;
    } else {
        // 이미지 배경 좌표 — v-ai image 스테이지 정점 셰이더 등가:
        // 중심 좌표계에서 aspect 보정 → mirror/rotation 행렬 → cover scale+offset
        var c = in.uv * 2.0 - 1.0;
        c *= p.bg_aspect.xy;
        c = vec2f(p.bg_mat.x * c.x + p.bg_mat.z * c.y, p.bg_mat.y * c.x + p.bg_mat.w * c.y);
        c /= p.bg_aspect.xy;
        bg = blur_bg_image((c * 0.5 + 0.5) * p.bg_scale + p.bg_offset);
    }

    // 매팅 합성 (RVM): fgr = 모델이 복원한 순수 전경색 — 단 **모델 해상도**
    // (256×144)라 전면 사용하면 인물이 통째로 소프트해진다 (1280 입력에서 5배
    // 업스케일 — 실제로 눈에 띄어 수리). 매팅의 실익(머리카락에 옛 배경이 배는
    // 오염 제거)은 경계 불확실 구간에만 있으므로: 알파가 확실한 내부는 카메라
    // 원본(풀해상), 경계만 fgr. 원리적 해결(풀해상 fgr)은 mnv4-RVM 교체 때.
    var fg = color;
    if (p.use_fgr > 0.5) {
        let f = textureSampleLevel(fgr_tex, samp, cuv, 0.0).rgb;
        let interior = smoothstep(p.cov.y, 1.0, raw);
        fg = mix(f, color, interior);
    }
    // light wrapping — 배경 빛이 인물 윤곽에 감기는 효과 (screen 블렌드).
    // v-ai에선 단색(#hex)도 image 스테이지를 타므로 색/이미지 배경 공통.
    if (p.bg_mode >= 2u) {
        let lwm = 1.0 - max(0.0, raw - p.cov.y) / (1.0 - p.cov.y);
        let lw = p.light_wrap * lwm * bg;
        fg = 1.0 - (1.0 - fg) * (1.0 - lw);
    }

    let pm = smoothstep(p.cov.x, p.cov.y, raw);
    // 스필 억제/엣지 다크닝은 매팅 모드에서도 살아있는 조정 파라미터다 —
    // fgr가 오염을 원리적으로 줄이지만 잔여 스필(실제 조명 반사·fgr 추정 오차)은
    // 남는다. 끄고 싶으면 파라미터를 0으로 (코드로 막지 않는다).
    // 단 passthrough(원본 배경)는 v-ai에 스필/엣지 자체가 없다 — 모드로 끈다.
    let edge = clamp(1.0 - abs(pm * 2.0 - 1.0), 0.0, 1.0);
    if (p.bg_mode != 0u) {
        fg = mix(fg, vec3f(gray(fg)), edge * clamp(p.spill, 0.0, 1.0));
    }
    bg = mix(bg, vec3f(gray(bg)), clamp(p.grayscale, 0.0, 1.0)) * p.brightness;

    var fin = mix(bg, fg, pm);
    if (p.bg_mode != 0u) {
        fin *= 1.0 - edge * clamp(p.edge_dark, 0.0, 1.0) * 0.06;
    }
    fin = apply_relight(fin, pm, buv);
    return vec4f(fin, 1.0);
}
