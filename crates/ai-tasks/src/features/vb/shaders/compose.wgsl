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
    // 터치업/메이크업 (vcxrust_ai pack 이식, CPU가 128² 오버레이로 굽는다):
    // tu_map/mk_map = 프레임 정규화 좌표 → 오버레이 uv (xy=scale, zw=offset),
    // tu_par = x 알파(0.62×strength; 0=off), y 미사용, zw 블러 스트라이드(uv 단위)
    tu_map: vec4f,
    tu_par: vec4f,
    mk_map: vec4f,
}
@group(0) @binding(5) var<uniform> p: P;
// RVM 전경색 (매팅) — use_fgr=0이면 미사용 (프레임 뷰가 더미로 바인딩됨)
@group(0) @binding(6) var fgr_tex: texture_2d<f32>;
// 터치업 피부 마스크 (R8 128²) + 메이크업 오버레이 (RGBA8 128² ×2)
@group(0) @binding(7) var touchup_tex: texture_2d<f32>;
@group(0) @binding(8) var makeup_mul_tex: texture_2d<f32>;
@group(0) @binding(9) var makeup_over_tex: texture_2d<f32>;

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

// 128² 오버레이 bilinear (스트레이트 알파, clamp) — vcxrust sample_overlay 등가
fn sample_overlay(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    let mpos = uv * 128.0 - 0.5;
    let x0 = i32(floor(mpos.x));
    let y0 = i32(floor(mpos.y));
    let fx = mpos.x - f32(x0);
    let fy = mpos.y - f32(y0);
    let xc = clamp(x0, 0, 127);
    let yc = clamp(y0, 0, 127);
    let x1 = clamp(x0 + 1, 0, 127);
    let y1 = clamp(y0 + 1, 0, 127);
    let a = textureLoad(tex, vec2i(xc, yc), 0);
    let b = textureLoad(tex, vec2i(x1, yc), 0);
    let c = textureLoad(tex, vec2i(xc, y1), 0);
    let d = textureLoad(tex, vec2i(x1, y1), 0);
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

fn touchup_weight(cuv: vec2f) -> f32 {
    let uv = cuv * p.tu_map.xy + p.tu_map.zw;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 0.0; }
    return sample_overlay(touchup_tex, uv).r * p.tu_par.x;
}

// 3×3 가우스(1-2-1), 스트라이드 = blur_px(프레임 px → uv 선변환). vcxrust pack은
// 크롭 줌 시 blur_px/crop_scale 보정을 하지만 여기선 불필요 — 프레임 좌표계(cuv)
// 샘플이라 크롭 확대가 화면 블러 반경을 자동으로 같이 키운다.
fn touchup_blur(cuv: vec2f) -> vec3f {
    var acc = vec3f(0.0);
    var ws = 0.0;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let w = f32((2 - abs(dx)) * (2 - abs(dy)));
            let uv = cuv + vec2f(f32(dx), f32(dy)) * p.tu_par.zw;
            acc += textureSampleLevel(frame, samp, uv, 0.0).rgb * w;
            ws += w;
        }
    }
    return acc / max(ws, 1e-5);
}

// 메이크업 2단 합성 — 웹 drawMakeup 등가:
// multiply 레이어(섀도·립 본체) base×mix(1,color,α) → source-over 레이어 mix
fn apply_makeup(base: vec3f, cuv: vec2f) -> vec3f {
    let uv = cuv * p.mk_map.xy + p.mk_map.zw;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return base; }
    var out = base;
    let m = sample_overlay(makeup_mul_tex, uv);
    if (m.a > 0.003) { out = out * mix(vec3f(1.0), m.rgb, m.a); }
    let o = sample_overlay(makeup_over_tex, uv);
    if (o.a > 0.003) { out = mix(out, o.rgb, o.a); }
    return out;
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
    // 터치업 — 피부 마스크 가중 소프트 블러 + 1.02 밝기 리프트 (vcxrust pack
    // 이식 — 단 luma가 아니라 RGB에 적용: 웹 drawTouchUp이 RGB 블러다)
    if (p.tu_par.x > 0.0) {
        let tu = touchup_weight(cuv);
        if (tu > 0.003) {
            fg = mix(fg, min(touchup_blur(cuv) * 1.02, vec3f(1.0)), tu);
        }
    }
    // 메이크업 — 립/블러셔/아이섀도 컬러 오버레이 (얼굴 밖은 uv 가드로 무변화)
    if (p.mk_map.x > 0.0) {
        fg = apply_makeup(fg, cuv);
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
