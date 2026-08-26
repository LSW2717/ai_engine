// mtoon.wgsl — VRM 아바타 렌더 셰이더 (스키닝 + 모프 + MToon 툰 셰이딩 + 아웃라인).
//
// MToon(VRM 0.x) 조명식의 v1 부분집합:
//   lit = mix(shade, base, ramp(N·L))  — ramp 는 _ShadeShift/_ShadeToony 로
//   경계 위치·폭을 정하는 smoothstep (Unity MToon 의 shading 커브 등가).
//   림/매트캡/이미시브는 스코프 밖 (보유 에셋 6종 미사용 확인).
// 출력은 items3d 와 같은 규약 — 라이트/텍스처는 리니어로 계산하고 srgb_oetf 로
// 인코드해 비-sRGB 타깃에 쓴다.
//
// 스키닝: JOINTS_0 u16×4 / WEIGHTS_0 f32×4, 팔레트는 uniform array<mat4>
// (MAX_JOINTS 96 — 보유 에셋 최대 65). 모프는 정점당 vec4 4개
// [blink_dpos, blink_dnrm, aa_dpos, aa_dnrm] 의 read-only storage.

const MAX_JOINTS: u32 = 96u;

struct Camera {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    // xyz = 키라이트 방향(월드, 정규화), w = 노출
    light_dir: vec4<f32>,
    // rgb = 라이트 색, a = 앰비언트 플로어
    light_color: vec4<f32>,
}

struct Material {
    base_color: vec4<f32>,
    shade_color: vec4<f32>,
    outline_color: vec4<f32>,
    // x = outline 폭(월드 m), y = cutoff, z = shade_shift, w = shade_toony
    params: vec4<f32>,
    // x = blend_mode(0 opaque/1 cutout/2 blend), yzw 예약
    mode: vec4<f32>,
}

struct Joints {
    m: array<mat4x4<f32>, MAX_JOINTS>,
}

// x = blink, y = aa(입), zw 예약
struct MorphWeights {
    w: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> joints: Joints;
@group(0) @binding(2) var<uniform> morph_w: MorphWeights;

@group(1) @binding(0) var<uniform> mat_u: Material;
@group(1) @binding(1) var main_tex: texture_2d<f32>;
@group(1) @binding(2) var tex_samp: sampler;
@group(1) @binding(3) var shade_tex: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> morph: array<vec4<f32>>;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

fn skin_matrix(j: vec4<u32>, w: vec4<f32>) -> mat4x4<f32> {
    return joints.m[j.x] * w.x + joints.m[j.y] * w.y
        + joints.m[j.z] * w.z + joints.m[j.w] * w.w;
}

fn morphed_pos(in: VsIn) -> vec3<f32> {
    let base = in.vi * 4u;
    return in.pos + morph_w.w.x * morph[base].xyz + morph_w.w.y * morph[base + 2u].xyz;
}

fn morphed_normal(in: VsIn) -> vec3<f32> {
    let base = in.vi * 4u;
    return in.normal + morph_w.w.x * morph[base + 1u].xyz + morph_w.w.y * morph[base + 3u].xyz;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let skin = skin_matrix(in.joints, in.weights);
    let wp = cam.model * (skin * vec4<f32>(morphed_pos(in), 1.0));
    // 스킨/모델의 회전만 노멀에 적용 (균등 스케일 전제 — VRoid/OSA 에셋 공통)
    let n4 = cam.model * (skin * vec4<f32>(morphed_normal(in), 0.0));
    var out: VsOut;
    out.pos = cam.view_proj * wp;
    out.normal = normalize(n4.xyz);
    out.uv = in.uv;
    return out;
}

// 아웃라인: 스킨 후 월드 노멀 방향으로 밀어낸 뒤 앞면 컬링(인버티드 헐).
@vertex
fn vs_outline(in: VsIn) -> VsOut {
    let skin = skin_matrix(in.joints, in.weights);
    let wp0 = cam.model * (skin * vec4<f32>(morphed_pos(in), 1.0));
    let n4 = cam.model * (skin * vec4<f32>(morphed_normal(in), 0.0));
    let n = normalize(n4.xyz);
    let wp = vec4<f32>(wp0.xyz + n * mat_u.params.x, wp0.w);
    var out: VsOut;
    out.pos = cam.view_proj * wp;
    out.normal = n;
    out.uv = in.uv;
    return out;
}

// three sRGBTransferOETF — items3d.wgsl 과 동일 (pow(1/2.2) 근사 아님).
fn srgb_oetf(v: vec3<f32>) -> vec3<f32> {
    let hi = pow(v, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
    let lo = v * 12.92;
    return select(hi, lo, v <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(main_tex, tex_samp, in.uv);
    let base = mat_u.base_color * tex;
    let shade = mat_u.shade_color * textureSample(shade_tex, tex_samp, in.uv);
    if mat_u.mode.x == 1.0 && base.a < mat_u.params.y {
        discard;
    }
    let n = normalize(in.normal);
    let ndl = dot(n, cam.light_dir.xyz);
    // MToon shading: shift 가 경계 위치, (1-toony) 가 경계 폭
    let shift = mat_u.params.z;
    let width = max(1.0 - mat_u.params.w, 1e-3);
    let ramp = smoothstep(shift, shift + width, ndl);
    let ambient = cam.light_color.a;
    let lit = mix(shade.rgb, base.rgb, ramp);
    let color = lit * cam.light_color.rgb * cam.light_dir.w
        * (ambient + (1.0 - ambient) * max(ramp, 0.55));
    return vec4<f32>(srgb_oetf(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0))), base.a);
}

@fragment
fn fs_outline(in: VsOut) -> @location(0) vec4<f32> {
    let c = mat_u.outline_color.rgb;
    return vec4<f32>(srgb_oetf(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
