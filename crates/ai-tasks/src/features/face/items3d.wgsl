// items3d.wgsl — 3D 아이템(모자/안경/수염) 렌더 셰이더.
//
// 웹 face-3d.ts 는 three MeshStandardMaterial + RoomEnvironment IBL + ACESFilmic
// 이다. 이 셰이더는 그 조명식을 "근사"가 아니라 three 셰이더 청크 그대로 이식한다:
//   BRDF_GGX / F_Schlick / V_GGX_SmithCorrelated / D_GGX  (bsdfs)
//   DFGApprox + computeMultiscattering                    (envmap_physical)
//   RE_IndirectDiffuse/Specular_Physical 조립 순서         (lights_fragment_end)
//   ACESFilmicToneMapping + RRTAndODTFit                  (tonemapping)
//   sRGBTransferOETF                                      (colorspace)
//
// 조명 구성 — 웹과 1:1:
//   HemisphereLight(하늘 흰색 / 바닥 0x39404d, 0.55)  → 간접 확산
//   DirectionalLight(1.1, 씬 틴트 절반 반영)          → 직접 확산+스펙큘러
//   scene.environment = PMREM(RoomEnvironment)        → 간접 확산+반사
//
// three STANDARD 경로에서 irradiance(헤미)와 iblIrradiance(환경)는 분리되어
// 흐른다. 환경 확산은 RE_IndirectSpecular 안에서 에너지 보상 계수
// (1 - totalScatteringDielectric)를 곱해 더해진다 — 아래 조립도 같은 순서다.

const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 0.3183098861837907;
const EPSILON: f32 = 1e-6;

// 웹 three 광원 intensity 와 동일한 값 — 나란히 보고 맞출 것.
const HEMI_INTENSITY: f32 = 0.55;
const KEY_INTENSITY: f32 = 1.1;
// three MeshStandardMaterial.envMapIntensity 기본값. 환경 전체 밝기 노브.
const ENV_INTENSITY: f32 = 1.0;
// 환경 큐브맵 밉 수 — roughness→LOD 매핑용 (scripts/bake_room_env.py 와 동일해야 함).
const ENV_CUBE_MIPS: f32 = 7.0;
// three 투과 패스 배경 — WebGLRenderer.renderTransmissionPass 는 clearAlpha<1 이면
// setClearColor(0xffffff, 0.5) 로 클리어한다. face-3d 는 alpha:true(clearAlpha=0)라
// 항상 이 경로다. 우리 씬에는 별도 배경 타깃이 없으므로 그 상수를 그대로 쓴다.
const TRANSMISSION_BACKDROP: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const TRANSMISSION_BACKDROP_ALPHA: f32 = 0.5;

// RoomEnvironment 확산 조사도 SH9 (E/PI) — scripts/bake_room_env.py 생성.
// three getIBLIrradiance 의 ×PI 와 BRDF_Lambert 의 ×1/PI 가 상쇄된 값이라
// 그대로 envColor 자리에 넣는다.
const ENV_SH: array<vec3<f32>, 9> = array<vec3<f32>, 9>(
    vec3<f32>(3.791548, 3.791548, 3.791548),
    vec3<f32>(1.173777, 1.173777, 1.173777),
    vec3<f32>(1.091222, 1.091222, 1.091222),
    vec3<f32>(0.185307, 0.185307, 0.185307),
    vec3<f32>(0.054058, 0.054058, 0.054058),
    vec3<f32>(0.271769, 0.271769, 0.271769),
    vec3<f32>(0.304001, 0.304001, 0.304001),
    vec3<f32>(-0.086852, -0.086852, -0.086852),
    vec3<f32>(-0.157005, -0.157005, -0.157005),
);

struct Camera {
    view_proj: mat4x4<f32>,
    // 키라이트 방향(단위, 월드=카메라 공간) + 노출
    key_dir: vec4<f32>,   // xyz=dir, w=exposure
    key_color: vec4<f32>, // rgb=키 색(씬 틴트 절반 반영), w 미사용
}
@group(0) @binding(0) var<uniform> cam: Camera;
// RoomEnvironment GGX 프리필터 큐브맵 (RGBA16F, 64^2 x 7밉).
@group(0) @binding(1) var env_tex: texture_cube<f32>;
@group(0) @binding(2) var env_samp: sampler;

struct MeshParams {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    // x=metallic, y=roughness, z=has_texture(0/1), w=mode
    // mode: 0=일반 메시, 1=챙 그림자 스프라이트(절차 방사 그라데이션)
    factors: vec4<f32>,
    // x=has_mr_tex, y=has_normal_tex, z=normal_scale, w=double_sided(0/1)
    factors2: vec4<f32>,
    // x=transmission(KHR_materials_transmission), yzw=미사용
    factors3: vec4<f32>,
}
@group(1) @binding(0) var<uniform> mesh: MeshParams;
@group(1) @binding(1) var base_tex: texture_2d<f32>;
@group(1) @binding(2) var base_samp: sampler;
// glTF metallicRoughnessTexture — G=roughness, B=metallic. 선형 데이터.
@group(1) @binding(3) var mr_tex: texture_2d<f32>;
// 탄젠트 공간 노멀맵. 선형 데이터.
@group(1) @binding(4) var normal_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    let wp = mesh.model * vec4<f32>(position, 1.0);
    out.pos = cam.view_proj * wp;
    // 유사변환(균등 스케일) 전제 — 3x3 만 적용, fs 에서 정규화.
    let n = (mesh.model * vec4<f32>(normal, 0.0)).xyz;
    out.normal = n;
    out.uv = uv;
    out.world_pos = wp.xyz;
    return out;
}

// ═══════════════ three bsdfs 이식 ═══════════════

fn pow2(x: f32) -> f32 { return x * x; }
// saturate 는 WGSL 빌트인과 이름이 겹칠 수 있어 sat/sat3 로 둔다.
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)); }

// three F_Schlick — exp2 기반 Schlick 근사(정확한 pow5 아님, three 와 동일).
fn f_schlick(f0: vec3<f32>, f90: f32, dot_vh: f32) -> vec3<f32> {
    let fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
    return f0 * (1.0 - fresnel) + vec3<f32>(f90 * fresnel);
}

fn v_ggx_smith_correlated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
    let a2 = pow2(alpha);
    let gv = dot_nl * sqrt(a2 + (1.0 - a2) * pow2(dot_nv));
    let gl = dot_nv * sqrt(a2 + (1.0 - a2) * pow2(dot_nl));
    return 0.5 / max(gv + gl, EPSILON);
}

fn d_ggx(alpha: f32, dot_nh: f32) -> f32 {
    let a2 = pow2(alpha);
    let denom = pow2(dot_nh) * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / pow2(denom);
}

fn brdf_ggx(
    light_dir: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    f0: vec3<f32>,
    f90: f32,
    roughness: f32,
) -> vec3<f32> {
    let alpha = pow2(roughness);
    let half_dir = normalize(light_dir + view_dir);
    let dot_nl = sat(dot(normal, light_dir));
    let dot_nv = sat(dot(normal, view_dir));
    let dot_nh = sat(dot(normal, half_dir));
    let dot_vh = sat(dot(view_dir, half_dir));
    let f = f_schlick(f0, f90, dot_vh);
    let v = v_ggx_smith_correlated(alpha, dot_nl, dot_nv);
    let d = d_ggx(alpha, dot_nh);
    return f * (v * d);
}

// SH9 확산 조사도 평가 (Ramamoorthi-Hanrahan). 상수항 하나였던 ENV_AMBIENT 를
// 대체 — 방향성이 생겨야 표면이 평면으로 안 보인다.
fn env_diffuse(n: vec3<f32>) -> vec3<f32> {
    let x = n.x; let y = n.y; let z = n.z;
    var e = ENV_SH[0] * 0.282095;
    e += ENV_SH[1] * (0.488603 * y);
    e += ENV_SH[2] * (0.488603 * z);
    e += ENV_SH[3] * (0.488603 * x);
    e += ENV_SH[4] * (1.092548 * x * y);
    e += ENV_SH[5] * (1.092548 * y * z);
    e += ENV_SH[6] * (0.315392 * (3.0 * z * z - 1.0));
    e += ENV_SH[7] * (1.092548 * x * z);
    e += ENV_SH[8] * (0.546274 * (x * x - y * y));
    return max(e, vec3<f32>(0.0));
}

// three getIBLRadiance — 반사벡터를 roughness^4 만큼 노멀 쪽으로 당기고
// roughness→LOD 로 프리필터 밉을 고른다.
fn env_radiance(view_dir: vec3<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    var r = reflect(-view_dir, n);
    let r4 = roughness * roughness * roughness * roughness;
    r = normalize(mix(r, n, r4));
    let lod = roughness * (ENV_CUBE_MIPS - 1.0);
    return textureSampleLevel(env_tex, env_samp, r, lod).rgb;
}

// 환경 BRDF — three 0.185 는 dfgLUT 텍스처를 쓰지만, 여기서는 동일 곡선의
// 해석 근사(Karis, three 0.17x 이전의 DFGApprox)를 쓴다. LUT 대비 오차 ~1%,
// 텍스처 바인딩 하나를 아낀다.
fn dfg_approx(dot_nv: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

struct Scatter {
    single: vec3<f32>,
    multi: vec3<f32>,
}

// three computeMultiscattering — Fdez-Agüera 다중산란 보상.
fn compute_multiscattering(fab: vec2<f32>, fr: vec3<f32>, f90: f32) -> Scatter {
    let fss_ess = fr * fab.x + vec3<f32>(f90 * fab.y);
    let ess = fab.x + fab.y;
    let ems = 1.0 - ess;
    let favg = fr + (vec3<f32>(1.0) - fr) * 0.047619;
    let fms = fss_ess * favg / (vec3<f32>(1.0) - ems * favg);
    var s: Scatter;
    s.single = fss_ess;
    s.multi = fms * ems;
    return s;
}

// ═══════════════ three 톤매핑/색공간 이식 ═══════════════

fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// three ACESFilmicToneMapping — Stephen Hill fit + ACES 입출력 행렬.
// Narkowicz 근사와 달리 행렬 변환이 있고 exposure 를 0.6 으로 나눈다.
fn aces_filmic(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
    let aces_input = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let aces_output = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    var c = color_in * (exposure / 0.6);
    c = aces_input * c;
    c = rrt_and_odt_fit(c);
    c = aces_output * c;
    return sat3(c);
}

// three sRGBTransferOETF — pow(1/2.2) 근사가 아닌 정확한 sRGB 전달함수.
fn srgb_oetf(v: vec3<f32>) -> vec3<f32> {
    let hi = pow(v, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
    let lo = v * 12.92;
    return select(hi, lo, v <= vec3<f32>(0.0031308));
}

// three getTangentFrame (normal_pars_fragment) — TANGENT 어트리뷰트가 없는
// GLB 라 화면 미분으로 탄젠트 프레임을 만든다(Mikkelsen).
fn tangent_frame(eye_pos: vec3<f32>, surf_norm: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let q0 = dpdx(eye_pos);
    let q1 = dpdy(eye_pos);
    let st0 = dpdx(uv);
    let st1 = dpdy(uv);
    let q1perp = cross(q1, surf_norm);
    let q0perp = cross(surf_norm, q0);
    let t = q1perp * st0.x + q0perp * st1.x;
    let b = q1perp * st0.y + q0perp * st1.y;
    let det = max(dot(t, t), dot(b, b));
    var scale = 0.0;
    if det > 0.0 {
        scale = inverseSqrt(det);
    }
    return mat3x3<f32>(t * scale, b * scale, surf_norm);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // 챙 그림자 스프라이트 — 방사 그라데이션(웹 shadowTex: 0→0.32, 0.6→0.18, 1→0).
    // 웹은 MeshBasicMaterial{toneMapped:false} — 톤매핑/조명 모두 통과시키지 않는다.
    if mesh.factors.w > 0.5 {
        let d = length(in.uv * 2.0 - 1.0);
        var a: f32;
        if d < 0.6 {
            a = mix(0.32, 0.18, d / 0.6);
        } else {
            a = mix(0.18, 0.0, sat((d - 0.6) / 0.4));
        }
        // 검정 프리멀티 — rgb 0.
        return vec4<f32>(0.0, 0.0, 0.0, a);
    }

    var base = mesh.base_color;
    if mesh.factors.z > 0.5 {
        base = base * textureSample(base_tex, base_samp, in.uv);
    }
    if base.a < 0.004 {
        discard;
    }

    // 단면 재질(doubleSided=false) — three 는 side=FrontSide 라 뒷면을 컬링한다.
    // 파이프라인은 cull_mode=None 이므로 여기서 버려 동작을 맞춘다.
    if mesh.factors2.w < 0.5 && !front_facing {
        discard;
    }

    // 양면 재질 — three DoubleSide 는 gl_FrontFacing 으로 뒤집는다(normal_fragment_begin).
    // 셰이딩 노멀과 시선의 내적 부호로 판정하면 스무스 노멀의 실루엣 근처에서
    // 앞면인데도 뒤집혀 명암이 튄다(고개를 숙일 때 수염 색이 바뀌던 원인).
    var n = normalize(in.normal);
    if mesh.factors2.w > 0.5 && !front_facing {
        n = -n;
    }
    // 노멀맵 적용 전 지오메트리 노멀 — geometryRoughness 계산에 쓴다.
    let non_perturbed_normal = n;

    // 탄젠트 공간 노멀맵 — three normal_fragment_maps 와 동일.
    if mesh.factors2.y > 0.5 {
        var map_n = textureSample(normal_tex, base_samp, in.uv).xyz * 2.0 - 1.0;
        map_n = vec3<f32>(map_n.xy * mesh.factors2.z, map_n.z);
        n = normalize(tangent_frame(in.world_pos, n, in.uv) * map_n);
    }

    let view_dir = normalize(-in.world_pos);
    // glTF: metallicRoughnessTexture 의 G=roughness, B=metallic 이 factor 와 곱해진다.
    // factor 키가 생략된 GLB 는 스펙상 1.0/1.0 이라, 텍스처를 안 읽으면 통째로
    // 거친 금속이 되어 반사가 사라진다(글래스1 렌즈).
    var metalness = mesh.factors.x;
    var roughness = mesh.factors.y;
    if mesh.factors2.x > 0.5 {
        let mr = textureSample(mr_tex, base_samp, in.uv);
        roughness = roughness * mr.g;
        metalness = metalness * mr.b;
    }
    metalness = sat(metalness);
    // three lights_physical_fragment — 스페큘러 안티에일리어싱.
    // 화면상 노멀 변화율을 러프니스에 더해, 곡률이 큰 곳/실루엣에서 거울 하이라이트가
    // 픽셀 단위로 반짝이는 걸 억제한다. 클램프 → 가산 → 상한 순서까지 three 와 동일.
    let dxy = max(abs(dpdx(non_perturbed_normal)), abs(dpdy(non_perturbed_normal)));
    let geometry_roughness = max(max(dxy.x, dxy.y), dxy.z);
    roughness = max(roughness, 0.0525);
    roughness = min(roughness + geometry_roughness, 1.0);

    // three PhysicalMaterial 세팅
    let diffuse_contribution = base.rgb * (1.0 - metalness);
    let specular_color_dielectric = vec3<f32>(0.04);
    let specular_color_metallic = base.rgb;
    let specular_f90 = 1.0;

    let key_dir = cam.key_dir.xyz;
    let exposure = cam.key_dir.w;
    let key_color = cam.key_color.rgb;

    // ── 직접광: DirectionalLight ──
    // three getDirectionalLightInfo: directLight.color = color * intensity
    // RE_Direct_Physical: irradiance = dotNL * directLight.color
    let dot_nl = sat(dot(n, key_dir));
    let direct_irradiance = key_color * (KEY_INTENSITY * dot_nl);
    let f0 = mix(specular_color_dielectric, specular_color_metallic, metalness);
    let direct_diffuse = direct_irradiance * RECIPROCAL_PI * diffuse_contribution;
    let direct_specular = direct_irradiance
        * brdf_ggx(key_dir, view_dir, n, f0, specular_f90, roughness);

    // ── 간접 확산: HemisphereLight ──
    // three getHemisphereLightIrradiance: mix(ground, sky, 0.5*dotNL+0.5)
    // 하늘 흰색, 바닥 0x39404d = (0.224, 0.251, 0.302) (sRGB→linear 는 three 가
    // Color.setHex 시점에 처리 — 아래 값은 이미 linear).
    let hemi_t = 0.5 * n.y + 0.5;
    let hemi_irradiance =
        mix(vec3<f32>(0.0409, 0.0513, 0.0743), vec3<f32>(1.0), hemi_t) * HEMI_INTENSITY;
    let indirect_diffuse_hemi = hemi_irradiance * RECIPROCAL_PI * diffuse_contribution;

    // ── 간접 환경: IBL ──
    // getIBLIrradiance = PI * envColor * envMapIntensity  (envColor = SH9 확산)
    // getIBLRadiance   = envColor(roughness LOD) * envMapIntensity
    let ibl_irradiance = PI * env_diffuse(n) * ENV_INTENSITY;
    let radiance = env_radiance(view_dir, n, roughness) * ENV_INTENSITY;

    let dot_nv = sat(dot(n, view_dir));
    let fab = dfg_approx(dot_nv, roughness);
    let sc_dielectric = compute_multiscattering(fab, specular_color_dielectric, specular_f90);
    let sc_metallic = compute_multiscattering(fab, specular_color_metallic, specular_f90);

    let single_scattering = mix(sc_dielectric.single, sc_metallic.single, metalness);
    let multi_scattering = mix(sc_dielectric.multi, sc_metallic.multi, metalness);
    let total_scattering_dielectric = sc_dielectric.single + sc_dielectric.multi;

    let cosine_weighted_irradiance = ibl_irradiance * RECIPROCAL_PI;
    let indirect_specular = radiance * single_scattering
        + multi_scattering * cosine_weighted_irradiance;
    let indirect_diffuse_env = diffuse_contribution
        * (vec3<f32>(1.0) - total_scattering_dielectric)
        * cosine_weighted_irradiance;

    var total_diffuse = direct_diffuse + indirect_diffuse_hemi + indirect_diffuse_env;
    let total_specular = direct_specular + indirect_specular;
    var alpha = base.a;

    // three transmission_fragment + getIBLVolumeRefraction.
    // KHR_materials_volume 이 없어 thickness=0 / attenuationDistance=inf →
    // 굴절 광선 길이 0 → 자기 픽셀을 샘플하고 volumeAttenuation 은 1 이다.
    // 그 배경이 위 상수(흰색, 알파 0.5)라 전체를 해석적으로 계산할 수 있다.
    let transmission = mesh.factors3.x;
    if transmission > 0.0 {
        let transmittance = diffuse_contribution; // × volumeAttenuation(=1)
        let attenuated = transmittance * TRANSMISSION_BACKDROP;
        // EnvironmentBRDF( n, v, specularColorBlended, specularF90, roughness )
        let f_env = f0 * fab.x + vec3<f32>(specular_f90 * fab.y);
        let tf = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
        let transmitted_rgb = (vec3<f32>(1.0) - f_env) * attenuated;
        let transmitted_a = 1.0 - (1.0 - TRANSMISSION_BACKDROP_ALPHA) * tf;
        total_diffuse = mix(total_diffuse, transmitted_rgb, transmission);
        alpha = alpha * mix(1.0, transmitted_a, transmission);
    }

    var color = total_diffuse + total_specular;

    color = aces_filmic(color, exposure);
    // 디스플레이(sRGB) 인코딩 후 프리멀티 — pack 셰이더가 sRGB 공간 알파-오버,
    // 웹 canvas drawImage 규약과 동일.
    color = srgb_oetf(color);
    return vec4<f32>(color * alpha, alpha);
}

// 오클루더 — 깊이만 기록 (컬러 라이트 마스크 0 파이프라인에서 사용).
@fragment
fn fs_occluder(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
