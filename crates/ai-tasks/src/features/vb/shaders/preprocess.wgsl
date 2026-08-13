// GPU 전처리 — 프레임 텍스처를 모델 입력(NHWC-C4 스토리지)으로 직결.
// CPU 픽셀 왕복 0: 리사이즈(bilinear HW 샘플러) + [0,1] 정규화가 한 패스.
// c=3(RGB) 전제 → cg=1, 레인 [r,g,b,0].

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
struct P { w: u32, h: u32, cg: u32, _pad: u32 }
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(p.w), f32(p.h));
    let rgb = textureSampleLevel(frame, samp, uv, 0.0).rgb;
    out[(gid.y * p.w + gid.x) * p.cg] = vec4f(rgb, 0.0);
}
