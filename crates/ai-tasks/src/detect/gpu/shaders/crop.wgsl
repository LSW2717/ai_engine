// GPU 회전 ROI 크롭 — 프레임 텍스처 → 랜드마크 입력(NHWC-C4 스토리지) 직결.
// `crop_u8_rgb`(detect/roi.rs)와 같은 수학: OpenCV warpPerspective 규약(dst 정수
// 픽셀 코너 정합, +0.5 텍셀 센터 없음), replicate 경계(clamp), bilinear 수동
// (HW 샘플러의 8비트 가중치 양자화를 피해 CPU와 f32 동일 경로), [lo,hi] 정규화
// (랜드마크 규약은 [0,1]). c=3 전제 → 픽셀당 vec4 하나, 레인 [r,g,b,0].

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;
struct P {
    dst: u32,
    cg: u32,
    src_w: u32,
    src_h: u32,
    cx: f32,
    cy: f32,
    rw: f32,
    rh: f32,
    sinr: f32,
    cosr: f32,
    lo: f32,
    hi: f32,
}
@group(0) @binding(2) var<uniform> p: P;

fn texel(x: i32, y: i32) -> vec3f {
    return textureLoad(frame, vec2i(x, y), 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p.dst || gid.y >= p.dst) { return; }
    let dx = f32(gid.x) / f32(p.dst) - 0.5;
    let dy = f32(gid.y) / f32(p.dst) - 0.5;
    let sx = clamp(
        p.cx + p.rw * dx * p.cosr - p.rh * dy * p.sinr,
        0.0,
        f32(p.src_w) - 1.0,
    );
    let sy = clamp(
        p.cy + p.rw * dx * p.sinr + p.rh * dy * p.cosr,
        0.0,
        f32(p.src_h) - 1.0,
    );
    let x0 = i32(sx);
    let y0 = i32(sy);
    let x1 = min(x0 + 1, i32(p.src_w) - 1);
    let y1 = min(y0 + 1, i32(p.src_h) - 1);
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let rgb = texel(x0, y0) * (1.0 - tx) * (1.0 - ty)
        + texel(x1, y0) * tx * (1.0 - ty)
        + texel(x0, y1) * (1.0 - tx) * ty
        + texel(x1, y1) * tx * ty;
    let val = rgb * (p.hi - p.lo) + vec3f(p.lo);
    out[(gid.y * p.dst + gid.x) * p.cg] = vec4f(val, 0.0);
}
