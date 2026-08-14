// GPU 레터박스 — 프레임 텍스처 → 디텍터 입력(NHWC-C4 스토리지) 직결.
// `letterbox_u8_rgb`(detect/letterbox.rs)와 같은 수학: keep_aspect 중앙 정렬,
// 픽셀 중심 bilinear(수동 — HW 샘플러의 8비트 가중치 양자화를 피해 CPU와 f32
// 동일 경로), 콘텐츠 밖은 lo(검정 패딩=BORDER_ZERO 등가), [lo,hi] 정규화.
// c=3(RGB) 전제 → 픽셀당 vec4 하나, 레인 [r,g,b,0].

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;
struct P {
    dst_w: u32,
    dst_h: u32,
    cg: u32,
    src_w: u32,
    src_h: u32,
    _p0: u32,
    lo: f32,
    hi: f32,
}
@group(0) @binding(2) var<uniform> p: P;

fn texel(x: i32, y: i32) -> vec3f {
    return textureLoad(frame, vec2i(x, y), 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) { return; }
    let sw = f32(p.src_w);
    let sh = f32(p.src_h);
    let dw = f32(p.dst_w);
    let dh = f32(p.dst_h);
    let scale = min(dw / sw, dh / sh);
    let cw = sw * scale;
    let ch = sh * scale;
    let ox = (dw - cw) * 0.5;
    let oy = (dh - ch) * 0.5;
    let fx = f32(gid.x) + 0.5 - ox;
    let fy = f32(gid.y) + 0.5 - oy;
    var rgb = vec3f(0.0); // 콘텐츠 밖 → 0*(hi-lo)+lo = lo
    if (!(fx < 0.0 || fy < 0.0 || fx > cw || fy > ch)) {
        let sx = clamp(fx / scale - 0.5, 0.0, sw - 1.0);
        let sy = clamp(fy / scale - 0.5, 0.0, sh - 1.0);
        let x0 = i32(sx);
        let y0 = i32(sy);
        let x1 = min(x0 + 1, i32(p.src_w) - 1);
        let y1 = min(y0 + 1, i32(p.src_h) - 1);
        let tx = sx - f32(x0);
        let ty = sy - f32(y0);
        rgb = texel(x0, y0) * (1.0 - tx) * (1.0 - ty)
            + texel(x1, y0) * tx * (1.0 - ty)
            + texel(x0, y1) * (1.0 - tx) * ty
            + texel(x1, y1) * tx * ty;
    }
    let val = rgb * (p.hi - p.lo) + vec3f(p.lo);
    out[(gid.y * p.dst_w + gid.x) * p.cg] = vec4f(val, 0.0);
}
