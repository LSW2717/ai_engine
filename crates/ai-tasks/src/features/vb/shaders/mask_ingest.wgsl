// 마스크 인제스트 — 모델 출력(NHWC-C4 스테이징 텍스처) → r8 마스크 + 시간 EMA.
// mode 0: 2채널 로짓 → sigmoid(person-bg) (= 2-class softmax, v-ai 규약)
// mode 1: 알파 직출 (RVM pha류 — lane 0)
// 시간 EMA(동적 α): diff > thr ? max_a : min_a — 마스크 떨림 억제의 핵심.
// history는 이전 프레임 출력(핑퐁) — 파이프라인이 매 프레임 교대 바인딩.

@group(0) @binding(0) var raw_tex: texture_2d<f32>;    // rgba32float (unfilterable)
@group(0) @binding(1) var history: texture_2d<f32>;    // r8unorm (이전 프레임)
struct P { mode: u32, _p0: u32, _p1: u32, _p2: u32, diff: f32, min_a: f32, max_a: f32, _p3: f32 }
@group(0) @binding(2) var<uniform> p: P;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let dim = textureDimensions(raw_tex);
    let xy = vec2i(in.uv * vec2f(dim));
    let v = textureLoad(raw_tex, xy, 0);
    var curr: f32;
    if (p.mode == 0u) {
        curr = 1.0 / (1.0 + exp(v.r - v.g));   // sigmoid(person - bg)
    } else {
        curr = v.r;
    }
    let prev = textureLoad(history, xy, 0).r;
    let a = select(p.min_a, p.max_a, abs(curr - prev) > p.diff);
    return vec4f(mix(prev, curr, a), 0.0, 0.0, 1.0);   // r8unorm 타깃 — r만 쓰임
}
