// 인물 bbox 리덕션 — EMA 이후 마스크(r8, 모델 해상도)에서 v > 0.5 픽셀의
// min/max x·y + count. v-ai `_scanPersonBBox`(u8 > 127, CPU 순회) 등가를
// GPU에서: 워크그룹 공유 atomic으로 선축약 후 리더 스레드만 전역 atomic 5개 —
// 전역 경합은 워크그룹 수(256×144 → 144개)뿐. 리드백은 20B가 전부 (저사양 원칙).
//
// 호출자 규약: 디스패치 전 out을 {min=0xffffffff, max=0, count=0}으로 초기화.

@group(0) @binding(0) var mask: texture_2d<f32>;
struct BBox {
    min_x: atomic<u32>,
    max_x: atomic<u32>,
    min_y: atomic<u32>,
    max_y: atomic<u32>,
    count: atomic<u32>,
}
@group(0) @binding(1) var<storage, read_write> out: BBox;

var<workgroup> wmin_x: atomic<u32>;
var<workgroup> wmax_x: atomic<u32>;
var<workgroup> wmin_y: atomic<u32>;
var<workgroup> wmax_y: atomic<u32>;
var<workgroup> wcount: atomic<u32>;

@compute @workgroup_size(16, 16)
fn cs(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        atomicStore(&wmin_x, 0xffffffffu);
        atomicStore(&wmax_x, 0u);
        atomicStore(&wmin_y, 0xffffffffu);
        atomicStore(&wmax_y, 0u);
        atomicStore(&wcount, 0u);
    }
    workgroupBarrier();
    let dim = textureDimensions(mask);
    if (gid.x < dim.x && gid.y < dim.y) {
        // r8unorm: k/255 — v > 0.5 ⇔ k ≥ 128 ⇔ v-ai `u8 > 127`
        if (textureLoad(mask, vec2i(gid.xy), 0).r > 0.5) {
            atomicMin(&wmin_x, gid.x);
            atomicMax(&wmax_x, gid.x);
            atomicMin(&wmin_y, gid.y);
            atomicMax(&wmax_y, gid.y);
            atomicAdd(&wcount, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u && atomicLoad(&wcount) > 0u) {
        atomicMin(&out.min_x, atomicLoad(&wmin_x));
        atomicMax(&out.max_x, atomicLoad(&wmax_x));
        atomicMin(&out.min_y, atomicLoad(&wmin_y));
        atomicMax(&out.max_y, atomicLoad(&wmax_y));
        atomicAdd(&out.count, atomicLoad(&wcount));
    }
}
