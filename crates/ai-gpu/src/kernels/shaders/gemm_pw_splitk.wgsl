// pointwise 1×1 conv — split-K 변형 (M 중간 + K 큼: 심층 저해상 병목의 기아 해법)
// Tiled는 M=144급에서 워크그룹 20~25개로 GPU를 굶긴다 (실측 960->160 = 120 GFLOP/s).
// 워크그룹(256스레드) = (PXT픽셀 × NGT출력그룹) 타일 × SPLIT개 k-청크. 스레드가
// 자기 청크의 부분합을 만들고 공유메모리로 청크 축 리덕션 — 단일 디스패치.
// px×ng 2D 타일이 핵심: 1px×CPW ng(픽셀당 워크그룹)로 펴면 워크그룹마다 가중치
// 슬라이스 전체를 L2에서 재독한다(960→160 기준 ~88MB/프레임). PXT=8이 가중치
// 라인을 8픽셀이 공유해 L2 트래픽을 ~5배 줄인다 (입력은 NGT=4 스레드가 공유).
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS(M/KG/NG/CPW/SPLIT/KC/PXT/NGT/NGTC), EPILOGUE

//@TYPES

@group(0) @binding(1) var<storage, read> IN: array<sv4>;
@group(0) @binding(2) var<storage, read> W: array<wv4>;
@group(0) @binding(3) var<storage, read> BIAS: array<sv4>;
//@RES_BINDING
//@OUT_BINDING
//@CONSTS

var<workgroup> sh: array<vec4f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cl = t % CPW; // 타일 내 셀
    let chunk = t / CPW; // k-청크
    let pxt = wg.x / NGTC;
    let ngt = wg.x - pxt * NGTC;
    let lpx = cl / NGT;
    let px = pxt * PXT + lpx;
    let ng = ngt * NGT + (cl - lpx * NGT);
    let live = px < M && ng < NG;
    var acc = vec4f(0.0);
    if (live) {
        let k1 = min(chunk * KC + KC, KG);
        for (var kg = chunk * KC; kg < k1; kg = kg + 1u) {
            let a = vec4f(IN[px * KG + kg]);
            let wb = (kg * NG + ng) * 4u;
            acc = acc
                + vec4f(
                    dot(vec4f(W[wb]), a),
                    dot(vec4f(W[wb + 1u]), a),
                    dot(vec4f(W[wb + 2u]), a),
                    dot(vec4f(W[wb + 3u]), a),
                );
        }
    }
    sh[t] = acc;
    workgroupBarrier();
    if (chunk == 0u && live) {
        var acc2 = sh[t];
        for (var s = 1u; s < SPLIT; s = s + 1u) {
            acc2 = acc2 + sh[t + s * CPW];
        }
        let out_idx = px * NG + ng;
        //@EPILOGUE
        OUT[out_idx] = sv4(acc2);
    }
}
