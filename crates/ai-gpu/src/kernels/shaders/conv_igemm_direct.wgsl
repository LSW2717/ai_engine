// 일반 conv — direct 변형: 스레드 = 가로 인접 PB픽셀(4 또는 8) × 1출력그룹.
// WebGL2 엔진의 승리 공식 이식: (tap, kg)당 가중치 4페치를 4픽셀이 공유 → 가중치
// 트래픽 1/4. 탭 외부 언롤(오프셋 리터럴), kg 런타임 루프, 공유메모리 없음.
// 작은 cout(타일 낭비)과 작은 K(스템)에서 tiled보다 빠르다.
// 스레드 매핑: i = blk*NG + ng (인접 스레드 = 같은 블록의 인접 ng) — 4스레드가
// 같은 입력 주소를 공유(브로드캐스트)하고 블록 인접성이 L1 라인을 채운다.
// (실험 기록: wg=같은 ng의 64블록 재편성은 입력 주소가 64갈래로 흩어져 전면 2배
// 퇴행 — KG 소형 conv은 W가 L1 상주라 입력 지역성이 지배한다.)
// 슬롯: TYPES, RES_BINDING, OUT_BINDING, CONSTS, TAP_LOOPS, STORE4

//@TYPES

//@BINDINGS
//@CONSTS

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= NBLK * NG) {
        return;
    }
    let blk = i / NG;
    let ng = i - blk * NG;
    let oy = blk / BPR;
    let ox0 = (blk - oy * BPR) * PB;
    //@ACC_DECL
    //@TAP_LOOPS
    //@STORE4
}
