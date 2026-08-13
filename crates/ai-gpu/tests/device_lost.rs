//! device.lost 구독 검증 — destroy()가 lost_reason()에 나타나야 한다.
//! (실전 시나리오는 드라이버 리셋·GPU 프로세스 크래시지만, 테스트에서 유일하게
//! 결정적으로 유발 가능한 lost는 destroy다.)

use ai_gpu::GpuContext;

#[test]
fn destroy_sets_lost_reason() {
    let ctx = match GpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    assert!(ctx.lost_reason().is_none(), "초기 상태는 lost 아님");

    ctx.device.destroy();
    // 콜백은 poll(유지보수 틱)에서 해소된다 — Wait이 destroy 후 에러를 줄 수 있어 무시
    let _ = ctx.device.poll(wgpu::PollType::Poll);

    let reason = ctx.lost_reason();
    assert!(reason.is_some(), "destroy 후 lost_reason()이 Some이어야 함");
    println!("lost_reason: {}", reason.unwrap());
}
