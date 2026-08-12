//! 비동기 리드백 — map_async를 전부 발행한 뒤 네이티브는 단일 poll(Wait),
//! 웹은 브라우저가 콜백을 해소한다(poll은 웹에서 no-op으로 문서화됨).
//! 같은 async 함수가 양 플랫폼에서 동작한다.

use crate::context::GpuContext;

/// 제출된 GPU 작업이 끝날 때까지 대기 (네이티브: poll(Wait), 웹: on_submitted_work_done)
pub async fn wait_idle(ctx: &GpuContext) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| format!("device.poll 실패: {e:?}"))?;
        Ok(())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (tx, rx) = futures_channel::oneshot::channel();
        ctx.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        rx.await.map_err(|_| "on_submitted_work_done 콜백 유실".to_string())
    }
}

/// MAP_READ 스테이징 버퍼들을 한꺼번에 읽는다.
pub async fn read_buffers(
    ctx: &GpuContext,
    buffers: &[&wgpu::Buffer],
) -> Result<Vec<Vec<u8>>, String> {
    let mut receivers = Vec::with_capacity(buffers.len());
    for buf in buffers {
        let (tx, rx) = futures_channel::oneshot::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        receivers.push(rx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    ctx.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|e| format!("device.poll 실패: {e:?}"))?;
    #[cfg(target_arch = "wasm32")]
    let _ = ctx; // 웹은 브라우저 이벤트 루프가 콜백을 해소

    let mut out = Vec::with_capacity(buffers.len());
    for (rx, buf) in receivers.into_iter().zip(buffers) {
        rx.await
            .map_err(|_| "map_async 콜백 유실".to_string())?
            .map_err(|e| format!("buffer map 실패: {e:?}"))?;
        let data = buf
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range 실패: {e:?}"))?
            .to_vec();
        buf.unmap();
        out.push(data);
    }
    Ok(out)
}
