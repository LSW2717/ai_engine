//! GPU 입력 전처리 커널 게이트 — 레터박스·회전 크롭 커널의 출력을 **CPU 기준
//! 구현**(letterbox_u8_rgb / crop_u8_rgb)과 원소 단위로 대조한다. 모델 불필요 —
//! 커널이 임의 스토리지 버퍼에 쓰므로 desc만 맞추면 된다.
//!
//! 수동 bilinear(textureLoad)라 CPU와 f32 동일 경로 — 허용 오차는 연산 순서
//! (fma 융합) 차이만 남긴 1e-4. HW 샘플러를 썼다면 8비트 가중치 양자화로
//! ~4e-3까지 벌어졌을 것이다.

use ai_core::{DType, TensorDesc};
use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::detect::letterbox::letterbox_u8_rgb;
use ai_tasks::detect::roi::{crop_u8_rgb, Roi};
use ai_tasks::GpuPre;

const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

/// 커널 출력 버퍼(NHWC-C4)를 읽어 논리 NHWC f32로 되돌린다
fn readback_logical(ctx: &GpuContext, buf: &wgpu::Buffer, desc: &TensorDesc) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex-input-staging"),
        size: desc.size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, desc.size_bytes());
    ctx.queue.submit([enc.finish()]);
    let bytes = pollster::block_on(ai_gpu::readback::read_buffers(ctx, &[&staging]))
        .expect("리드백")
        .remove(0);
    ai_core::pack::unpack_nhwc(&bytes, desc)
}

fn storage(ctx: &GpuContext, desc: &TensorDesc) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex-input-out"),
        size: desc.size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn kernels_match_cpu_reference() {
    let Ok(rgb) = std::fs::read(FRAME) else {
        eprintln!("skip: 프레임 픽스처 없음");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let mut pre = GpuPre::new(&ctx);
    pre.frame.upload_rgb(&ctx, &rgb, w, h);
    let (view, _, _) = pre.frame.view().unwrap();
    let view = view.clone(); // pre 재borrow와 무관하게 쓰도록 핸들 복제 (wgpu 리소스는 Arc)

    // ── 레터박스 (face 프리셋 규약: 128², [-1,1], 세로 패딩 발생) ──
    let desc = TensorDesc::new(128, 128, 3, DType::F32);
    let out = storage(&ctx, &desc);
    pre.letterbox_into(&ctx, &view, w, h, &out, &desc, -1.0, 1.0).unwrap();
    let gpu = readback_logical(&ctx, &out, &desc);
    let cpu = letterbox_u8_rgb(&rgb, w as usize, h as usize, 128, 128, -1.0, 1.0);
    let d = max_diff(&gpu, &cpu);
    println!("letterbox 128² [-1,1] max diff {d:.2e}");
    assert!(d < 1e-4, "레터박스 CPU/GPU 불일치: {d}");
    // 패딩 픽셀은 정확히 lo (0,0은 콘텐츠 밖 — 256×144→128²는 상하 패딩)
    assert_eq!(gpu[0], -1.0, "패딩이 lo가 아님");

    // ── 레터박스 (palm 규약: 192², [0,1]) ──
    let desc = TensorDesc::new(192, 192, 3, DType::F32);
    let out = storage(&ctx, &desc);
    pre.letterbox_into(&ctx, &view, w, h, &out, &desc, 0.0, 1.0).unwrap();
    let gpu = readback_logical(&ctx, &out, &desc);
    let cpu = letterbox_u8_rgb(&rgb, w as usize, h as usize, 192, 192, 0.0, 1.0);
    let d = max_diff(&gpu, &cpu);
    println!("letterbox 192² [0,1] max diff {d:.2e}");
    assert!(d < 1e-4, "레터박스(palm) CPU/GPU 불일치: {d}");

    // ── 회전 크롭 (프레임 밖으로 걸치는 ROI — replicate 경계까지 검증) ──
    let desc = TensorDesc::new(256, 256, 3, DType::F32);
    let out = storage(&ctx, &desc);
    let roi = Roi { cx: 200.0, cy: 40.0, w: 150.0, h: 150.0, rotation: 0.35 };
    pre.crop_into(&ctx, &view, w, h, &roi, &out, &desc, 0.0, 1.0).unwrap();
    let gpu = readback_logical(&ctx, &out, &desc);
    let cpu = crop_u8_rgb(&rgb, w as usize, h as usize, &roi, 256);
    let d = max_diff(&gpu, &cpu);
    println!("crop 256² rot 0.35 (경계 걸침) max diff {d:.2e}");
    assert!(d < 1e-4, "크롭 CPU/GPU 불일치: {d}");

    // ── 회전 크롭 (무회전, 프레임 안) ──
    let out = storage(&ctx, &desc);
    let roi = Roi { cx: 128.0, cy: 72.0, w: 100.0, h: 100.0, rotation: 0.0 };
    pre.crop_into(&ctx, &view, w, h, &roi, &out, &desc, 0.0, 1.0).unwrap();
    let gpu = readback_logical(&ctx, &out, &desc);
    let cpu = crop_u8_rgb(&rgb, w as usize, h as usize, &roi, 256);
    let d = max_diff(&gpu, &cpu);
    println!("crop 256² 무회전 max diff {d:.2e}");
    assert!(d < 1e-4, "크롭(무회전) CPU/GPU 불일치: {d}");
}
