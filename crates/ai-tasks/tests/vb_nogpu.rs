//! 세션 없는 VideoPipeline 게이트 — **GPU 세그 모델 없이** 외부 마스크 합성
//! 전체(ingest→upsample→refine→compose)가 도는지 검증 (B/C 티어의 근거:
//! RVM 로드·컴파일 없이 GPU 합성만 쓴다).
//!
//! 모델 파일이 전혀 필요 없다 — 프레임 픽스처만 있으면 된다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::vb::VideoPipeline;

const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn nogpu_mask_composites() {
    let Ok(rgb) = std::fs::read(FRAME) else {
        eprintln!("skip: 프레임 픽스처 없음");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let (mw, mh) = (128u32, 72u32); // CPU 모델 해상도라 가정 (프레임과 달라도 됨)
    let ctx = GpuContext::new_blocking().unwrap();
    let mut pipe = VideoPipeline::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
    pipe.apply_json(r##"{"background":"#00ff00"}"##).unwrap();

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vb-nogpu-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tview = target.create_view(&Default::default());

    let mut rgba = vec![255u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
    }
    // 절반 마스크: 왼쪽 절반 인물(α=1), 오른쪽 배경(α=0) — 합성 경계가 보이게
    let mut mask = vec![0f32; (mw * mh) as usize];
    for y in 0..mh {
        for x in 0..mw / 2 {
            mask[(y * mw + x) as usize] = 1.0;
        }
    }
    pipe.with_frame_texture_nogpu(&ctx, w, h, (mw, mh), |tex| {
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    })
    .unwrap();
    pipe.process_mask_nogpu(&ctx, &mask, 1, mw, mh, false, w, h, &tview).unwrap();

    // 리드백 (256×4=1024B/행 — 256 정렬 충족)
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vb-nogpu-staging"),
        size: (w * 4 * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let px = pollster::block_on(ai_gpu::readback::read_buffers(&ctx, &[&staging]))
        .expect("리드백")
        .remove(0);

    let at = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    // 오른쪽(배경): 초록. 왼쪽(인물): 프레임 픽셀 (초록 순색 아님)
    let bg = at(224, 72);
    assert!(bg[1] > 200 && bg[0] < 60 && bg[2] < 60, "배경이 초록 아님: {bg:?}");
    let fg = at(32, 72);
    let frame_px = {
        let i = ((72 * w + 32) * 3) as usize;
        [rgb[i], rgb[i + 1], rgb[i + 2]]
    };
    let d: i32 = fg
        .iter()
        .zip(&frame_px)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(d < 40, "인물 영역이 프레임 픽셀과 동떨어짐: {fg:?} vs {frame_px:?}");
    println!("nogpu 합성 OK — bg {bg:?} / fg {fg:?} (frame {frame_px:?})");
}
