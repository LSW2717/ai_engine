//! 터치업/메이크업 게이트 — **오버레이가 합성 출력에 실제로 반영되는지**를
//! 픽셀로 검증한다 ("크래시 없음" 스모크는 uniform 0 오배선을 못 잡는다 —
//! vb_pipeline 마스크 전멸 사고의 교훈).
//!
//! 합성 랜드마크(원형 오벌+립+눈)를 실프레임 위에 놓고 process_gpu_mask
//! (ema=false, 전체 인물 마스크)로 렌더: ① fx on/off가 립 영역에서 달라야 하고
//! ② 얼굴 밖 원거리 픽셀은 완전 동일해야 하며 ③ update_face_fx(None)이
//! 베이스라인을 비트 단위로 복원해야 한다. 모델(.sw) 없으면 스킵.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::vb::VideoPipeline;
use ai_tasks::GpuSession;

const SW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

/// 얼굴형 합성 랜드마크 478개 — 정규화 좌표. 오벌/립/눈/눈썹/볼만 유의미하게
/// 배치 (rasterize가 쓰는 인덱스), 나머지는 얼굴 중심 (bbox 오염 방지).
fn synthetic_points(fw: f32, fh: f32) -> Vec<[f32; 3]> {
    let (cx, cy) = (128.0f32, 72.0f32);
    let mut pts = vec![[cx / fw, cy / fh, 0.0]; 478];
    let mut set = |i: usize, x: f32, y: f32| pts[i] = [x / fw, y / fh, 0.0];
    const FACE_OVAL: [usize; 36] = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ];
    const LIPS_OUTER: [usize; 20] = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    ];
    const LIPS_INNER: [usize; 20] = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ];
    const LASH_RIGHT: [usize; 9] = [33, 246, 161, 160, 159, 158, 157, 173, 133];
    const LASH_LEFT: [usize; 9] = [263, 466, 388, 387, 386, 385, 384, 398, 362];
    for (k, &i) in FACE_OVAL.iter().enumerate() {
        let a = k as f32 / 36.0 * std::f32::consts::TAU;
        set(i, cx + 40.0 * a.cos(), cy + 48.0 * a.sin());
    }
    set(234, cx - 40.0, cy);
    set(454, cx + 40.0, cy);
    for (k, &i) in LIPS_OUTER.iter().enumerate() {
        let a = k as f32 / 20.0 * std::f32::consts::TAU;
        set(i, cx + 14.0 * a.cos(), cy + 24.0 + 6.0 * a.sin());
    }
    for (k, &i) in LIPS_INNER.iter().enumerate() {
        let a = k as f32 / 20.0 * std::f32::consts::TAU;
        set(i, cx + 8.0 * a.cos(), cy + 24.0 + 3.0 * a.sin());
    }
    for (k, &i) in LASH_RIGHT.iter().enumerate() {
        set(i, cx - 24.0 + k as f32 * 2.0, cy - 10.0);
    }
    for (k, &i) in LASH_LEFT.iter().enumerate() {
        set(i, cx + 8.0 + k as f32 * 2.0, cy - 10.0);
    }
    set(205, cx - 22.0, cy + 8.0);
    set(425, cx + 22.0, cy + 8.0);
    pts
}

fn read_target(ctx: &GpuContext, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("face-fx-staging"),
        size: (w * 4 * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4), // 256×4=1024 — 256 정렬 충족
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    pollster::block_on(ai_gpu::readback::read_buffers(ctx, &[&staging]))
        .expect("리드백")
        .remove(0)
}

#[test]
fn face_fx_appears_and_clears() {
    let (Ok(sw), Ok(rgb)) = (std::fs::read(SW), std::fs::read(FRAME)) else {
        eprintln!("skip: rvm_256x144.sw 또는 프레임 없음");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let seg = pollster::block_on(GpuSession::load(&ctx, &sw)).expect("RVM 로드");
    let mut pipe = VideoPipeline::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("face-fx-target"),
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
    let ones = vec![1.0f32; (w * h) as usize];
    let mut render = |pipe: &mut VideoPipeline| {
        pipe.with_frame_texture(&ctx, &seg, w, h, |tex| {
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
        // 전체 인물 마스크(α=1)·ema off — 결정적, fg가 전 화면 (fx 관찰 최적)
        pipe.process_gpu_mask(&ctx, &seg, &ones, 1, w, h, false, w, h, &tview).unwrap();
        read_target(&ctx, &target, w, h)
    };

    let base = render(&mut pipe);

    // fx on — 립 틴트가 확실히 보이는 룩
    pipe.apply_json(
        r##"{"touchUp":{"enabled":true,"strength":1.0},
             "makeup":{"enabled":true,"intensity":1.0,
               "lip":{"color":"#d98f95","alpha":0.45},
               "blush":{"color":"#edaab2","alpha":0.18,"size":0.23},
               "shadow":{"color":"#b98d84","alpha":0.16}}}"##,
    )
    .unwrap();
    let pts = synthetic_points(w as f32, h as f32);
    pipe.update_face_fx(&ctx, Some(&pts));
    let fx = render(&mut pipe);

    let diff_at = |a: &[u8], b: &[u8], x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        (0..3).map(|c| (a[i + c] as i32 - b[i + c] as i32).unsigned_abs()).max().unwrap()
    };
    // ① 립 밴드 윈도 최대 diff — 립은 outer−inner **밴드**만 칠해진다 (even-odd,
    // 입 안은 구멍이라 정중앙은 무변화가 정상). 알파 0.45면 수십 레벨.
    let mut lip = 0;
    for y in 88..=104u32 {
        for x in 118..=138u32 {
            lip = lip.max(diff_at(&base, &fx, x, y));
        }
    }
    assert!(lip > 8, "립 틴트가 안 보인다 (창 max diff {lip})");
    // ② 얼굴 오벌 안 피부(이마 부근)에서 터치업 리프트/블러 diff
    let skin = diff_at(&base, &fx, 128, 40);
    assert!(skin > 0, "터치업이 안 보인다");
    // ③ 얼굴 밖 원거리(우하단)는 무변화
    let far = diff_at(&base, &fx, 250, 140);
    assert_eq!(far, 0, "얼굴 밖 픽셀이 변했다 ({far})");

    // ④ 소실(None) → 베이스라인 비트 복원
    pipe.update_face_fx(&ctx, None);
    let off = render(&mut pipe);
    assert_eq!(base, off, "fx 해제 후 베이스라인 미복원");
    println!("face-fx lip diff {lip} / skin diff {skin} / far 0 / clear 복원 OK");
}
