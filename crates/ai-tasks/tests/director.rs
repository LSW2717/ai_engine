//! Director e2e — 단일 JSON 설정으로 세그+얼굴 fx+집중도가 실모델·실프레임에서
//! 한 프레임 루프로 도는지 (ai-ffi/ai-wasm이 접착할 오케스트레이터의 게이트).
//! 검증: ①비디오 경로 합성 출력 ②집중도 FOCUSED 도달(픽스처 pitch −21.8°가
//! 온타깃 박스 하한 −22° 안) ③웜 detach 후 재가동 ④passthrough/needs_render 판정.
//! 모델 없으면 스킵.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::Director;

const SEG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
const FACE_DET: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_detector.sw");
const FACE_LM: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_landmarks.sw");
const GAZE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/gaze.sw");
const GAZE_BS: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/mediapipe/face/face_blendshapes.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn director_e2e() {
    let (Ok(seg), Ok(det), Ok(lm), Ok(gaze), Ok(rgb)) = (
        std::fs::read(SEG),
        std::fs::read(FACE_DET),
        std::fs::read(FACE_LM),
        std::fs::read(GAZE),
        std::fs::read(FRAME),
    ) else {
        eprintln!("skip: 모델/픽스처 없음 (make convert-rvm-web convert-mediapipe)");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let mut d = Director::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
    d.set_model("seg", seg).unwrap();
    d.set_model("face_det", det).unwrap();
    d.set_model("face_lm", lm).unwrap();
    d.set_model("gaze", gaze).unwrap();
    if let Ok(bs) = std::fs::read(GAZE_BS) {
        d.set_model("gaze_bs", bs).unwrap();
    }

    assert!(d.passthrough(), "초기엔 완전 무가공");
    d.apply_json(
        r##"{"background":"#00a05a","touchUp":{"enabled":true,"strength":0.7},
             "focusDetection":{"enabled":true,"detectFps":10}}"##,
    )
    .unwrap();
    assert!(d.needs_render() && d.tasks_active());

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("director-target"),
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

    let mut run = |d: &mut Director, t: f64| {
        pollster::block_on(d.with_frame(&ctx, w, h, |tex| {
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
        }))
        .unwrap();
        pollster::block_on(d.frame(&ctx, w, h, Some(&tview), Some(&rgb), t)).unwrap();
    };

    for i in 0..6 {
        run(&mut d, i as f64 * 100.0);
    }
    let j = d.focus_json();
    assert!(
        j.contains("\"status\":\"FOCUSED\""),
        "픽스처는 온타깃(pitch −21.8 ≥ −22)이어야: {j}"
    );
    assert!(j.contains("\"yaw\":"), "필터 각도 포함: {j}");
    assert!(d.poll_gesture_json().is_none(), "handDetection off — 이벤트 없음");

    // 웜 detach → 같은 설정으로 즉시 재가동 (세션 유지 — 로드 없이)
    d.detach();
    let j = d.focus_json();
    assert!(j.contains("INITIALIZING"), "detach는 상태 리셋: {j}");
    for i in 0..6 {
        run(&mut d, 1000.0 + i as f64 * 100.0);
    }
    assert!(
        d.focus_json().contains("\"status\":\"FOCUSED\""),
        "웜 재가동 실패: {}",
        d.focus_json()
    );

    // 합성 출력 확인 — 배경 단색이 실제로 깔렸는지 (우상단 = 배경 영역)
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("director-staging"),
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
        .unwrap()
        .remove(0);
    let i = ((10 * w + 245) * 4) as usize; // 우상단 — 인물 밖
    assert!(
        px[i + 1] > 90 && px[i + 1] > px[i] && px[i + 1] > px[i + 2],
        "배경 #00a05a 미반영: rgb=({},{},{})",
        px[i],
        px[i + 1],
        px[i + 2]
    );
    println!(
        "director e2e OK — focus={} bg=({},{},{})",
        d.focus_json(),
        px[i],
        px[i + 1],
        px[i + 2]
    );
}

/// 회귀: 메이크업을 끄면(패치 `makeup:null`) 립 오버레이가 즉시 사라져야 한다.
/// 끄면 face 태스크가 안 돌아 update_face_fx가 호출되지 않고, compose uniform에
/// 마지막 mk_map이 잔존해 입술이 마지막 얼굴 위치에 고정 표시되던 버그의 재현.
#[test]
fn makeup_off_clears_stale_overlay() {
    let (Ok(seg), Ok(det), Ok(lm), Ok(rgb)) = (
        std::fs::read(SEG),
        std::fs::read(FACE_DET),
        std::fs::read(FACE_LM),
        std::fs::read(FRAME),
    ) else {
        eprintln!("skip: 모델/픽스처 없음 (make convert-rvm-web convert-mediapipe)");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let mut d = Director::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
    d.set_model("seg", seg).unwrap();
    d.set_model("face_det", det).unwrap();
    d.set_model("face_lm", lm).unwrap();

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mk-target"),
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

    let mut t_ms = 0.0f64;
    let mut run = |d: &mut Director, n: usize, t: &mut f64| {
        for _ in 0..n {
            pollster::block_on(d.with_frame(&ctx, w, h, |tex| {
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
            }))
            .unwrap();
            pollster::block_on(d.frame(&ctx, w, h, Some(&tview), Some(&rgb), *t)).unwrap();
            *t += 100.0;
        }
    };
    let capture = || {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mk-staging"),
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
        pollster::block_on(ai_gpu::readback::read_buffers(&ctx, &[&staging]))
            .unwrap()
            .remove(0)
    };
    let max_diff = |a: &[u8], b: &[u8]| {
        a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
    };

    // 1) 메이크업 없는 기준 프레임 (마스크 EMA 수렴까지 돌린다)
    d.apply_json(r##"{"background":"#00a05a"}"##).unwrap();
    run(&mut d, 8, &mut t_ms);
    let clean = capture();

    // 2) 메이크업 on — 립 오버레이가 실제로 그려져야 테스트가 유효
    d.apply_json(
        r##"{"makeup":{"enabled":true,"intensity":1.0,
             "lip":{"color":"#ff0000","alpha":1.0},
             "blush":{"color":"#ff4466","alpha":0.9,"size":0.35},
             "shadow":{"color":"#aa3355","alpha":0.8}}}"##,
    )
    .unwrap();
    run(&mut d, 6, &mut t_ms);
    let on = capture();
    assert!(
        max_diff(&on, &clean) > 40,
        "메이크업이 그려지지 않음 — 테스트 무효 (max diff {})",
        max_diff(&on, &clean)
    );

    // 3) 메이크업 off — 잔존 오버레이 없이 기준 프레임으로 복귀해야 한다
    d.apply_json(r##"{"makeup":null}"##).unwrap();
    run(&mut d, 3, &mut t_ms);
    let off = capture();
    assert!(
        max_diff(&off, &clean) <= 8,
        "메이크업 off 후 오버레이 잔존 (max diff {})",
        max_diff(&off, &clean)
    );
}
