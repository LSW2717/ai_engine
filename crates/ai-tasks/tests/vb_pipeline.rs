//! VideoPipeline 네이티브 게이트 — **마스크가 실제로 생기는지**를 실프레임으로 검증.
//!
//! 브라우저 스모크("크래시 없음")가 마스크 전멸을 PASS로 통과시킨 사고의 재발
//! 방지: RVM + 실제 사람 프레임(tests/data/frame_256x144.rgb, 전경 ~18.5%)을
//! 단색 배경으로 합성해 전경 비율을 실측한다. pha 평균도 찍어 어느 단계가
//! 죽었는지(추론 vs 마스크 스택) 즉시 갈린다.
//!
//! 모델(.sw)이 없으면 스킵 — `make convert-rvm-web`.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::vb::VideoPipeline;
use ai_tasks::GpuSession;

const SW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
const FRAME: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/frame_256x144.rgb");

#[test]
fn rvm_mask_appears() {
    let (Ok(sw), Ok(rgb)) = (std::fs::read(SW), std::fs::read(FRAME)) else {
        eprintln!("skip: rvm_256x144.sw 또는 프레임 없음");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let ctx = GpuContext::new_blocking().unwrap();
    let mut seg = pollster::block_on(GpuSession::load(&ctx, &sw)).expect("RVM 로드");
    // 이등분 0: 먼저 **0 입력**을 업로드해 pha≈0으로 만든다. 이후 파이프라인이
    // 0.18을 만들면 전처리 컴퓨트가 실제로 입력 버퍼에 썼다는 증명이 된다
    // (전처리가 안 쓰면 0 입력이 재사용돼 pha가 0에 머문다 — 실제로 그랬던 버그).
    let zeros = vec![0f32; (256 * 144 * 3) as usize];
    seg.upload(&ctx, &zeros).unwrap();
    pollster::block_on(seg.infer(&ctx)).unwrap();
    let pha0 = pollster::block_on(seg.read_output(&ctx, "pha")).unwrap();
    println!("sanity(zero input) pha mean {:.4} (기대 ~0)", pha0.iter().sum::<f32>() / pha0.len() as f32);

    let mut pipe = VideoPipeline::new(&ctx, wgpu::TextureFormat::Rgba8Unorm);
    pipe.apply_json(r##"{"background":"#00ff00"}"##).unwrap();

    // 출력 타깃 (COPY_SRC로 리드백)
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vb-test-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tview = target.create_view(&Default::default());

    // 프레임 업로드 (u8 RGB → RGBA)
    let mut rgba = vec![255u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
    }
    let mut pha_first: Vec<f32> = Vec::new();
    for i in 0..5 {
        // EMA 수렴을 위해 수 프레임 (pha 모드 α=0.6/0.9)
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
        pollster::block_on(pipe.process_gpu(&ctx, &mut seg, w, h, &tview)).unwrap();
        if i == 0 {
            pha_first = pollster::block_on(seg.read_output(&ctx, "pha")).unwrap();
        }
    }

    // 진단 1: pha 자체가 살아있나 (추론+전처리 검증 — 여기가 0이면 전처리/추론 문제)
    let pha = pollster::block_on(seg.read_output(&ctx, "pha")).expect("pha 리드백");
    let pha_mean: f32 = pha.iter().sum::<f32>() / pha.len() as f32;
    println!("pha mean {pha_mean:.4} (기대 ~0.18)");
    // 순환 상태(r1~r4) 피드백 증명: 같은 프레임 반복 시 GRU 워밍업으로 pha가
    // 진화해야 한다. 상태가 0에 고정(피드백 끊김)이면 매 프레임 완전 동일.
    let state_evo = pha_first
        .iter()
        .zip(&pha)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("순환상태 진화(1프레임 vs 5프레임 pha 최대차) {state_evo:.5} — 0이면 피드백 끊김");
    assert!(state_evo > 1e-4, "순환 상태 피드백이 안 돈다 (핑퐁 사망)");

    // 진단 2: 합성 출력 전경 비율 (초록 배경이 아닌 픽셀)
    let bpr = w * 4; // 1024 — 256 정렬 ✓
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vb-test-read"),
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let data = slice.get_mapped_range().expect("map").to_vec();
    let mut fg = 0usize;
    for px in data.chunks_exact(4) {
        // 순수 초록 배경(#00ff00)에서 충분히 벗어난 픽셀 = 전경
        if !(px[0] < 60 && px[1] > 180 && px[2] < 60) {
            fg += 1;
        }
    }
    let frac = fg as f32 / (w * h) as f32;
    println!("전경 비율 {frac:.3} (기대 0.10~0.35)");
    // 프레임타임 분리 계측 (동기화 포함 — 측정규율): 추론 단독 vs 파이프라인 전체.
    // 이펙트 스택(EMA/JBF/refine/blur/compose)이 추론에 얹는 비용을 격리한다.
    let reps = 50;
    for _ in 0..5 {
        pollster::block_on(seg.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        pollster::block_on(seg.infer(&ctx)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let infer_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        pollster::block_on(pipe.process_gpu(&ctx, &mut seg, w, h, &tview)).unwrap();
    }
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    let full_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "프레임타임(동기화 포함): 추론 단독 {infer_ms:.2}ms | 파이프라인 전체 {full_ms:.2}ms | 이펙트 스택 오버헤드 {:.2}ms",
        full_ms - infer_ms
    );

    assert!(pha_mean > 0.05, "pha 전멸 — 전처리(f16)/추론 경로 사망");
    assert!(frac > 0.05 && frac < 0.6, "합성 전경 비율 비정상 {frac} — 마스크 스택 사망");

    // ── bbox 게이트 1 (CPU 스캔 경로): 사각 마스크 주입 → 정규화 bbox 정밀 ──
    // process_gpu_mask는 마스크가 CPU에 있으므로 GPU 리덕션 없이 즉시 스캔한다
    // (v-ai _scanPersonBBox CPU 힙 경로 등가: v>0.5, 1% 문턱, 리드백 0)
    pipe.apply_json(r#"{"framing":{"enabled":true}}"#).unwrap();
    let (mw, mh) = (256usize, 144usize);
    let mut mask = vec![0f32; mw * mh];
    for y in 36..108 {
        for x in 64..192 {
            mask[y * mw + x] = 1.0;
        }
    }
    pipe.process_gpu_mask(&ctx, &seg, &mask, 1, 256, 144, false, w, h, &tview).unwrap();
    let bb = pipe.last_bbox().expect("CPU bbox 스캔 실패");
    println!("bbox(cpu) {bb:?} (기대 [0.25, 0.75, 0.25, 0.75])");
    for (got, want) in bb.iter().zip([0.25f32, 0.75, 0.25, 0.75]) {
        assert!((got - want).abs() < 0.01, "CPU bbox 불일치: {bb:?}");
    }
    // 인물 소실: 0 마스크 → None (1% 문턱)
    let zero_mask = vec![0f32; mw * mh];
    pipe.process_gpu_mask(&ctx, &seg, &zero_mask, 1, 256, 144, false, w, h, &tview).unwrap();
    assert!(pipe.last_bbox().is_none(), "0 마스크인데 bbox가 남음 — 1% 문턱 사망");

    // ── bbox 게이트 2 (GPU 리덕션 경로): 실추론 마스크 → 20B 리드백 링 →
    // pha CPU 스캔과 교차검증 (GPU는 EMA 이후 마스크라 경계 1~2px 차 허용) ──
    for _ in 0..4 {
        pollster::block_on(pipe.process_gpu(&ctx, &mut seg, w, h, &tview)).unwrap();
        pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();
    }
    let gpu_bb = pipe.last_bbox().expect("GPU bbox 리드백 미도착 — 링/펌프 사망");
    let pha = pollster::block_on(seg.read_output(&ctx, "pha")).unwrap();
    let cpu_bb = ai_tasks::features::vb::framing::scan_bbox_cpu(&pha, mw, mh, 1)
        .expect("pha에 인물 없음");
    println!("bbox(gpu) {gpu_bb:?} vs pha cpu 스캔 {cpu_bb:?}");
    for (g, c) in gpu_bb.iter().zip(cpu_bb) {
        assert!((g - c).abs() <= 0.021, "GPU/CPU bbox 불일치: {gpu_bb:?} vs {cpu_bb:?}");
    }
}
