//! 스레드 기아 conv 변형 스윕 (진단용, #[ignore])
//!
//! RVM 프레임의 갭은 전부 저해상 심층 conv 44개에 있다 (실측: 병렬성 충분한 op은
//! 757 GMAC/s로 webgl2 평균 708을 넘고, 기아 op은 258 GMAC/s). 현재 변형 선택 정책
//! (`nblk*ng < 4096 && kg >= 16` → Splitk)이 실제로 최선인지 shape별로 강제 A/B한다.
//!
//! **측정 위생** (이 세션에서 세 번 데인 것):
//! - 반복 3회 후 최소값 — 한 번 재고 결론 내지 말 것
//! - 같은 출력 버퍼에 연속 기록하므로 디스패치 간 배리어가 포함된다(실그래프와 유사)
//! - 단, 200회 연속이라 가중치가 L2에 상주한다 → **절대값은 낙관적, 변형 간 상대비교만
//!   유효**. 최종 판정은 실그래프 프레임타임으로 한다.

use ai_core::ops::Conv2d;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};
use ai_gpu::bench::bench_kernel;
use ai_gpu::kernels::common::source::SrcView;
use ai_gpu::kernels::conv_igemm::{ConvIgemmSpec, IgemmVariant};
use ai_gpu::testsuite::{storage_in, storage_out};
use ai_gpu::GpuContext;

/// 후보 커널을 1회 실행하고 출력을 f32로 읽어온다 (정확도 검증용).
/// **속도만 재고 끝내면 "빠른데 틀린" 조합을 채택하게 된다** — SPLIT×CPW≠256 조합이
/// K의 절반만 계산하면서 2배 빨라 보였던 실제 사례가 있었다.
fn run_and_read(
    ctx: &GpuContext,
    spec: &dyn ai_gpu::kernel::KernelSpec,
    in_bufs: &[&ai_gpu::wgpu::Buffer],
    desc: &TensorDesc,
) -> Vec<f32> {
    use ai_gpu::wgpu;
    // 후보마다 **새** 출력 버퍼 — 공유하면 앞선 정답 후보가 남긴 값을 읽고
    // "통과"하는 가짜 검증이 된다. WebGPU 버퍼는 0으로 초기화되므로 별도 clear 불필요.
    let out = storage_out(ctx, desc.size_bytes());
    let bufs: Vec<&wgpu::Buffer> = in_bufs.iter().copied().chain(std::iter::once(&out)).collect();
    let compiled = pollster::block_on(ai_gpu::kernel::compile(ctx, spec)).unwrap();
    let pbuf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&pbuf, 0, &[0u8; 16]);
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &pbuf,
            offset: 0,
            size: Some(std::num::NonZeroU64::new(256).unwrap()),
        }),
    }];
    for (i, b) in bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: (i + 1) as u32, resource: b.as_entire_binding() });
    }
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compiled.bgl,
        entries: &entries,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: desc.size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&compiled.pipeline);
        pass.set_bind_group(0, &bg, &[0]);
        let g = spec.workgroups();
        pass.dispatch_workgroups(g[0], g[1], g[2]);
    }
    enc.copy_buffer_to_buffer(&out, 0, &staging, 0, desc.size_bytes());
    ctx.queue.submit([enc.finish()]);
    let bytes = pollster::block_on(ai_gpu::readback::read_buffers(ctx, &[&staging])).unwrap().remove(0);
    pack::unpack_nhwc(&bytes, desc)
}

/// (라벨, ih, iw, cin, cout, concat 파트 채널들)
/// plan에서 뽑은 실제 기아 shape — 대부분 concat 융합 입력이라 파트 구성까지 재현한다.
const SHAPES: &[(&str, u32, u32, u32, u32, &[u32])] = &[
    ("decode3 171->80", 18, 32, 171, 80, &[128, 40, 3]),
    ("decode2 107->40", 36, 64, 107, 40, &[80, 24, 3]),
    ("gru4 128->128", 9, 16, 128, 128, &[64, 64]),
    ("gru4 128->64", 9, 16, 128, 64, &[64, 64]),
    ("gru3 80->80", 18, 32, 80, 80, &[40, 40]),
    ("gru3 80->40", 18, 32, 80, 40, &[40, 40]),
    ("gru2 40->40", 36, 64, 40, 40, &[20, 20]),
    ("gru2 40->20", 36, 64, 40, 20, &[20, 20]),
    ("gru1 32->32", 72, 128, 32, 32, &[16, 16]),
    ("gru1 32->16", 72, 128, 32, 16, &[16, 16]),
];

const REPEATS: usize = 3;

#[test]
#[ignore]
fn sweep_starved() {
    let ctx = GpuContext::new_blocking().unwrap();
    let dt = DType::F32;
    println!("adapter: {}\n", ctx.caps.info.name);
    println!("Splitk 기하(SPLIT×CPW) 스윕 — 현재 정책 vs 전 조합 최선\n");

    let mut tot_now = 0.0;
    let mut tot_best = 0.0;
    for (label, ih, iw, cin, cout, parts) in SHAPES {
        let op = Conv2d {
            cin: *cin,
            cout: *cout,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
            };
        let (oh, ow) = op.out_hw(*ih, *iw);
        let dout = TensorDesc::new(oh, ow, *cout, dt);
        let wts = XorShift32::new(3).vec_f32((cout * cin * 9) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, *cout, *cin, 3, 3, 4, dt);
        let bias = pack::pack_bias(&vec![0f32; *cout as usize], *cout, dt);
        let flops = 2.0 * (oh * ow) as f64 * *cout as f64 * *cin as f64 * 9.0;

        // concat 파트별 입력 버퍼 (실그래프와 같은 멀티소스 구성)
        let part_bufs: Vec<ai_gpu::wgpu::Buffer> = parts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d = TensorDesc::new(*ih, *iw, *c, dt);
                let data = XorShift32::new(5 + i as u32).vec_f32(d.elems());
                storage_in(&ctx, &pack::pack_nhwc(&data, &d))
            })
            .collect();
        let out = storage_out(&ctx, dout.size_bytes());
        let wbuf = storage_in(&ctx, &wb);
        let bbuf = storage_in(&ctx, &bias);

        let mut srcs = [SrcView::NONE; 3];
        for (i, c) in parts.iter().enumerate() {
            srcs[i] = SrcView::plain(*c);
        }
        let base = ConvIgemmSpec {
            srcs,
            ..ConvIgemmSpec::from_op(&op, *ih, *iw, false, dt)
        };
        let policy = base.variant();

        let mut in_bufs: Vec<&ai_gpu::wgpu::Buffer> = part_bufs.iter().collect();
        in_bufs.push(&wbuf);
        in_bufs.push(&bbuf);
        let mut bufs = in_bufs.clone();
        bufs.push(&out);

        // Direct, 그리고 Splitk × 기하 조합
        let mut cands: Vec<(String, ConvIgemmSpec)> =
            vec![("Direct".into(), ConvIgemmSpec { force_variant: Some(IgemmVariant::Direct), ..base })];
        // SPLIT×CPW == 256 (워크그룹 256스레드)인 조합만 유효하다.
        for (sp, cp) in [(2u32, 128u32), (4, 64), (8, 32), (16, 16), (32, 8)] {
            cands.push((
                format!("Splitk{sp}x{cp}"),
                ConvIgemmSpec {
                    force_variant: Some(IgemmVariant::Splitk),
                    force_geom: Some((sp, cp)),
                    ..base
                },
            ));
        }
        // CPU 레퍼런스 (파트를 채널축으로 이어붙인 입력)
        let mut cat = vec![0f32; (*ih * *iw * *cin) as usize];
        {
            let mut coff = 0u32;
            for (i, c) in parts.iter().enumerate() {
                let d = TensorDesc::new(*ih, *iw, *c, dt);
                let data = XorShift32::new(5 + i as u32).vec_f32(d.elems());
                for p in 0..(*ih * *iw) as usize {
                    for ch in 0..*c as usize {
                        cat[p * *cin as usize + coff as usize + ch] = data[p * *c as usize + ch];
                    }
                }
                coff += c;
            }
        }
        let want = ai_core::reference::conv::conv2d(
            &op,
            *ih,
            *iw,
            &cat,
            &wts,
            Some(&vec![0f32; *cout as usize]),
            None,
        );

        let mut results: Vec<(f64, String)> = Vec::new();
        for (name, spec) in &cands {
            // 정확도 먼저 — 틀린 조합은 속도를 재지도 않는다
            let got = run_and_read(&ctx, spec, &in_bufs, &dout);
            let err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f32, f32::max);
            if err > 1e-3 {
                println!("  {label} {name}: 정확도 실패 max_err {err:.3e} — 후보 제외");
                continue;
            }
            let mut best = f64::MAX;
            for _ in 0..REPEATS {
                let r =
                    pollster::block_on(bench_kernel(&ctx, spec, &[0u8; 16], &bufs, flops)).unwrap();
                best = best.min(r.gpu_min_ms.unwrap_or(r.wall_ms) * 1e3);
            }
            results.push((best, name.clone()));
        }
        // 현재 정책이 고르는 것
        let now = {
            let spec = ConvIgemmSpec { force_variant: Some(policy), ..base };
            let mut best = f64::MAX;
            for _ in 0..REPEATS {
                let r =
                    pollster::block_on(bench_kernel(&ctx, &spec, &[0u8; 16], &bufs, flops)).unwrap();
                best = best.min(r.gpu_min_ms.unwrap_or(r.wall_ms) * 1e3);
            }
            best
        };
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (bus, bname) = &results[0];
        let mark = if *bus < now * 0.95 { "  ← 개선 여지" } else { "" };
        println!("{label:<20} 현재({policy:?}) {now:7.1}µs | 최선 {bname:>12} {bus:7.1}µs{mark}");
        tot_now += now;
        tot_best += bus;
    }
    println!(
        "\n합계: 현재정책 {tot_now:.0}µs / 최선 {tot_best:.0}µs  (개선 여지 {:.0}µs)",
        tot_now - tot_best
    );
}
