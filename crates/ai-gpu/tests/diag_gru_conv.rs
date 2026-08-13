//! GRU candidate conv 단독 정확도 (진단용, #[ignore])
//!
//! 실그래프 이분탐색에서 op[89] (9×16, concat 2×64 → cout 64, k3, act=Tanh)가
//! CPU 대비 max_err 1.0으로 갈린다. 바로 앞 op[87] (같은 shape, cout 128, act=Sigmoid)은
//! 통과한다. 변형(Direct/Splitk)·뷰 흡수·슬롯 재사용은 이미 배제됐다.
//! 활성화와 cout을 격자로 돌려 어느 축이 범인인지 가른다.

use ai_core::ops::Conv2d;
use ai_core::rng::XorShift32;
use ai_core::{pack, reference, Activation, DType, TensorDesc};
use ai_gpu::kernel::KernelSpec;
use ai_gpu::kernels::common::source::SrcView;
use ai_gpu::kernels::conv_igemm::{ConvIgemmSpec, IgemmVariant};
use ai_gpu::testsuite::{storage_in, storage_out};
use ai_gpu::GpuContext;

fn run(ctx: &GpuContext, spec: &dyn KernelSpec, bufs: &[&ai_gpu::wgpu::Buffer], desc: &TensorDesc) -> Vec<f32> {
    use ai_gpu::wgpu;
    let compiled = pollster::block_on(ai_gpu::kernel::compile(ctx, spec)).unwrap();
    let out = storage_out(ctx, desc.size_bytes());
    let all: Vec<&wgpu::Buffer> = bufs.iter().copied().chain(std::iter::once(&out)).collect();
    let pbuf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&pbuf, 0, &[0u8; 16]);
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &pbuf, offset: 0, size: Some(std::num::NonZeroU64::new(256).unwrap()),
        }),
    }];
    for (i, b) in all.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: (i + 1) as u32, resource: b.as_entire_binding() });
    }
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &compiled.bgl, entries: &entries,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: desc.size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
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

#[test]
#[ignore]
fn diag_gru_conv() {
    let ctx = GpuContext::new_blocking().unwrap();
    let dt = DType::F32;
    let (ih, iw) = (9u32, 16u32);
    println!("{:<10} {:<8} {:>8} {:>12} {:>12}", "act", "cout", "변형", "max_err", "판정");

    // viewed=true: 파트0을 128채널 텐서의 뒤쪽 64채널(뷰)로 준다 — 실그래프 GRU 구성.
    // 지금까지 테스트가 전부 plain 파트만 봐서 이 케이스가 비어 있었다.
    for viewed in [false, true] {
    for act in [Activation::Tanh] {
        for cout in [64u32] {
            for force in [None, Some(IgemmVariant::Direct), Some(IgemmVariant::Splitk)] {
                let cin = 128u32;
                let op = Conv2d {
                    cin, cout, kh: 3, kw: 3, sh: 1, sw: 1,
                    pad: [1; 4], dil: 1, groups: 1, act,
                };
                let dout = TensorDesc::new(ih, iw, cout, dt);
                // concat 2파트 (실그래프와 동일 구성)
                let parts: Vec<Vec<f32>> = (0..2)
                    .map(|i| {
                        let d = TensorDesc::new(ih, iw, 64, dt);
                        XorShift32::new(11 + i).vec_f32(d.elems())
                    })
                    .collect();
                let px = (ih * iw) as usize;
                let bufs_owned: Vec<ai_gpu::wgpu::Buffer> = if viewed {
                    // 파트0 = 128채널 백킹의 채널 64..128 구간
                    let big_d = TensorDesc::new(ih, iw, 128, dt);
                    let mut big = vec![0f32; px * 128];
                    for p in 0..px {
                        for c in 0..64 {
                            big[p * 128 + c] = -1.0; // 앞쪽 64채널은 더미 (읽히면 결과가 깨진다)
                            big[p * 128 + 64 + c] = parts[0][p * 64 + c];
                        }
                    }
                    vec![
                        storage_in(&ctx, &pack::pack_nhwc(&big, &big_d)),
                        storage_in(&ctx, &pack::pack_nhwc(&parts[1], &TensorDesc::new(ih, iw, 64, dt))),
                    ]
                } else {
                    parts
                        .iter()
                        .map(|p| storage_in(&ctx, &pack::pack_nhwc(p, &TensorDesc::new(ih, iw, 64, dt))))
                        .collect()
                };
                let wts = XorShift32::new(3).vec_f32((cout * cin * 9) as usize);
                let bias = XorShift32::new(4).vec_f32(cout as usize);
                let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, 3, 3, 4, dt);
                let wbuf = storage_in(&ctx, &wb);
                let bbuf = storage_in(&ctx, &pack::pack_bias(&bias, cout, dt));
                let mut spec = ConvIgemmSpec::from_op(&op, ih, iw, false, dt);
                spec.srcs[0] = if viewed { SrcView::view(64, 128, 64) } else { SrcView::plain(64) };
                spec.srcs[1] = SrcView::plain(64);
                spec.force_variant = force;

                let mut bufs: Vec<&ai_gpu::wgpu::Buffer> = bufs_owned.iter().collect();
                bufs.push(&wbuf);
                bufs.push(&bbuf);
                let got = run(&ctx, &spec, &bufs, &dout);

                // CPU 레퍼런스 (파트를 채널축으로 이어붙임)
                let mut cat = vec![0f32; px * cin as usize];
                for p in 0..px {
                    for (i, part) in parts.iter().enumerate() {
                        for c in 0..64usize {
                            cat[p * cin as usize + i * 64 + c] = part[p * 64 + c];
                        }
                    }
                }
                let want = reference::conv::conv2d(&op, ih, iw, &cat, &wts, Some(&bias), None);
                let err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
                println!(
                    "{:<10} {cout:<8} {:>8} viewed={viewed:<5} {err:>12.3e} {:>12}",
                    act.tag(),
                    format!("{:?}", spec.variant()),
                    if err < 1e-3 { "OK" } else { "실패" }
                );
            }
        }
    }
    }
}
