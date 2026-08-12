//! RVM 144×256 실그래프 핫 shape 단독 마이크로벤치 (진단용, #[ignore])
//!
//! plan.json에서 추출한 실제 shape을 단독 실행해 "커널 자체 효율"을 측정한다.
//! 그래프 wall 프레임타임과의 차이 = 디스패치/배리어/스케줄링 오버헤드.
//! per-op 타임스탬프 프로파일(profile_infer)은 Metal 패스 중첩 때문에 부풀려지므로
//! 이 단독 실측이 커널 최적화의 근거 수치다.

use ai_core::ops::Conv2d;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};
use ai_gpu::bench::bench_kernel;
use ai_gpu::kernels::conv_dw::ConvDwSpec;
use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
use ai_gpu::kernels::gemm_pw::GemmPwSpec;
use ai_gpu::testsuite::{storage_in, storage_out};
use ai_gpu::GpuContext;

fn filled(ctx: &GpuContext, desc: &TensorDesc, seed: u32) -> ai_gpu::wgpu::Buffer {
    let data = XorShift32::new(seed).vec_f32(desc.elems());
    storage_in(ctx, &pack::pack_nhwc(&data, desc))
}

fn report(tag: &str, n: u32, r: &ai_gpu::bench::BenchResult) {
    let us = r.gpu_min_ms.unwrap_or(r.wall_ms) * 1e3;
    println!(
        "{n}x {tag:<44} {us:8.1}us  (x{n} = {:8.1}us)  {:7.1} GFLOP/s",
        us * n as f64,
        r.gflops
    );
}

#[test]
#[ignore]
fn bench_rvm_hot() {
    let ctx = GpuContext::new_blocking().unwrap();
    let dt = DType::F32;
    println!("adapter: {}\n", ctx.caps.info.name);
    let mut sum = 0.0;

    // ---- 일반 conv k3 (igemm) — (개수, ih, iw, cin, cout, k, s) ----
    for (n, ih, iw, cin, cout, k, s) in [
        (1u32, 144u32, 256u32, 35u32, 16u32, 3u32, 1u32),
        (1, 144, 256, 16, 16, 3, 1),
        (1, 144, 256, 3, 16, 3, 2), // 스템
        (1, 72, 128, 59, 32, 3, 1),
        (1, 72, 128, 32, 32, 3, 1),
        (1, 72, 128, 32, 16, 3, 1),
        (1, 36, 64, 107, 40, 3, 1),
        (1, 18, 32, 171, 80, 3, 1),
    ] {
        let op = Conv2d {
            cin,
            cout,
            kh: k,
            kw: k,
            sh: s,
            sw: s,
            pad: [(k - 1) / 2; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
        };
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, cin, dt);
        let dout = TensorDesc::new(oh, ow, cout, dt);
        let spec = ConvIgemmSpec::from_op(&op, ih, iw, false, dt);
        let wts = XorShift32::new(3).vec_f32((cout * cin * k * k) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, k, k, 4, dt);
        let bufs = [
            filled(&ctx, &din, 5),
            storage_in(&ctx, &wb),
            storage_in(&ctx, &pack::pack_bias(&vec![0f32; cout as usize], cout, dt)),
            storage_out(&ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (oh * ow) as f64 * cout as f64 * cin as f64 * (k * k) as f64;
        let r = pollster::block_on(bench_kernel(
            &ctx,
            &spec,
            &[0u8; 16],
            &bufs.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();
        report(
            &format!("igemm {ih}x{iw} {cin}->{cout} k{k} s{s} [{:?}]", spec.variant()),
            n,
            &r,
        );
        sum += r.gpu_min_ms.unwrap_or(r.wall_ms) * n as f64;
    }

    // ---- 1×1 conv (gemm_pw) — (개수, h, w, cin, cout) ----
    for (n, h, w, cin, cout) in [
        (1u32, 144u32, 256u32, 24u32, 16u32),
        (2, 144, 256, 16, 4),
        (1, 144, 256, 16, 16),
        (1, 72, 128, 16, 64),
        (2, 36, 64, 24, 72),
        (2, 18, 32, 40, 120),
        (1, 18, 32, 40, 240),
        (3, 9, 16, 160, 960),
        (2, 9, 16, 960, 160),
        (2, 9, 16, 112, 672),
        (2, 9, 16, 672, 112),
        (2, 1, 1, 960, 240),
        (2, 1, 1, 240, 960),
    ] {
        let din = TensorDesc::new(h, w, cin, dt);
        let dout = TensorDesc::new(h, w, cout, dt);
        let spec = GemmPwSpec {
            m: h * w,
            kg: din.cg(),
            ng: dout.cg(),
            act: Activation::Relu,
            residual: false,
            dt,
        };
        let wts = XorShift32::new(3).vec_f32((cout * cin) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, 1, 1, 4, dt);
        let bufs = [
            filled(&ctx, &din, 5),
            storage_in(&ctx, &wb),
            storage_in(&ctx, &pack::pack_bias(&vec![0f32; cout as usize], cout, dt)),
            storage_out(&ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (h * w) as f64 * cout as f64 * cin as f64;
        let r = pollster::block_on(bench_kernel(
            &ctx,
            &spec,
            &[0u8; 16],
            &bufs.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();
        report(&format!("pw {h}x{w} {cin}->{cout}"), n, &r);
        sum += r.gpu_min_ms.unwrap_or(r.wall_ms) * n as f64;
    }

    // ---- depthwise — (개수, ih, iw, c, k, s, d) ----
    for (n, ih, iw, c, k, s, d) in [
        (4u32, 144u32, 256u32, 4u32, 3u32, 1u32, 1u32),
        (1, 72, 128, 64, 3, 2, 1),
        (1, 36, 64, 72, 3, 1, 1),
        (2, 18, 32, 120, 5, 1, 1),
        (2, 9, 16, 960, 5, 1, 2),
    ] {
        let op = Conv2d {
            cin: c,
            cout: c,
            kh: k,
            kw: k,
            sh: s,
            sw: s,
            pad: [(d * (k - 1)) / 2; 4],
            dil: d,
            groups: c,
            act: Activation::Relu,
        };
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, c, dt);
        let dout = TensorDesc::new(oh, ow, c, dt);
        let spec = ConvDwSpec::from_op(&op, ih, iw, false, dt);
        let wts = XorShift32::new(3).vec_f32((c * k * k) as usize);
        let bufs = [
            filled(&ctx, &din, 5),
            storage_in(&ctx, &pack::pack_weights_dw(&wts, c, k, k, dt)),
            storage_in(&ctx, &pack::pack_bias(&vec![0f32; c as usize], c, dt)),
            storage_out(&ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (oh * ow) as f64 * c as f64 * (k * k) as f64;
        let r = pollster::block_on(bench_kernel(
            &ctx,
            &spec,
            &[0u8; 16],
            &bufs.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();
        report(&format!("dw {ih}x{iw} c{c} k{k} s{s} d{d}"), n, &r);
        sum += r.gpu_min_ms.unwrap_or(r.wall_ms) * n as f64;
    }

    println!("\n주요 shape 합계(개수 가중): {sum:.3} ms  (실그래프 wall과 비교할 것)");
}
