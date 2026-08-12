//! 공유 마이크로벤치마크 코어 — ai-bench(네이티브)와 wasm 데모가 같은 루틴을 돌린다.
//!
//! 방법: 파이프라인 생성 시간 측정 → 워밍업 5회 → 측정 20회(각 = 한 패스에
//! 10연속 디스패치, 패스 경계 타임스탬프) ÷ 10 — Chrome의 100µs 양자화 무력화.
//! 같은 출력 버퍼에 연속 기록하므로 dispatch 사이 배리어가 포함된다(실그래프의
//! 의존 체인과 유사한 조건). min/median과 GFLOP/s, wall-clock 폴백을 보고한다.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use ai_core::ops::CoordMode;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};

use crate::context::GpuContext;
use crate::kernel::{self, KernelSpec};
use crate::kernels::conv_dw::ConvDwSpec;
use crate::kernels::conv_igemm::ConvIgemmSpec;
use crate::kernels::elementwise::{ElementwiseSpec, EwOperand};
use crate::kernels::gemm_pw::GemmPwSpec;
use crate::kernels::gpool::GpoolSpec;
use crate::kernels::resize_bilinear::ResizeBilinearSpec;
use crate::profile::GpuProfiler;
use crate::readback;
use crate::testsuite::{params_buffer, storage_in, storage_out};
use ai_core::ops::BinaryOp;

const WARMUP: u32 = 5;
const MEASURES: u32 = 20;
const DISPATCHES_PER_PASS: u32 = 10;

pub struct BenchResult {
    pub name: String,
    /// 디스패치당 GPU 시간 (timestamp, min) — 미지원 기기는 None
    pub gpu_min_ms: Option<f64>,
    /// 디스패치당 GPU 시간 (timestamp, median)
    pub gpu_median_ms: Option<f64>,
    /// 디스패치당 wall-clock (submit+wait 전체 ÷ 총 디스패치)
    pub wall_ms: f64,
    /// GFLOP/s (gpu_min 기준, 없으면 wall 기준)
    pub gflops: f64,
    /// 파이프라인 생성+컴파일 시간
    pub pipeline_ms: f64,
}

async fn bench_kernel(
    ctx: &GpuContext,
    spec: &dyn KernelSpec,
    params: &[u8],
    buffers: &[&wgpu::Buffer],
    flops_per_dispatch: f64,
) -> Result<BenchResult, String> {
    let t0 = Instant::now();
    let compiled = kernel::compile(ctx, spec).await?;
    let pipeline_ms = t0.elapsed().as_secs_f64() * 1e3;

    let pbuf = params_buffer(ctx, params);
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &pbuf,
            offset: 0,
            size: Some(std::num::NonZeroU64::new(256).unwrap()),
        }),
    }];
    for (i, buf) in buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (i + 1) as u32,
            resource: buf.as_entire_binding(),
        });
    }
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compiled.bgl,
        entries: &entries,
    });
    let groups = spec.workgroups();

    // 워밍업 — 백엔드 컴파일 강제
    {
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&compiled.pipeline);
            pass.set_bind_group(0, &bind_group, &[0]);
            for _ in 0..WARMUP {
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
        }
        ctx.queue.submit([enc.finish()]);
        readback::wait_idle(ctx).await?;
    }

    // 측정 — MEASURES개 패스를 한 번에 제출, 패스마다 타임스탬프
    let profiler = GpuProfiler::new(ctx, MEASURES);
    let wall_t0 = Instant::now();
    let mut enc =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for m in 0..MEASURES {
        let tw = profiler.as_ref().map(|p| p.timestamp_writes(m));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: tw,
        });
        pass.set_pipeline(&compiled.pipeline);
        pass.set_bind_group(0, &bind_group, &[0]);
        for _ in 0..DISPATCHES_PER_PASS {
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
    }
    if let Some(p) = &profiler {
        p.resolve(&mut enc, MEASURES);
    }
    ctx.queue.submit([enc.finish()]);
    readback::wait_idle(ctx).await?;
    let wall_total = wall_t0.elapsed().as_secs_f64() * 1e3;
    let wall_ms = wall_total / (MEASURES * DISPATCHES_PER_PASS) as f64;

    let (gpu_min_ms, gpu_median_ms) = if let Some(p) = &profiler {
        let mut per_pass = p.read_ms(ctx, MEASURES).await?;
        per_pass.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = per_pass[0] / DISPATCHES_PER_PASS as f64;
        let med = per_pass[per_pass.len() / 2] / DISPATCHES_PER_PASS as f64;
        (Some(min), Some(med))
    } else {
        (None, None)
    };

    let basis_ms = gpu_min_ms.unwrap_or(wall_ms);
    let gflops = flops_per_dispatch / (basis_ms * 1e-3) / 1e9;

    Ok(BenchResult {
        name: spec.cache_key(&ctx.caps),
        gpu_min_ms,
        gpu_median_ms,
        wall_ms,
        gflops,
        pipeline_ms,
    })
}

fn filled(ctx: &GpuContext, desc: &TensorDesc, seed: u32) -> wgpu::Buffer {
    let data = XorShift32::new(seed).vec_f32(desc.elems());
    storage_in(ctx, &pack::pack_nhwc(&data, desc))
}

/// 벤치 shape 표 — (a) RVM plan.json 실측 144×256 세트, (b) 512×288 스케일 세트
pub async fn run_benchmarks(ctx: &GpuContext) -> Result<Vec<BenchResult>, String> {
    let mut out = Vec::new();
    let dt = DType::F32;

    // ---- pointwise GEMM ----
    // (h, w, cin, cout): a4 와이드-M, a5 심층, a6 SE M=1, b3 512×288 와이드
    for (h, w, cin, cout, seed) in [
        (72u32, 128u32, 16u32, 64u32, 11u32), // (a) M=9216
        (9, 16, 576, 96, 12),                 // (a) 심층 M=144
        (1, 1, 576, 144, 13),                 // (a) SE M=1
        (144, 256, 16, 64, 14),               // (b) M=36864
        (18, 32, 576, 96, 15),                // (b) 심층 M=576 (tiled 경계)
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
        let wts = XorShift32::new(seed).vec_f32((cout * cin) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, 1, 1, 4, dt);
        let bias = vec![0f32; cout as usize];
        let bufs = [
            filled(ctx, &din, seed),
            storage_in(ctx, &wb),
            storage_in(ctx, &pack::pack_bias(&bias, cout, dt)),
            storage_out(ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (h * w) as f64 * cout as f64 * cin as f64;
        out.push(
            bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    // ---- depthwise ----
    for (ih, iw, c, k, s, seed) in [
        (72u32, 128u32, 64u32, 3u32, 1u32, 21u32), // (a)
        (36, 64, 96, 5, 2, 22),                    // (a) k5 s2
        (144, 256, 64, 3, 1, 23),                  // (b)
    ] {
        let op = ai_core::ops::Conv2d::depthwise(c, k, s, Activation::Relu);
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, c, dt);
        let dout = TensorDesc::new(oh, ow, c, dt);
        let spec = ConvDwSpec::from_op(&op, ih, iw, false, dt);
        let wts = XorShift32::new(seed).vec_f32((c * k * k) as usize);
        let bias = vec![0f32; c as usize];
        let bufs = [
            filled(ctx, &din, seed),
            storage_in(ctx, &pack::pack_weights_dw(&wts, c, k, k, dt)),
            storage_in(ctx, &pack::pack_bias(&bias, c, dt)),
            storage_out(ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (oh * ow) as f64 * c as f64 * (k * k) as f64;
        out.push(
            bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    // ---- 스템 conv (implicit GEMM direct) ----
    for (ih, iw, seed) in [(144u32, 256u32, 31u32), (288, 512, 32)] {
        let op = ai_core::ops::Conv2d {
            cin: 3,
            cout: 16,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Hardswish,
        };
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, 3, dt);
        let dout = TensorDesc::new(oh, ow, 16, dt);
        let spec = ConvIgemmSpec::from_op(&op, ih, iw, false, dt);
        let wts = XorShift32::new(seed).vec_f32(16 * 3 * 9);
        let (wb, _) = pack::pack_weights_conv(&wts, 16, 3, 3, 3, 4, dt);
        let bias = vec![0f32; 16];
        let bufs = [
            filled(ctx, &din, seed),
            storage_in(ctx, &wb),
            storage_in(ctx, &pack::pack_bias(&bias, 16, dt)),
            storage_out(ctx, dout.size_bytes()),
        ];
        let flops = 2.0 * (oh * ow) as f64 * 16.0 * 3.0 * 9.0;
        out.push(
            bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    // ---- f16 스토리지 변형 (지원 기기): 대역폭 중심 shape에서 델타 측정 ----
    if ctx.caps.f16 {
        let dt16 = DType::F16;
        // pw 와이드-M (512×288 스케일)
        {
            let (h, w, cin, cout) = (144u32, 256u32, 16u32, 64u32);
            let din = TensorDesc::new(h, w, cin, dt16);
            let dout = TensorDesc::new(h, w, cout, dt16);
            let spec = GemmPwSpec {
                m: h * w,
                kg: din.cg(),
                ng: dout.cg(),
                act: Activation::Relu,
                residual: false,
                dt: dt16,
            };
            let wts = XorShift32::new(71).vec_f32((cout * cin) as usize);
            let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, 1, 1, 4, dt16);
            let bias = vec![0f32; cout as usize];
            let bufs = [
                filled(ctx, &din, 71),
                storage_in(ctx, &wb),
                storage_in(ctx, &pack::pack_bias(&bias, cout, dt16)),
                storage_out(ctx, dout.size_bytes()),
            ];
            let flops = 2.0 * (h * w) as f64 * cout as f64 * cin as f64;
            out.push(
                bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops)
                    .await?,
            );
        }
        // dw 대공간 (512×288 스케일)
        {
            let (ih, iw, c) = (144u32, 256u32, 64u32);
            let op = ai_core::ops::Conv2d::depthwise(c, 3, 1, Activation::Relu);
            let (oh, ow) = op.out_hw(ih, iw);
            let din = TensorDesc::new(ih, iw, c, dt16);
            let dout = TensorDesc::new(oh, ow, c, dt16);
            let spec = ConvDwSpec::from_op(&op, ih, iw, false, dt16);
            let wts = XorShift32::new(72).vec_f32((c * 9) as usize);
            let bias = vec![0f32; c as usize];
            let bufs = [
                filled(ctx, &din, 72),
                storage_in(ctx, &pack::pack_weights_dw(&wts, c, 3, 3, dt16)),
                storage_in(ctx, &pack::pack_bias(&bias, c, dt16)),
                storage_out(ctx, dout.size_bytes()),
            ];
            let flops = 2.0 * (oh * ow) as f64 * c as f64 * 9.0;
            out.push(
                bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops)
                    .await?,
            );
        }
    }

    // ---- gpool ----
    for (h, w, c, seed) in [(9u32, 16u32, 576u32, 41u32), (18, 32, 576, 42)] {
        let din = TensorDesc::new(h, w, c, dt);
        let dout = TensorDesc::new(1, 1, c, dt);
        let spec = GpoolSpec { h, w, c, dt };
        let bufs = [filled(ctx, &din, seed), storage_out(ctx, dout.size_bytes())];
        let flops = (h * w * c) as f64;
        out.push(
            bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    // ---- resize 2× ----
    for (ih, iw, c, seed) in [(18u32, 32u32, 40u32, 51u32), (36, 64, 40, 52)] {
        let (oh, ow) = (ih * 2, iw * 2);
        let din = TensorDesc::new(ih, iw, c, dt);
        let dout = TensorDesc::new(oh, ow, c, dt);
        let spec =
            ResizeBilinearSpec { ih, iw, c, oh, ow, mode: CoordMode::HalfPixel, dt };
        let bufs = [filled(ctx, &din, seed), storage_out(ctx, dout.size_bytes())];
        let flops = 8.0 * (oh * ow * c) as f64;
        out.push(
            bench_kernel(ctx, &spec, &[0u8; 16], &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    // ---- elementwise add ----
    for (h, w, c, seed) in [(72u32, 128u32, 16u32, 61u32), (144, 256, 16, 62)] {
        let desc = TensorDesc::new(h, w, c, dt);
        let spec = ElementwiseSpec {
            op: BinaryOp::Add,
            operand: EwOperand::Tensor,
            act: Activation::None,
            len_vec4: desc.vec4_len() as u32,
            dt,
        };
        let bufs = [
            filled(ctx, &desc, seed),
            filled(ctx, &desc, seed + 1),
            storage_out(ctx, desc.size_bytes()),
        ];
        let mut params = [0u8; 16];
        params[4..8].copy_from_slice(&desc.cg().to_le_bytes());
        params[12..16].copy_from_slice(&(desc.vec4_len() as u32).to_le_bytes());
        let flops = (h * w * c) as f64;
        out.push(
            bench_kernel(ctx, &spec, &params, &bufs.iter().collect::<Vec<_>>(), flops).await?,
        );
    }

    Ok(out)
}
