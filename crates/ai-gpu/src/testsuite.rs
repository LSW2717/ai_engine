//! 공유 정확도 스위트 — 네이티브 `cargo test`와 wasm 데모가 **같은 케이스, 같은 시드**를
//! 실행한다. 브라우저에서 숫자가 다르면 그건 플랫폼 차이지 입력 차이가 아니다.

use ai_core::ops::BinaryOp;
use ai_core::reference;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};

use crate::context::GpuContext;
use crate::kernel::{self, KernelSpec};
use crate::kernels::elementwise::ElementwiseSpec;
use crate::readback;

pub struct CaseResult {
    pub name: String,
    pub passed: bool,
    pub max_err: f32,
    pub tol: f32,
}

pub const ATOL_F32: f32 = 1e-4;
pub const RTOL_F32: f32 = 1e-4;
pub const ATOL_F16: f32 = 1e-2;
pub const RTOL_F16: f32 = 1e-2;

pub fn tol_for(dt: DType) -> (f32, f32) {
    match dt {
        DType::F32 => (ATOL_F32, RTOL_F32),
        DType::F16 => (ATOL_F16, RTOL_F16),
    }
}

pub fn compare(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) -> CaseResult {
    let mut max_err = 0f32;
    let mut passed = got.len() == want.len();
    for (g, w) in got.iter().zip(want) {
        let err = (g - w).abs();
        max_err = max_err.max(err);
        if !(err <= atol + rtol * w.abs()) {
            passed = false;
        }
    }
    CaseResult { name: name.into(), passed, max_err, tol: atol }
}

// ---- GPU 헬퍼 (arena 도입 전까지의 직접 버퍼 경로; 벤치도 재사용) ----

pub fn storage_in(ctx: &GpuContext, bytes: &[u8]) -> wgpu::Buffer {
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buf, 0, bytes);
    buf
}

pub fn storage_out(ctx: &GpuContext, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// 256B 슬롯 하나짜리 params 버퍼
pub(crate) fn params_buffer(ctx: &GpuContext, data: &[u8]) -> wgpu::Buffer {
    assert!(data.len() <= 256);
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buf, 0, data);
    buf
}

/// 단일 커널 실행: buffers는 spec.bindings() 순서(입력들…출력), 마지막이 출력.
pub(crate) async fn run_single(
    ctx: &GpuContext,
    spec: &dyn KernelSpec,
    params: &[u8],
    buffers: &[&wgpu::Buffer],
    out_size: u64,
) -> Result<Vec<u8>, String> {
    let compiled = kernel::compile(ctx, spec).await?;
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

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&compiled.pipeline);
        pass.set_bind_group(0, &bind_group, &[0]);
        let g = spec.workgroups();
        pass.dispatch_workgroups(g[0], g[1], g[2]);
    }
    let out_buf = buffers.last().unwrap();
    enc.copy_buffer_to_buffer(out_buf, 0, &staging, 0, out_size);
    ctx.queue.submit([enc.finish()]);

    Ok(readback::read_buffers(ctx, &[&staging]).await?.remove(0))
}

// ---- elementwise 케이스 ----

fn ew_params(scalar: f32, cg: u32, len_vec4: u32) -> [u8; 16] {
    let mut p = [0u8; 16];
    p[0..4].copy_from_slice(&scalar.to_le_bytes());
    p[4..8].copy_from_slice(&cg.to_le_bytes());
    p[12..16].copy_from_slice(&len_vec4.to_le_bytes());
    p
}

pub async fn run_elementwise(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    use crate::kernels::elementwise::EwOperand;

    let mut results = Vec::new();
    let acts = [Activation::None, Activation::Relu, Activation::Sigmoid, Activation::Hardswish];
    let ops = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Prelu];
    let operands = [
        EwOperand::Tensor,
        EwOperand::Scalar { scalar_first: false },
        EwOperand::Scalar { scalar_first: true },
    ];
    // W 홀수 + C%4≠0 (구 엔진 제약 회귀 검증)
    let small = TensorDesc::new(3, 5, 6, DType::F32);
    // RVM급 큰 텐서 1개
    let large = TensorDesc::new(72, 128, 16, DType::F32);

    let mut seed = 100u32;
    for op in ops {
        for operand in operands {
            for act in acts {
                seed += 1;
                results.push(ew_case(ctx, op, operand, act, &small, seed).await?);
            }
        }
    }
    results
        .push(ew_case(ctx, BinaryOp::Add, EwOperand::Tensor, Activation::Relu, &large, 999).await?);
    // 단항 (act 단독 op의 lowering 대상)
    results.push(
        ew_case(ctx, BinaryOp::Add, EwOperand::Unary, Activation::Clamp01, &small, 1003).await?,
    );
    // SE 채널 브로드캐스트 (tensor × [1,1,C] 벡터)
    results.push(
        ew_case(ctx, BinaryOp::Mul, EwOperand::ChannelVector, Activation::None, &small, 1001)
            .await?,
    );
    results.push(
        ew_case(ctx, BinaryOp::Mul, EwOperand::ChannelVector, Activation::None, &large, 1002)
            .await?,
    );
    // GRU mix (a + z·(b-a)) — 홀수/큰 텐서 양쪽
    results.push(ew_case(ctx, BinaryOp::Add, EwOperand::Mix, Activation::None, &small, 1004).await?);
    results.push(ew_case(ctx, BinaryOp::Add, EwOperand::Mix, Activation::None, &large, 1005).await?);
    Ok(results)
}

async fn ew_case(
    ctx: &GpuContext,
    op: BinaryOp,
    operand: crate::kernels::elementwise::EwOperand,
    act: Activation,
    desc: &TensorDesc,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::elementwise::EwOperand;

    let mut rng = XorShift32::new(seed);
    let a = rng.vec_f32(desc.elems());
    let scalar_val = 0.7f32;

    let spec =
        ElementwiseSpec { op, operand, act, len_vec4: desc.vec4_len() as u32, dt: desc.dt,
            views: [crate::kernels::common::source::SrcView::NONE; 3],
            out_cg: 0,
        };
    let name = format!("{} {}x{}x{}", spec.cache_key(&ctx.caps), desc.h, desc.w, desc.c);

    let a_buf = storage_in(ctx, &pack::pack_nhwc(&a, desc));
    let out = storage_out(ctx, desc.size_bytes());
    let params = ew_params(scalar_val, desc.cg(), spec.len_vec4);

    let (want, got_bytes) = match operand {
        EwOperand::Scalar { scalar_first } => {
            let want = reference::elementwise::binary_scalar(op, &a, scalar_val, scalar_first, act);
            let bytes = run_single(ctx, &spec, &params, &[&a_buf, &out], desc.size_bytes()).await?;
            (want, bytes)
        }
        EwOperand::Tensor => {
            let b = rng.vec_f32(desc.elems());
            let want = reference::elementwise::binary(op, &a, &b, act);
            let b_buf = storage_in(ctx, &pack::pack_nhwc(&b, desc));
            let bytes =
                run_single(ctx, &spec, &params, &[&a_buf, &b_buf, &out], desc.size_bytes()).await?;
            (want, bytes)
        }
        EwOperand::Unary => {
            let want: Vec<f32> = a.iter().map(|v| act.apply(*v)).collect();
            let bytes = run_single(ctx, &spec, &params, &[&a_buf, &out], desc.size_bytes()).await?;
            (want, bytes)
        }
        EwOperand::ChannelVector => {
            // B = [1,1,C] 채널 벡터, 브로드캐스트 곱/합
            let bvec = rng.vec_f32(desc.c as usize);
            let mut want = vec![0f32; desc.elems()];
            for px in 0..(desc.h * desc.w) as usize {
                for ch in 0..desc.c as usize {
                    let v = op.apply(a[px * desc.c as usize + ch], bvec[ch]);
                    want[px * desc.c as usize + ch] = act.apply(v);
                }
            }
            let bdesc = TensorDesc::new(1, 1, desc.c, desc.dt);
            let b_buf = storage_in(ctx, &pack::pack_nhwc(&bvec, &bdesc));
            let bytes =
                run_single(ctx, &spec, &params, &[&a_buf, &b_buf, &out], desc.size_bytes()).await?;
            (want, bytes)
        }
        EwOperand::Mix => {
            // out = (1-z)·a + z·b (GRU 갱신)
            let b = rng.vec_f32(desc.elems());
            let z = rng.vec_f32(desc.elems());
            let want: Vec<f32> = (0..desc.elems())
                .map(|i| act.apply(a[i] + z[i] * (b[i] - a[i])))
                .collect();
            let b_buf = storage_in(ctx, &pack::pack_nhwc(&b, desc));
            let z_buf = storage_in(ctx, &pack::pack_nhwc(&z, desc));
            let bytes = run_single(
                ctx,
                &spec,
                &params,
                &[&a_buf, &b_buf, &z_buf, &out],
                desc.size_bytes(),
            )
            .await?;
            (want, bytes)
        }
    };

    let got = pack::unpack_nhwc(&got_bytes, desc);
    Ok(compare(&name, &got, &want, ATOL_F32, RTOL_F32))
}

// ---- pointwise GEMM 케이스 ----

#[allow(clippy::too_many_arguments)]
async fn pw_case(
    ctx: &GpuContext,
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    act: Activation,
    residual: bool,
    dt: DType,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::gemm_pw::GemmPwSpec;
    use ai_core::ops::Conv2d;

    let din = TensorDesc::new(h, w, cin, dt);
    let dout = TensorDesc::new(h, w, cout, dt);
    let mut rng = XorShift32::new(seed);
    let input = rng.vec_f32(din.elems());
    let wts = rng.vec_f32((cout * cin) as usize);
    let bias = rng.vec_f32(cout as usize);
    let res = residual.then(|| rng.vec_f32(dout.elems()));

    let op = Conv2d::pointwise(cin, cout, act);
    let want = reference::conv::conv2d(&op, h, w, &input, &wts, Some(&bias), res.as_deref());

    let spec = GemmPwSpec { m: h * w, kg: din.cg(), ng: dout.cg(), act, residual, dt, wdt: dt };
    let name = format!("{} ({}x{} {}->{})", spec.cache_key(&ctx.caps), h, w, cin, cout);

    let (wbytes, _kg_pad) = pack::pack_weights_conv(&wts, cout, cin, 1, 1, 4, dt);
    let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
    let w_buf = storage_in(ctx, &wbytes);
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, cout, dt));
    let out_buf = storage_out(ctx, dout.size_bytes());

    let got_bytes = if let Some(r) = &res {
        let r_buf = storage_in(ctx, &pack::pack_nhwc(r, &dout));
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &r_buf, &out_buf], dout.size_bytes())
            .await?
    } else {
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &out_buf], dout.size_bytes())
            .await?
    };

    let got = pack::unpack_nhwc(&got_bytes, &dout);
    let (atol, rtol) = tol_for(dt);
    Ok(compare(&name, &got, &want, atol, rtol))
}

pub async fn run_gemm_pw(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    let mut results = Vec::new();
    let acts = [Activation::None, Activation::Relu, Activation::Hardswish, Activation::Sigmoid];
    // (h, w, cin, cout): small M=1 / M=144 / 홀수 채널·W, tiled 정합·비정합 타일
    let shapes = [
        (1u32, 1u32, 64u32, 16u32),  // SE 극단 (M=1)
        (9, 16, 128, 24),            // 심층 저해상도 (M=144)
        (3, 5, 6, 10),               // W 홀수 + C%4≠0
        (72, 128, 16, 64),           // tiled 와이드 M
        (24, 32, 6, 10),             // tiled + 홀수 채널
        (23, 33, 8, 20),             // tiled + M%32≠0, NG%8≠0
        (9, 16, 960, 160),           // splitk 실 RVM shape (SPLIT=8)
        (9, 16, 240, 68),            // splitk 셀 비배수 에지 (2448 % 32 ≠ 0)
        (11, 13, 72, 68),            // splitk 홀수 M·NG 에지
    ];
    let mut seed = 2000;
    for (h, w, cin, cout) in shapes {
        for act in acts {
            seed += 1;
            results.push(pw_case(ctx, h, w, cin, cout, act, false, DType::F32, seed).await?);
        }
    }
    // residual 융합 (활성화 후 더하기 규약)
    results.push(pw_case(ctx, 9, 16, 32, 32, Activation::Relu, true, DType::F32, 3001).await?);
    // splitk + residual (RVM 960→160 잔차 블록)
    results.push(pw_case(ctx, 9, 16, 960, 160, Activation::Relu, true, DType::F32, 3003).await?);
    results
        .push(pw_case(ctx, 72, 128, 16, 16, Activation::Hardswish, true, DType::F32, 3002).await?);
    // f16 스토리지 변형 (지원 기기에서만)
    if ctx.caps.f16 {
        results
            .push(pw_case(ctx, 72, 128, 16, 64, Activation::Relu, false, DType::F16, 3101).await?);
        results.push(
            pw_case(ctx, 9, 16, 128, 24, Activation::Hardswish, false, DType::F16, 3102).await?,
        );
        results.push(pw_case(ctx, 24, 32, 16, 16, Activation::Relu, true, DType::F16, 3103).await?);
    }
    Ok(results)
}

// ---- depthwise conv 케이스 ----

#[allow(clippy::too_many_arguments)]
async fn dw_case(
    ctx: &GpuContext,
    ih: u32,
    iw: u32,
    c: u32,
    k: u32,
    s: u32,
    act: Activation,
    residual: bool,
    dt: DType,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::conv_dw::ConvDwSpec;
    use ai_core::ops::Conv2d;

    let op = Conv2d::depthwise(c, k, s, act);
    let (oh, ow) = op.out_hw(ih, iw);
    let din = TensorDesc::new(ih, iw, c, dt);
    let dout = TensorDesc::new(oh, ow, c, dt);

    let mut rng = XorShift32::new(seed);
    let input = rng.vec_f32(din.elems());
    let wts = rng.vec_f32((c * k * k) as usize);
    let bias = rng.vec_f32(c as usize);
    let res = residual.then(|| rng.vec_f32(dout.elems()));

    let want = reference::conv::conv2d(&op, ih, iw, &input, &wts, Some(&bias), res.as_deref());

    let spec = ConvDwSpec::from_op(&op, ih, iw, residual, dt);
    let name = spec.cache_key(&ctx.caps);

    let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
    let w_buf = storage_in(ctx, &pack::pack_weights_dw(&wts, c, k, k, dt));
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, c, dt));
    let out_buf = storage_out(ctx, dout.size_bytes());

    let got_bytes = if let Some(r) = &res {
        let r_buf = storage_in(ctx, &pack::pack_nhwc(r, &dout));
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &r_buf, &out_buf], dout.size_bytes())
            .await?
    } else {
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &out_buf], dout.size_bytes())
            .await?
    };

    let got = pack::unpack_nhwc(&got_bytes, &dout);
    let (atol, rtol) = tol_for(dt);
    Ok(compare(&name, &got, &want, atol, rtol))
}

pub async fn run_conv_dw(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    let mut results = Vec::new();
    let acts = [Activation::None, Activation::Relu, Activation::Hardswish];
    // (ih, iw, c, k, s): RVM 대표 shape + 홀수/에지
    let shapes = [
        (72u32, 128u32, 64u32, 3u32, 1u32), // 대공간 dw
        (36, 64, 96, 5, 2),                 // MNv3 5×5 s2
        (33, 17, 6, 3, 2),                  // 홀수 W/H + C%4≠0
        (16, 16, 10, 5, 1),                 // k5 s1
    ];
    let mut seed = 4000;
    for (ih, iw, c, k, s) in shapes {
        for act in acts {
            seed += 1;
            results.push(dw_case(ctx, ih, iw, c, k, s, act, false, DType::F32, seed).await?);
        }
    }
    results.push(dw_case(ctx, 36, 64, 32, 3, 1, Activation::Relu, true, DType::F32, 5001).await?);
    // dilation d=2 (RVM 인코더 마지막 스테이지) — dw + igemm
    results.push(dilated_dw_case(ctx, 18, 32, 40, 5201).await?);
    results.push(dilated_igemm_case(ctx, 24, 32, 8, 12, 5202).await?);
    // f16 스토리지 변형
    if ctx.caps.f16 {
        results
            .push(dw_case(ctx, 72, 128, 64, 3, 1, Activation::Relu, false, DType::F16, 5101).await?);
        results
            .push(dw_case(ctx, 36, 64, 96, 5, 2, Activation::None, false, DType::F16, 5102).await?);
    }
    Ok(results)
}

/// dilation=2 depthwise (RVM 인코더 마지막 스테이지 형태)
async fn dilated_dw_case(
    ctx: &GpuContext,
    ih: u32,
    iw: u32,
    c: u32,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::conv_dw::ConvDwSpec;
    use ai_core::ops::Conv2d;

    let op = Conv2d {
        cin: c,
        cout: c,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        pad: [2; 4], // d=2 k3의 same pad
        dil: 2,
        groups: c,
        act: Activation::Relu,
    };
    let (oh, ow) = op.out_hw(ih, iw);
    let din = TensorDesc::new(ih, iw, c, DType::F32);
    let dout = TensorDesc::new(oh, ow, c, DType::F32);
    let mut rng = XorShift32::new(seed);
    let input = rng.vec_f32(din.elems());
    let wts = rng.vec_f32((c * 9) as usize);
    let bias = rng.vec_f32(c as usize);
    let want = reference::conv::conv2d(&op, ih, iw, &input, &wts, Some(&bias), None);
    let spec = ConvDwSpec::from_op(&op, ih, iw, false, DType::F32);
    let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
    let w_buf = storage_in(ctx, &pack::pack_weights_dw(&wts, c, 3, 3, DType::F32));
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, c, DType::F32));
    let out_buf = storage_out(ctx, dout.size_bytes());
    let bytes =
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &out_buf], dout.size_bytes())
            .await?;
    let got = pack::unpack_nhwc(&bytes, &dout);
    Ok(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32))
}

/// dilation=2 일반 conv
async fn dilated_igemm_case(
    ctx: &GpuContext,
    ih: u32,
    iw: u32,
    cin: u32,
    cout: u32,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::conv_igemm::ConvIgemmSpec;
    use ai_core::ops::Conv2d;

    let op = Conv2d {
        cin,
        cout,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        pad: [2; 4],
        dil: 2,
        groups: 1,
        act: Activation::Hardswish,
    };
    let (oh, ow) = op.out_hw(ih, iw);
    let din = TensorDesc::new(ih, iw, cin, DType::F32);
    let dout = TensorDesc::new(oh, ow, cout, DType::F32);
    let mut rng = XorShift32::new(seed);
    let input = rng.vec_f32(din.elems());
    let wts = rng.vec_f32((cout * cin * 9) as usize);
    let bias = rng.vec_f32(cout as usize);
    let want = reference::conv::conv2d(&op, ih, iw, &input, &wts, Some(&bias), None);
    let spec = ConvIgemmSpec::from_op(&op, ih, iw, false, DType::F32);
    let (wbytes, _) = pack::pack_weights_conv(&wts, cout, cin, 3, 3, 4, DType::F32);
    let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
    let w_buf = storage_in(ctx, &wbytes);
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, cout, DType::F32));
    let out_buf = storage_out(ctx, dout.size_bytes());
    let bytes =
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &out_buf], dout.size_bytes())
            .await?;
    let got = pack::unpack_nhwc(&bytes, &dout);
    Ok(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32))
}

// ---- 일반 conv (implicit GEMM) 케이스 ----

#[allow(clippy::too_many_arguments)]
async fn igemm_case(
    ctx: &GpuContext,
    ih: u32,
    iw: u32,
    cin: u32,
    cout: u32,
    k: u32,
    s: u32,
    pad: [u32; 4],
    act: Activation,
    residual: bool,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::conv_igemm::ConvIgemmSpec;
    use ai_core::ops::Conv2d;

    let op = Conv2d { cin, cout, kh: k, kw: k, sh: s, sw: s, pad, dil: 1, groups: 1, act };
    let (oh, ow) = op.out_hw(ih, iw);
    let din = TensorDesc::new(ih, iw, cin, DType::F32);
    let dout = TensorDesc::new(oh, ow, cout, DType::F32);

    let mut rng = XorShift32::new(seed);
    let input = rng.vec_f32(din.elems());
    let wts = rng.vec_f32((cout * cin * k * k) as usize);
    let bias = rng.vec_f32(cout as usize);
    let res = residual.then(|| rng.vec_f32(dout.elems()));

    let want = reference::conv::conv2d(&op, ih, iw, &input, &wts, Some(&bias), res.as_deref());

    let spec = ConvIgemmSpec::from_op(&op, ih, iw, residual, DType::F32);
    let name = spec.cache_key(&ctx.caps);

    let (wbytes, _) = pack::pack_weights_conv(&wts, cout, cin, k, k, 4, DType::F32);
    let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
    let w_buf = storage_in(ctx, &wbytes);
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, cout, DType::F32));
    let out_buf = storage_out(ctx, dout.size_bytes());

    let got_bytes = if let Some(r) = &res {
        let r_buf = storage_in(ctx, &pack::pack_nhwc(r, &dout));
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &r_buf, &out_buf], dout.size_bytes())
            .await?
    } else {
        run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &w_buf, &b_buf, &out_buf], dout.size_bytes())
            .await?
    };

    let got = pack::unpack_nhwc(&got_bytes, &dout);
    Ok(compare(&name, &got, &want, ATOL_F32, RTOL_F32))
}

/// concat-into-conv 융합 케이스 — 파트 텐서들을 개별 버퍼로 주고, CPU 레퍼런스는
/// 채널 concat 후 일반 conv. 파트 시작은 4채널 정렬이어야 한다 (변환기 보장 규약).
async fn igemm_concat_case(
    ctx: &GpuContext,
    ih: u32,
    iw: u32,
    part_cs: &[u32],
    cout: u32,
    k: u32,
    act: Activation,
    residual: bool,
    seed: u32,
) -> Result<CaseResult, String> {
    use crate::kernels::conv_igemm::ConvIgemmSpec;
    use ai_core::ops::Conv2d;

    let cin: u32 = part_cs.iter().sum();
    let p = (k - 1) / 2;
    let op = Conv2d {
        cin,
        cout,
        kh: k,
        kw: k,
        sh: 1,
        sw: 1,
        pad: [p; 4],
        dil: 1,
        groups: 1,
        act,
    };
    let (oh, ow) = op.out_hw(ih, iw);
    let dout = TensorDesc::new(oh, ow, cout, DType::F32);

    let mut rng = XorShift32::new(seed);
    // 파트별 데이터 + CPU 쪽은 채널 concat한 단일 입력
    let parts: Vec<Vec<f32>> =
        part_cs.iter().map(|c| rng.vec_f32((ih * iw * c) as usize)).collect();
    let px = (ih * iw) as usize;
    let mut cat = vec![0f32; px * cin as usize];
    for pxi in 0..px {
        let mut off = 0usize;
        for (pv, c) in parts.iter().zip(part_cs) {
            let c = *c as usize;
            cat[pxi * cin as usize + off..pxi * cin as usize + off + c]
                .copy_from_slice(&pv[pxi * c..(pxi + 1) * c]);
            off += c;
        }
    }
    let wts = rng.vec_f32((cout * cin * k * k) as usize);
    let bias = rng.vec_f32(cout as usize);
    let res = residual.then(|| rng.vec_f32(dout.elems()));
    let want = reference::conv::conv2d(&op, ih, iw, &cat, &wts, Some(&bias), res.as_deref());

    let mut spec = ConvIgemmSpec::from_op(&op, ih, iw, residual, DType::F32);
    for (i, c) in part_cs.iter().enumerate() {
        spec.srcs[i] = crate::kernels::common::source::SrcView::plain(*c);
    }
    let name = spec.cache_key(&ctx.caps);

    let (wbytes, _) = pack::pack_weights_conv(&wts, cout, cin, k, k, 4, DType::F32);
    let part_bufs: Vec<wgpu::Buffer> = parts
        .iter()
        .zip(part_cs)
        .map(|(pv, c)| storage_in(ctx, &pack::pack_nhwc(pv, &TensorDesc::new(ih, iw, *c, DType::F32))))
        .collect();
    let w_buf = storage_in(ctx, &wbytes);
    let b_buf = storage_in(ctx, &pack::pack_bias(&bias, cout, DType::F32));
    let out_buf = storage_out(ctx, dout.size_bytes());

    let mut bufs: Vec<&wgpu::Buffer> = part_bufs.iter().collect();
    bufs.push(&w_buf);
    bufs.push(&b_buf);
    let r_buf = res.as_ref().map(|r| storage_in(ctx, &pack::pack_nhwc(r, &dout)));
    if let Some(rb) = &r_buf {
        bufs.push(rb);
    }
    bufs.push(&out_buf);
    let got_bytes = run_single(ctx, &spec, &[0u8; 16], &bufs, dout.size_bytes()).await?;

    let got = pack::unpack_nhwc(&got_bytes, &dout);
    Ok(compare(&name, &got, &want, ATOL_F32, RTOL_F32))
}

pub async fn run_conv_igemm(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    let mut results = Vec::new();
    let acts = [Activation::None, Activation::Relu, Activation::Hardswish];
    // (ih, iw, cin, cout, k, s): 스템 direct, tiled 정합/홀수, direct 소-M, k5
    let shapes = [
        (36u32, 64u32, 3u32, 16u32, 3u32, 2u32), // 스템형 (direct, K 극소)
        (24, 32, 16, 24, 3, 1),                  // tiled
        (40, 64, 20, 32, 3, 1),                  // tiled + C%4≠0
        (17, 23, 6, 8, 3, 2),                    // direct 소-M + 홀수
        (33, 45, 8, 12, 5, 2),                   // k5 direct
        (9, 16, 128, 128, 3, 1),                 // splitk (RVM 심층 k3, 기아 shape)
        (18, 32, 171, 80, 3, 1),                 // splitk 대형 K + C%4≠0
    ];
    let mut seed = 6000;
    for (ih, iw, cin, cout, k, s) in shapes {
        let p = (k - 1) / 2;
        for act in acts {
            seed += 1;
            results
                .push(igemm_case(ctx, ih, iw, cin, cout, k, s, [p; 4], act, false, seed).await?);
        }
    }
    // 비대칭 pad + residual
    results.push(
        igemm_case(ctx, 16, 16, 4, 8, 3, 1, [1, 0, 0, 1], Activation::Relu, false, 7001).await?,
    );
    // 실제 스템 크기 (128² — Direct4 변형이 걸리는 크기)
    results.push(
        igemm_case(ctx, 128, 128, 3, 16, 3, 2, [0, 0, 1, 1], Activation::None, false, 7008)
            .await?,
    );
    // tf2onnx SAME 패딩 (s2 → 비대칭 [0,0,1,1]) — MediaPipe 스템이 실제로 이 모양
    results.push(
        igemm_case(ctx, 32, 32, 3, 16, 3, 2, [0, 0, 1, 1], Activation::None, false, 7005).await?,
    );
    results.push(
        igemm_case(ctx, 32, 32, 3, 24, 5, 2, [1, 1, 2, 2], Activation::None, false, 7006).await?,
    );
    results.push(
        igemm_case(ctx, 24, 24, 16, 16, 3, 2, [0, 0, 1, 1], Activation::Relu, false, 7007).await?,
    );
    results.push(
        igemm_case(ctx, 24, 32, 16, 16, 3, 1, [1; 4], Activation::Hardswish, true, 7002).await?,
    );
    results
        .push(igemm_case(ctx, 9, 16, 8, 8, 3, 1, [1; 4], Activation::None, true, 7003).await?);
    // splitk + residual (RVM 심층 잔차 블록)
    results.push(
        igemm_case(ctx, 9, 16, 128, 128, 3, 1, [1; 4], Activation::Relu, true, 7004).await?,
    );
    // concat-into-conv 융합: 2파트(Direct), 3파트 홀수 꼬리(Direct), 2파트(Splitk)
    results.push(
        igemm_concat_case(ctx, 24, 32, &[16, 16], 24, 3, Activation::Relu, false, 7101).await?,
    );
    results.push(
        igemm_concat_case(ctx, 36, 64, &[80, 24, 3], 40, 3, Activation::Hardswish, false, 7102)
            .await?,
    );
    results.push(
        igemm_concat_case(ctx, 9, 16, &[64, 64], 128, 3, Activation::Relu, true, 7103).await?,
    );
    Ok(results)
}

// ---- 풀링 / 리사이즈 케이스 ----

pub async fn run_pool_resize(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    use crate::kernels::avgpool::AvgPoolSpec;
    use crate::kernels::gpool::GpoolSpec;
    use crate::kernels::resize_bilinear::ResizeBilinearSpec;
    use ai_core::ops::{AvgPool2d, CoordMode, ResizeBilinear};

    let mut results = Vec::new();

    // gpool: RVM SE 대표 + 홀수
    for (h, w, c, seed) in [(9u32, 16u32, 576u32, 8001u32), (3, 5, 6, 8002), (72, 128, 16, 8003)]
    {
        let din = TensorDesc::new(h, w, c, DType::F32);
        let dout = TensorDesc::new(1, 1, c, DType::F32);
        let mut rng = XorShift32::new(seed);
        let input = rng.vec_f32(din.elems());
        let want = reference::pool::global_avg_pool(&input, h, w, c);
        let spec = GpoolSpec { h, w, c, dt: DType::F32 };
        let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
        let out_buf = storage_out(ctx, dout.size_bytes());
        let bytes =
            run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &out_buf], dout.size_bytes()).await?;
        let got = pack::unpack_nhwc(&bytes, &dout);
        results.push(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32));
    }

    // maxpool: MediaPipe k2 s2 + pad_c(채널패드 접기) + 비4배수 채널
    {
        use crate::kernels::maxpool::MaxPoolSpec;
        use ai_core::ops::MaxPool2d;
        for (ih, iw, c, k, s, pad_c, seed) in [
            (16u32, 16u32, 8u32, 2u32, 2u32, 0u32, 8201u32),
            (16, 16, 44, 2, 2, 4, 8202),
            (7, 7, 6, 2, 2, 2, 8203),
            (9, 9, 12, 3, 1, 0, 8204),
        ] {
            let op = MaxPool2d { kh: k, kw: k, sh: s, sw: s, pad: [0; 4], pad_c };
            let (oh, ow) = op.out_hw(ih, iw);
            let din = TensorDesc::new(ih, iw, c, DType::F32);
            let dout = TensorDesc::new(oh, ow, c + pad_c, DType::F32);
            let mut rng = XorShift32::new(seed);
            let input = rng.vec_f32(din.elems());
            let want = reference::pool::max_pool(&op, ih, iw, c, &input);
            let spec = MaxPoolSpec::from_op(&op, ih, iw, c, DType::F32);
            let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
            let out_buf = storage_out(ctx, dout.size_bytes());
            let bytes =
                run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &out_buf], dout.size_bytes())
                    .await?;
            let got = pack::unpack_nhwc(&bytes, &dout);
            results.push(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32));
        }
    }

    // avgpool: RVM k==s 경로 + 일반형
    for (ih, iw, c, k, s, seed) in
        [(36u32, 64u32, 40u32, 2u32, 2u32, 8101u32), (18, 32, 6, 3, 3, 8102), (16, 16, 8, 2, 1, 8103)]
    {
        let op = AvgPool2d { kh: k, kw: k, sh: s, sw: s, pad: [0; 4] };
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, c, DType::F32);
        let dout = TensorDesc::new(oh, ow, c, DType::F32);
        let mut rng = XorShift32::new(seed);
        let input = rng.vec_f32(din.elems());
        let want = reference::pool::avg_pool(&op, ih, iw, c, &input);
        let spec = AvgPoolSpec::from_op(&op, ih, iw, c, DType::F32);
        let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
        let out_buf = storage_out(ctx, dout.size_bytes());
        let bytes =
            run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &out_buf], dout.size_bytes()).await?;
        let got = pack::unpack_nhwc(&bytes, &dout);
        results.push(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32));
    }

    // resize: 2× 업샘플(RVM 디코더) 양 좌표 모드 + 비정수 배율
    for (ih, iw, c, oh, ow, mode, seed) in [
        (18u32, 32u32, 40u32, 36u32, 64u32, CoordMode::HalfPixel, 8201u32),
        (18, 32, 40, 36, 64, CoordMode::Asymmetric, 8202),
        (13, 17, 6, 20, 30, CoordMode::HalfPixel, 8203),
    ] {
        let rop = ResizeBilinear { oh, ow, mode };
        let din = TensorDesc::new(ih, iw, c, DType::F32);
        let dout = TensorDesc::new(oh, ow, c, DType::F32);
        let mut rng = XorShift32::new(seed);
        let input = rng.vec_f32(din.elems());
        let want = reference::resize::resize_bilinear(&rop, ih, iw, c, &input);
        let spec = ResizeBilinearSpec { ih, iw, c, oh, ow, mode, dt: DType::F32, srcs: [crate::kernels::common::source::SrcView::NONE; 3] };
        let in_buf = storage_in(ctx, &pack::pack_nhwc(&input, &din));
        let out_buf = storage_out(ctx, dout.size_bytes());
        let bytes =
            run_single(ctx, &spec, &[0u8; 16], &[&in_buf, &out_buf], dout.size_bytes()).await?;
        let got = pack::unpack_nhwc(&bytes, &dout);
        results.push(compare(&spec.cache_key(&ctx.caps), &got, &want, ATOL_F32, RTOL_F32));
    }

    Ok(results)
}

// ---- Phase 1 종료 시험: MNv3 inverted-residual + SE 블록 전체 ----

/// expand pw → dw k3 → SE(gpool → squeeze pw → excite pw → 채널스케일 mul)
/// → project pw(+residual)를 **버퍼 풀 + params 테이블 + 파이프라인 캐시 +
/// 단일 인코더·단일 컴퓨트 패스·단일 submit**으로 실행 — 실전 executor 경로 그 자체.
pub async fn run_mobilenet_block(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    use crate::arena::{Arena, ArenaPlanner};
    use crate::cache::PipelineCache;
    use crate::kernel::OpDispatch;
    use crate::kernels::conv_dw::ConvDwSpec;
    use crate::kernels::elementwise::EwOperand;
    use crate::kernels::gemm_pw::GemmPwSpec;
    use crate::kernels::gpool::GpoolSpec;
    use crate::params::ParamsTable;
    use ai_core::ops::Conv2d;

    let (h, w) = (24u32, 32u32);
    let (cin, cexp) = (24u32, 72u32);
    let mut rng = XorShift32::new(42);

    // 텐서 desc
    let d_in = TensorDesc::new(h, w, cin, DType::F32);
    let d_exp = TensorDesc::new(h, w, cexp, DType::F32);
    let d_vec_exp = TensorDesc::new(1, 1, cexp, DType::F32);
    let d_vec_sq = TensorDesc::new(1, 1, cin, DType::F32);

    // 가중치/입력
    let input = rng.vec_f32(d_in.elems());
    let w_expand = rng.vec_f32((cexp * cin) as usize);
    let b_expand = rng.vec_f32(cexp as usize);
    let w_dw = rng.vec_f32((cexp * 9) as usize);
    let b_dw = rng.vec_f32(cexp as usize);
    let w_sq = rng.vec_f32((cin * cexp) as usize);
    let b_sq = rng.vec_f32(cin as usize);
    let w_ex = rng.vec_f32((cexp * cin) as usize);
    let b_ex = rng.vec_f32(cexp as usize);
    let w_proj = rng.vec_f32((cin * cexp) as usize);
    let b_proj = rng.vec_f32(cin as usize);

    // ---- CPU 레퍼런스 합성 ----
    let op_expand = Conv2d::pointwise(cin, cexp, Activation::Hardswish);
    let r_exp = reference::conv::conv2d(&op_expand, h, w, &input, &w_expand, Some(&b_expand), None);
    let op_dw = Conv2d::depthwise(cexp, 3, 1, Activation::Hardswish);
    let r_dw = reference::conv::conv2d(&op_dw, h, w, &r_exp, &w_dw, Some(&b_dw), None);
    let r_gp = reference::pool::global_avg_pool(&r_dw, h, w, cexp);
    let op_sq = Conv2d::pointwise(cexp, cin, Activation::Relu);
    let r_sq = reference::conv::conv2d(&op_sq, 1, 1, &r_gp, &w_sq, Some(&b_sq), None);
    let op_ex = Conv2d::pointwise(cin, cexp, Activation::Hardsigmoid);
    let r_ex = reference::conv::conv2d(&op_ex, 1, 1, &r_sq, &w_ex, Some(&b_ex), None);
    let mut r_scaled = vec![0f32; r_dw.len()];
    for px in 0..(h * w) as usize {
        for ch in 0..cexp as usize {
            r_scaled[px * cexp as usize + ch] = r_dw[px * cexp as usize + ch] * r_ex[ch];
        }
    }
    let op_proj = Conv2d::pointwise(cexp, cin, Activation::None);
    let want =
        reference::conv::conv2d(&op_proj, h, w, &r_scaled, &w_proj, Some(&b_proj), Some(&input));

    // ---- GPU: 버퍼 풀 + 단일 패스 ----
    let mut planner = ArenaPlanner::new();
    let v_in = planner.alloc(d_in);
    let v_exp = planner.alloc(d_exp);
    let v_dw = planner.alloc(d_exp);
    let v_gp = planner.alloc(d_vec_exp);
    let v_sq = planner.alloc(d_vec_sq);
    let v_ex = planner.alloc(d_vec_exp);
    let v_scaled = planner.alloc(d_exp);
    let v_out = planner.alloc(d_in);
    let arena = Arena::create(ctx, &planner)?;
    arena.upload(ctx, &v_in, &pack::pack_nhwc(&input, &d_in));

    // 가중치 버퍼 (읽기 전용이라 단일 버퍼+오프셋도 가능하지만 여기선 개별로)
    let wb = |w: &[f32], cout: u32, cin: u32| {
        let (bytes, _) = pack::pack_weights_conv(w, cout, cin, 1, 1, 4, DType::F32);
        storage_in(ctx, &bytes)
    };
    let bb = |b: &[f32], c: u32| storage_in(ctx, &pack::pack_bias(b, c, DType::F32));
    let buf_w_expand = wb(&w_expand, cexp, cin);
    let buf_b_expand = bb(&b_expand, cexp);
    let buf_w_dw = storage_in(ctx, &pack::pack_weights_dw(&w_dw, cexp, 3, 3, DType::F32));
    let buf_b_dw = bb(&b_dw, cexp);
    let buf_w_sq = wb(&w_sq, cin, cexp);
    let buf_b_sq = bb(&b_sq, cin);
    let buf_w_ex = wb(&w_ex, cexp, cin);
    let buf_b_ex = bb(&b_ex, cexp);
    let buf_w_proj = wb(&w_proj, cin, cexp);
    let buf_b_proj = bb(&b_proj, cin);

    // spec들
    let m = h * w;
    let s_expand = GemmPwSpec {
        m,
        kg: d_in.cg(),
        ng: d_exp.cg(),
        act: Activation::Hardswish,
        residual: false,
        dt: DType::F32, wdt: DType::F32 };
    let s_dw = ConvDwSpec::from_op(&op_dw, h, w, false, DType::F32);
    let s_gp = GpoolSpec { h, w, c: cexp, dt: DType::F32 };
    let s_sq = GemmPwSpec {
        m: 1,
        kg: d_vec_exp.cg(),
        ng: d_vec_sq.cg(),
        act: Activation::Relu,
        residual: false,
        dt: DType::F32, wdt: DType::F32 };
    let s_ex = GemmPwSpec {
        m: 1,
        kg: d_vec_sq.cg(),
        ng: d_vec_exp.cg(),
        act: Activation::Hardsigmoid,
        residual: false,
        dt: DType::F32, wdt: DType::F32 };
    let s_scale = ElementwiseSpec {
        op: BinaryOp::Mul,
        operand: EwOperand::ChannelVector,
        act: Activation::None,
        len_vec4: d_exp.vec4_len() as u32,
        dt: DType::F32,
            views: [crate::kernels::common::source::SrcView::NONE; 3],
            out_cg: 0,
        };
    let s_proj = GemmPwSpec {
        m,
        kg: d_exp.cg(),
        ng: d_in.cg(),
        act: Activation::None,
        residual: true,
        dt: DType::F32, wdt: DType::F32 };

    // 파이프라인 캐시 경유 컴파일 (동일 시그니처 공유 확인 겸)
    let mut cache = PipelineCache::new();
    let k_expand = cache.get_or_compile(ctx, &s_expand).await?;
    let k_dw = cache.get_or_compile(ctx, &s_dw).await?;
    let k_gp = cache.get_or_compile(ctx, &s_gp).await?;
    let k_sq = cache.get_or_compile(ctx, &s_sq).await?;
    let k_ex = cache.get_or_compile(ctx, &s_ex).await?;
    let k_scale = cache.get_or_compile(ctx, &s_scale).await?;
    let k_proj = cache.get_or_compile(ctx, &s_proj).await?;

    let params = ParamsTable::create(ctx, 1);
    params.write(ctx, 0, &ew_params(0.0, d_exp.cg(), s_scale.len_vec4));

    let bg = |kernel: &crate::kernel::CompiledKernel, bufs: &[wgpu::BindingResource]| {
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params.binding() }];
        for (i, r) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: (i + 1) as u32, resource: r.clone() });
        }
        std::sync::Arc::new(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.bgl,
            entries: &entries,
        }))
    };

    let ops = [
        OpDispatch {
            bind_group: bg(
                &k_expand,
                &[
                    arena.binding(&v_in),
                    buf_w_expand.as_entire_binding(),
                    buf_b_expand.as_entire_binding(),
                    arena.binding(&v_exp),
                ],
            ),
            groups: s_expand.workgroups(),
            kernel: k_expand,
            param_offset: 0,
            label: "expand pw".into(),
        },
        OpDispatch {
            bind_group: bg(
                &k_dw,
                &[
                    arena.binding(&v_exp),
                    buf_w_dw.as_entire_binding(),
                    buf_b_dw.as_entire_binding(),
                    arena.binding(&v_dw),
                ],
            ),
            groups: s_dw.workgroups(),
            kernel: k_dw,
            param_offset: 0,
            label: "dw k3".into(),
        },
        OpDispatch {
            bind_group: bg(&k_gp, &[arena.binding(&v_dw), arena.binding(&v_gp)]),
            groups: s_gp.workgroups(),
            kernel: k_gp,
            param_offset: 0,
            label: "gpool".into(),
        },
        OpDispatch {
            bind_group: bg(
                &k_sq,
                &[
                    arena.binding(&v_gp),
                    buf_w_sq.as_entire_binding(),
                    buf_b_sq.as_entire_binding(),
                    arena.binding(&v_sq),
                ],
            ),
            groups: s_sq.workgroups(),
            kernel: k_sq,
            param_offset: 0,
            label: "se squeeze".into(),
        },
        OpDispatch {
            bind_group: bg(
                &k_ex,
                &[
                    arena.binding(&v_sq),
                    buf_w_ex.as_entire_binding(),
                    buf_b_ex.as_entire_binding(),
                    arena.binding(&v_ex),
                ],
            ),
            groups: s_ex.workgroups(),
            kernel: k_ex,
            param_offset: 0,
            label: "se excite".into(),
        },
        OpDispatch {
            bind_group: bg(
                &k_scale,
                &[arena.binding(&v_dw), arena.binding(&v_ex), arena.binding(&v_scaled)],
            ),
            groups: s_scale.workgroups(),
            kernel: k_scale,
            param_offset: 0,
            label: "se scale".into(),
        },
        OpDispatch {
            bind_group: bg(
                &k_proj,
                &[
                    arena.binding(&v_scaled),
                    buf_w_proj.as_entire_binding(),
                    buf_b_proj.as_entire_binding(),
                    arena.binding(&v_in), // residual = 블록 입력
                    arena.binding(&v_out),
                ],
            ),
            groups: s_proj.workgroups(),
            kernel: k_proj,
            param_offset: 0,
            label: "project pw + res".into(),
        },
    ];

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: d_in.size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    kernel::record(&mut enc, &ops);
    enc.copy_buffer_to_buffer(arena.buffer(&v_out), 0, &staging, 0, d_in.size_bytes());
    ctx.queue.submit([enc.finish()]);

    let bytes = readback::read_buffers(ctx, &[&staging]).await?.remove(0);
    let got = pack::unpack_nhwc(&bytes, &d_in);
    Ok(vec![compare(
        "mnv3-block expand.hswish->dw3.hswish->SE->proj+res (24x32 c24/72)",
        &got,
        &want,
        ATOL_F32,
        RTOL_F32,
    )])
}

// ---- arena 실전 경로: 사전 생성 bind group + OpDispatch + record ----

/// 2-op 체인(t = relu(a+b); o = t*a)을 단일 arena·단일 패스로 실행.
/// 중간 텐서(vt)를 다음 op이 소비 — 패스 내 자동 해저드 배리어와
/// 같은 버퍼의 비중첩 다중 바인딩을 함께 검증한다.
pub async fn run_elementwise_arena(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    use crate::arena::{Arena, ArenaPlanner, TensorView};
    use crate::cache::PipelineCache;
    use crate::kernel::{CompiledKernel, OpDispatch};
    use crate::params::ParamsTable;
    use std::sync::Arc;

    let desc = TensorDesc::new(5, 7, 10, DType::F32); // W 홀수, C%4≠0
    let mut rng = XorShift32::new(777);
    let a = rng.vec_f32(desc.elems());
    let b = rng.vec_f32(desc.elems());

    let mut planner = ArenaPlanner::new();
    let va = planner.alloc(desc);
    let vb = planner.alloc(desc);
    let vt = planner.alloc(desc);
    let vo = planner.alloc(desc);
    let arena = Arena::create(ctx, &planner)?;
    arena.upload(ctx, &va, &pack::pack_nhwc(&a, &desc));
    arena.upload(ctx, &vb, &pack::pack_nhwc(&b, &desc));

    let params = ParamsTable::create(ctx, 2);
    let len = desc.vec4_len() as u32;
    params.write(ctx, 0, &ew_params(0.0, desc.cg(), len));
    params.write(ctx, 1, &ew_params(0.0, desc.cg(), len));

    let spec1 = ElementwiseSpec {
        op: BinaryOp::Add,
        operand: crate::kernels::elementwise::EwOperand::Tensor,
        act: Activation::Relu,
        len_vec4: len,
        dt: desc.dt,
            views: [crate::kernels::common::source::SrcView::NONE; 3],
            out_cg: 0,
        };
    let spec2 = ElementwiseSpec { op: BinaryOp::Mul, act: Activation::None, ..spec1 };

    let mut cache = PipelineCache::new();
    let k1 = cache.get_or_compile(ctx, &spec1).await?;
    let k2 = cache.get_or_compile(ctx, &spec2).await?;

    let bind = |k: &Arc<CompiledKernel>, x: &TensorView, y: &TensorView, o: &TensorView| {
        Arc::new(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &k.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.binding() },
                wgpu::BindGroupEntry { binding: 1, resource: arena.binding(x) },
                wgpu::BindGroupEntry { binding: 2, resource: arena.binding(y) },
                wgpu::BindGroupEntry { binding: 3, resource: arena.binding(o) },
            ],
        }))
    };

    let ops = [
        OpDispatch {
            bind_group: bind(&k1, &va, &vb, &vt),
            kernel: k1,
            groups: spec1.workgroups(),
            param_offset: ParamsTable::offset(0),
            label: "ew add relu".into(),
        },
        OpDispatch {
            bind_group: bind(&k2, &vt, &va, &vo),
            kernel: k2,
            groups: spec2.workgroups(),
            param_offset: ParamsTable::offset(1),
            label: "ew mul".into(),
        },
    ];

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: desc.size_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    kernel::record(&mut enc, &ops);
    enc.copy_buffer_to_buffer(arena.buffer(&vo), 0, &staging, 0, desc.size_bytes());
    ctx.queue.submit([enc.finish()]);

    let bytes = readback::read_buffers(ctx, &[&staging]).await?.remove(0);
    let got = pack::unpack_nhwc(&bytes, &desc);
    let t = reference::elementwise::binary(BinaryOp::Add, &a, &b, Activation::Relu);
    let want = reference::elementwise::binary(BinaryOp::Mul, &t, &a, Activation::None);
    Ok(vec![compare("ew arena-chain add.relu->mul 5x7x10", &got, &want, ATOL_F32, RTOL_F32)])
}

/// 전체 스위트 (커널이 늘 때마다 여기에 등록)
pub async fn run_all(ctx: &GpuContext) -> Result<Vec<CaseResult>, String> {
    let mut out = Vec::new();
    out.extend(run_elementwise(ctx).await?);
    out.extend(run_elementwise_arena(ctx).await?);
    out.extend(run_gemm_pw(ctx).await?);
    out.extend(run_conv_dw(ctx).await?);
    out.extend(run_conv_igemm(ctx).await?);
    out.extend(run_pool_resize(ctx).await?);
    out.extend(run_mobilenet_block(ctx).await?);
    Ok(out)
}
