//! 활성화별 파이프라인 컴파일 비용 (진단용, #[ignore])
//!
//! RVM 콜드 로드 실측에서 conv_igemm 14개 중 sigmoid/tanh 8개가 1350ms/1543ms를
//! 먹었다 — 같은 크기 셰이더의 relu판은 6ms. shape을 고정하고 활성화만 바꿔
//! 그 차이가 정말 활성화 때문인지 분리한다.

use ai_core::{Activation, DType};
use ai_gpu::kernels::common::activation::ALL;
use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
use ai_gpu::kernels::gemm_pw::GemmPwSpec;
use ai_gpu::kernel::KernelSpec;
use ai_gpu::GpuContext;

#[test]
#[ignore]
fn diag_act_compile() {
    let ctx = GpuContext::new_blocking().unwrap();
    println!("adapter: {}\n", ctx.caps.info.name);

    // RVM GRU 게이트와 같은 shape (9×16 128->128 k3, Splitk)
    let base = ConvIgemmSpec {
        ih: 9,
        iw: 16,
        cin: 128,
        cout: 128,
        k: 3,
        s: 1,
        d: 1,
        pad: [1; 4],
        act: Activation::Relu,
        residual: false,
        dt: DType::F32,
        wdt: DType::F32,
        force_variant: None,
        force_geom: None,
        srcs: [ai_gpu::kernels::common::source::SrcView::NONE; 3],
    };
    println!("== conv_igemm 9x16 128->128 k3 ({:?}) ==", base.variant());
    for (i, act) in ALL.into_iter().enumerate() {
        let spec = ConvIgemmSpec { act, ..base };
        let (ms, len) = cold_compile(&ctx, &spec, i as u32);
        println!("{:>12}: {ms:7.1}ms  ({len}B)", act.tag());
    }

    println!("\n== gemm_pw M144 KG32 NG32 ==");
    for (i, act) in ALL.into_iter().enumerate() {
        let spec =
            GemmPwSpec { m: 144, kg: 32, ng: 32, act, residual: false, dt: DType::F32, wdt: DType::F32 };
        let (ms, len) = cold_compile(&ctx, &spec, 100 + i as u32);
        println!("{:>12}: {ms:7.1}ms  ({len}B)", act.tag());
    }
}

/// OS Metal 셰이더 캐시를 무력화한 1회 컴파일 시간.
/// 소스에 유일한 주석(salt)을 붙이면 해시가 달라져 항상 콜드 경로를 탄다 —
/// 이걸 안 하면 앞선 실행에서 데워진 셰이더가 0.3ms로 나와 측정이 무의미해진다.
fn cold_compile(ctx: &GpuContext, spec: &dyn KernelSpec, salt: u32) -> (f64, usize) {
    let src = format!("// cold-salt {salt}\n{}", spec.wgsl(&ctx.caps));
    let len = src.len();
    let t0 = std::time::Instant::now();
    let module = ctx.device.create_shader_module(ai_gpu::wgpu::ShaderModuleDescriptor {
        label: None,
        source: ai_gpu::wgpu::ShaderSource::Wgsl(src.into()),
    });
    let _pipeline =
        ctx.device.create_compute_pipeline(&ai_gpu::wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    pollster::block_on(ai_gpu::readback::wait_idle(ctx)).unwrap();
    (t0.elapsed().as_secs_f64() * 1e3, len)
}
