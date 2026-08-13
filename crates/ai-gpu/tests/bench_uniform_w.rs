//! 프로브: direct conv 가중치를 uniform(Metal constant 주소공간)으로 바인딩하면
//! 빨라지는가 — Apple은 constant AS에 전용 캐시+브로드캐스트 경로가 있다.
//! 이기면 프로덕션 배관(64KB 이하 가중치 conv에 적용)에 투자한다. (#[ignore])

use ai_core::ops::Conv2d;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};
use ai_gpu::bench::bench_kernel;
use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
use ai_gpu::testsuite::{storage_in, storage_out};
use ai_gpu::GpuContext;

use ai_gpu::kernel::{KernelSpec, StorageDir};
use ai_gpu::context::DeviceCaps;

struct UniformW {
    base: ConvIgemmSpec,
    n_vec4: u32,
}

impl KernelSpec for UniformW {
    fn cache_key(&self, caps: &DeviceCaps) -> String {
        format!("{} W=uniform", self.base.cache_key(caps))
    }
    fn wgsl(&self, caps: &DeviceCaps) -> String {
        self.base.wgsl(caps).replace(
            "var<storage, read> W: array<wv4>;",
            &format!("var<uniform> W: array<vec4f, {}>;", self.n_vec4),
        )
    }
    fn bindings(&self) -> Vec<StorageDir> {
        vec![StorageDir::Read, StorageDir::Uniform, StorageDir::Read, StorageDir::ReadWrite]
    }
    fn workgroups(&self) -> [u32; 3] {
        self.base.workgroups()
    }
}

fn uniform_in(ctx: &GpuContext, bytes: &[u8]) -> ai_gpu::wgpu::Buffer {
    use ai_gpu::wgpu;
    use wgpu::util::DeviceExt;
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform-w"),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

#[test]
#[ignore]
fn bench_uniform_weights() {
    let ctx = GpuContext::new_blocking().unwrap();
    let dt = DType::F32;
    for (ih, iw, cin, cout, k) in
        [(144u32, 256u32, 16u32, 16u32, 3u32), (144, 256, 35, 16, 3), (72, 128, 32, 32, 3)]
    {
        let op = Conv2d {
            cin,
            cout,
            kh: k,
            kw: k,
            sh: 1,
            sw: 1,
            pad: [1; 4],
            dil: 1,
            groups: 1,
            act: Activation::Relu,
        };
        let (oh, ow) = op.out_hw(ih, iw);
        let din = TensorDesc::new(ih, iw, cin, dt);
        let dout = TensorDesc::new(oh, ow, cout, dt);
        let base = ConvIgemmSpec::from_op(&op, ih, iw, false, dt);
        let kgp = cin.div_ceil(4).next_multiple_of(4);
        let n_vec4 = k * k * kgp * cout.div_ceil(4) * 4;

        let wts = XorShift32::new(3).vec_f32((cout * cin * k * k) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, k, k, 4, dt);
        let bias = pack::pack_bias(&vec![0f32; cout as usize], cout, dt);
        let flops = 2.0 * (oh * ow) as f64 * cout as f64 * cin as f64 * (k * k) as f64;

        // A) 기존 storage 가중치
        let bufs = [
            storage_in(&ctx, &pack::pack_nhwc(&XorShift32::new(5).vec_f32(din.elems()), &din)),
            storage_in(&ctx, &wb),
            storage_in(&ctx, &bias),
            storage_out(&ctx, dout.size_bytes()),
        ];
        let ra = pollster::block_on(bench_kernel(
            &ctx,
            &base,
            &[0u8; 16],
            &bufs.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();

        // B) uniform 가중치
        let spec_u = UniformW { base, n_vec4 };
        let bufs_u = [
            storage_in(&ctx, &pack::pack_nhwc(&XorShift32::new(5).vec_f32(din.elems()), &din)),
            uniform_in(&ctx, &wb),
            storage_in(&ctx, &bias),
            storage_out(&ctx, dout.size_bytes()),
        ];
        let rb = pollster::block_on(bench_kernel(
            &ctx,
            &spec_u,
            &[0u8; 16],
            &bufs_u.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();

        println!(
            "{ih}x{iw} {cin}->{cout}: storage {:.1}us ({:.0} GF/s) | uniform {:.1}us ({:.0} GF/s)",
            ra.gpu_min_ms.unwrap_or(ra.wall_ms) * 1e3,
            ra.gflops,
            rb.gpu_min_ms.unwrap_or(rb.wall_ms) * 1e3,
            rb.gflops
        );
    }
}
