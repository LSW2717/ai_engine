//! 프로브: direct4의 가중치 4로드를 subgroup 셔플(레인당 1로드 + 4셔플)로 바꾸면
//! 빨라지는가. NG≤8이면 심드그룹(32레인)이 필요한 가중치 NG*4개를 전부 들 수 있다.
//! 주의: 부분 워크그룹의 early-return과 subgroup 균일성 — 프로브 shape은 나눠떨어짐.
//! (#[ignore])

use ai_core::ops::Conv2d;
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType, TensorDesc};
use ai_gpu::bench::bench_kernel;
use ai_gpu::context::DeviceCaps;
use ai_gpu::kernel::{KernelSpec, StorageDir};
use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
use ai_gpu::testsuite::{storage_in, storage_out};
use ai_gpu::GpuContext;

struct SubgroupW {
    base: ConvIgemmSpec,
    ng: u32,
}

impl KernelSpec for SubgroupW {
    fn cache_key(&self, caps: &DeviceCaps) -> String {
        format!("{} W=subgroup", self.base.cache_key(caps))
    }
    fn wgsl(&self, caps: &DeviceCaps) -> String {
        let src = self.base.wgsl(caps);
        let ng4 = self.ng * 4;
        // 1) enable 지시자 + 레인 빌트인
        let mut s = src;
        s = s.replace(
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(subgroup_invocation_id) sg_lane: u32) {",
        );
        // 2) 가중치 4로드 → 레인 1로드 + 4셔플. wb를 ng-무관 베이스로 바꾸고
        //    레인이 (lane % NG4)번째 vec4를 든 뒤 셔플로 분배한다.
        s = s.replace("* NG + ng) * 4u;", "* NG) * 4u;");
        s = s.replace(
            "let w0 = vec4f(W[wb]); let w1 = vec4f(W[wb + 1u]);",
            &format!(
                "let wl = vec4f(W[wb + (sg_lane % {ng4}u)]); let w0 = subgroupShuffle(wl, ng * 4u); let w1 = subgroupShuffle(wl, ng * 4u + 1u);"
            ),
        );
        s = s.replace(
            "let w2 = vec4f(W[wb + 2u]); let w3 = vec4f(W[wb + 3u]);",
            "let w2 = subgroupShuffle(wl, ng * 4u + 2u); let w3 = subgroupShuffle(wl, ng * 4u + 3u);",
        );
        s
    }
    fn bindings(&self) -> Vec<StorageDir> {
        self.base.bindings()
    }
    fn workgroups(&self) -> [u32; 3] {
        self.base.workgroups()
    }
}

#[test]
#[ignore]
fn bench_subgroup_weights() {
    let ctx = GpuContext::new_blocking().unwrap();
    if !ctx.caps.subgroups {
        eprintln!("skip: subgroup 미지원");
        return;
    }
    let dt = DType::F32;
    for (ih, iw, cin, cout) in
        [(144u32, 256u32, 16u32, 16u32), (144, 256, 35, 16), (72, 128, 32, 32)]
    {
        let k = 3u32;
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
        let wts = XorShift32::new(3).vec_f32((cout * cin * k * k) as usize);
        let (wb, _) = pack::pack_weights_conv(&wts, cout, cin, k, k, 4, dt);
        let flops = 2.0 * (oh * ow) as f64 * cout as f64 * cin as f64 * (k * k) as f64;

        let mk_bufs = |ctx: &GpuContext| {
            [
                storage_in(ctx, &pack::pack_nhwc(&XorShift32::new(5).vec_f32(din.elems()), &din)),
                storage_in(ctx, &wb),
                storage_in(ctx, &pack::pack_bias(&vec![0f32; cout as usize], cout, dt)),
                storage_out(ctx, dout.size_bytes()),
            ]
        };
        let b1 = mk_bufs(&ctx);
        let ra = pollster::block_on(bench_kernel(
            &ctx,
            &base,
            &[0u8; 16],
            &b1.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();
        let spec_s = SubgroupW { base, ng: cout.div_ceil(4) };
        let b2 = mk_bufs(&ctx);
        let rb = pollster::block_on(bench_kernel(
            &ctx,
            &spec_s,
            &[0u8; 16],
            &b2.iter().collect::<Vec<_>>(),
            flops,
        ))
        .unwrap();
        println!(
            "{ih}x{iw} {cin}->{cout}: 기존 {:.1}us ({:.0} GF/s) | subgroup {:.1}us ({:.0} GF/s)",
            ra.gpu_min_ms.unwrap_or(ra.wall_ms) * 1e3,
            ra.gflops,
            rb.gpu_min_ms.unwrap_or(rb.wall_ms) * 1e3,
            rb.gflops
        );
    }
}
