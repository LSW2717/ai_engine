//! 실제 이미지 1프레임 GPU vs CPU 이분탐색 (진단용, #[ignore])
//!
//! `bisect2`는 **랜덤 입력**으로 비교해서 5.8e-6로 통과하는데, 실제 사진을 넣으면
//! pha가 mean_err 0.083으로 갈린다 (CPU lowering은 ORT와 0.0016으로 일치 = 변환기는 정답).
//! 즉 GPU 커널 어딘가가 자연 이미지에서만 드러나는 방식으로 틀렸다.
//! 첫 발산 op을 찾는다.
//!
//! AI_FRAME_RGB=<256×144 rgb24 raw>

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;
use ai_gpu::GpuContext;
use ai_runtime::Model;

#[test]
#[ignore]
fn bisect_real_frame() {
    unsafe { std::env::set_var("AI_RT_NO_REUSE", "1") };
    let raw = std::fs::read(std::env::var("AI_FRAME_RGB").expect("AI_FRAME_RGB 필요")).unwrap();
    let input: Vec<f32> = raw.iter().map(|b| *b as f32 / 255.0).collect();
    assert_eq!(input.len(), 144 * 256 * 3);

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
        size: Some((256, 144)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: (1..=4).map(|i| (format!("r{i}i"), format!("r{i}o"))).collect(),
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let tid_of = |name: &str| {
        sw.tensors.iter().position(|t| t.name == name).map(|i| i as u32).unwrap()
    };

    let mut cpu = CpuExec::new(&sw, &blob);
    cpu.set_input(tid_of("src"), input.clone());
    cpu.run().unwrap();

    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
    model.upload_input(&ctx, "src", &input).unwrap();
    pollster::block_on(model.infer(&ctx)).unwrap();
    pollster::block_on(ai_gpu::readback::wait_idle(&ctx)).unwrap();

    let mut worst: Vec<(f32, usize, String)> = Vec::new();
    let mut frac_reported = false;
    for (i, op) in sw.ops.iter().enumerate() {
        use ai_core::format::SwOp::*;
        let out_tid = match op {
            Conv { out, .. } | Binary { out, .. } | Gpool { out, .. } | Avgpool { out, .. }
            | Resize { out, .. } | Concat { out, .. } | Chcopy { out, .. }
            | SeGate { out, .. } | Act { out, .. } | Mix { out, .. } => *out,
        };
        if sw.states.iter().any(|s| s.output == out_tid || s.input == out_tid) {
            continue;
        }
        let got = pollster::block_on(model.debug_read_tensor(&ctx, out_tid)).unwrap();
        let want = cpu.read(out_tid).unwrap();
        let max_err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        let nbad = got.iter().zip(&want).filter(|(g, w)| (*g - *w).abs() > 1e-3).count();
        let frac = nbad as f32 / got.len().max(1) as f32;
        let name = sw.tensors[out_tid as usize].name.clone();
        // 한두 셀짜리 경계 artifact는 무시하고 **비율**로 유의한 발산을 찾는다
        if frac > 0.01 && !frac_reported {
            frac_reported = true;
            let (oh2, ow2, oc2) = {
                let t = &sw.tensors[out_tid as usize];
                (t.h as usize, t.w as usize, t.c as usize)
            };
            let mut bad_ch = vec![0usize; oc2];
            let mut bad_row = vec![0usize; oh2];
            let mut bad_col = vec![0usize; ow2];
            for p in 0..oh2 * ow2 {
                for c in 0..oc2 {
                    if (got[p * oc2 + c] - want[p * oc2 + c]).abs() > 1e-3 {
                        bad_ch[c] += 1;
                        bad_row[p / ow2] += 1;
                        bad_col[p % ow2] += 1;
                    }
                }
            }
            let bc: Vec<usize> = (0..oc2).filter(|c| bad_ch[*c] > 0).collect();
            let br: Vec<usize> = (0..oh2).filter(|r| bad_row[*r] > 0).collect();
            let bcol: Vec<usize> = (0..ow2).filter(|c| bad_col[*c] > 0).collect();
            println!("    틀린 채널 {}/{oc2}: {:?}", bc.len(), &bc[..bc.len().min(16)]);
            println!("    틀린 행 {}/{oh2}: {:?}", br.len(), br);
            println!("    틀린 열 {}/{ow2}: {:?}", bcol.len(), bcol);
            println!(
                ">>> 최초 유의 발산 op[{i}] out={name} 틀린비율 {:.1}% max_err {max_err:.3e}",
                frac * 100.0
            );
            println!("    op: {op:?}");
        }
        if max_err > 1e-3 && worst.is_empty() {
            println!("최초 발산 op[{i}] out={name} max_err={max_err:.3e}");
            println!("  op: {op:?}");
            // 이 op의 **입력**들이 GPU에서 맞는지 확인 — 입력이 맞는데 출력이 틀리면
            // 커널/바인딩 문제, 입력이 이미 틀리면 앞선 비교가 놓친 것이다.
            let ins: Vec<u32> = match op {
                Conv { input, srcs, res, .. } => {
                    let mut v: Vec<u32> =
                        if srcs.is_empty() { vec![*input] } else { srcs.iter().map(|p| p.input).collect() };
                    if let Some(r) = res { v.push(*r); }
                    v
                }
                Binary { a, b, .. } => {
                    let mut v = vec![*a];
                    if let ai_core::format::SwOperand::Tensor { tid } = b { v.push(*tid); }
                    v
                }
                Mix { a, b, z, .. } => vec![*a, *b, *z],
                _ => vec![],
            };
            for t in ins {
                let g = pollster::block_on(model.debug_read_tensor(&ctx, t)).unwrap();
                let w = cpu.read(t).unwrap();
                let e = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let tt = &sw.tensors[t as usize];
                println!(
                    "  입력 tid={t} name={} {}x{}x{} alias={:?} max_err={e:.3e}",
                    tt.name, tt.h, tt.w, tt.c, tt.alias
                );
                // 뷰라면: 백킹이 맞는지, 백킹을 직접 잘라낸 값이 맞는지 각각 본다
                if let Some(a) = tt.alias {
                    let (root, off) = sw.resolve_alias(t);
                    let rt = &sw.tensors[root as usize];
                    let gr = pollster::block_on(model.debug_read_tensor(&ctx, root)).unwrap();
                    let wr = cpu.read(root).unwrap();
                    let er = gr.iter().zip(&wr).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    println!(
                        "    백킹 tid={root} name={} {}x{}x{} cg_off={off} (직접 alias {:?}) max_err={er:.3e}",
                        rt.name, rt.h, rt.w, rt.c, a
                    );
                    // 백킹에서 채널 구간을 직접 잘라 CPU 뷰와 비교 → gather 로직 검증
                    let (bc, vc, start) = (rt.c as usize, tt.c as usize, (off * 4) as usize);
                    let px = (tt.h * tt.w) as usize;
                    let mut sliced = vec![0f32; px * vc];
                    for p in 0..px {
                        for c in 0..vc {
                            sliced[p * vc + c] = gr[p * bc + start + c];
                        }
                    }
                    let es = sliced.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    println!("    백킹 직접 슬라이스 vs CPU뷰: max_err={es:.3e}");
                }
            }
            // 틀린 원소의 분포 — 경계 픽셀만인지, 특정 채널만인지, 전면적인지
            let (oh, ow, oc) = {
                let t = &sw.tensors[out_tid as usize];
                (t.h as usize, t.w as usize, t.c as usize)
            };
            let mut bad_ch = vec![0usize; oc];
            let mut bad_px = 0usize;
            let mut nbad = 0usize;
            for p in 0..oh * ow {
                let mut any = false;
                for c in 0..oc {
                    let k = p * oc + c;
                    if (got[k] - want[k]).abs() > 1e-3 {
                        bad_ch[c] += 1;
                        nbad += 1;
                        any = true;
                    }
                }
                if any { bad_px += 1; }
            }
            println!(
                "  출력 {oh}x{ow}x{oc}: 틀린 원소 {nbad}/{} ({:.1}%), 틀린 픽셀 {bad_px}/{}",
                oh * ow * oc,
                nbad as f32 / (oh * ow * oc) as f32 * 100.0,
                oh * ow
            );
            let ch_bad: Vec<usize> = (0..oc).filter(|c| bad_ch[*c] > 0).collect();
            println!("  틀린 채널 {}개: {:?}", ch_bad.len(), &ch_bad[..ch_bad.len().min(20)]);
            for k in (0..oh * ow * oc).filter(|k| (got[*k] - want[*k]).abs() > 1e-3).take(4) {
                println!("    [px {} ch {}] got {:.4} want {:.4}", k / oc, k % oc, got[k], want[k]);
            }
            // 마지막 타일(px 136..143, ch 48..63) 전체를 본다 — 워크그룹 커버리지 확인
            println!("  마지막 타일 (px 136..143, ch 48..63):");
            let mut zero_cnt = 0;
            let mut tot = 0;
            for p in 136..oh * ow {
                for c in 48..oc {
                    let k = p * oc + c;
                    tot += 1;
                    if got[k] == 0.0 { zero_cnt += 1; }
                }
            }
            println!("    GPU가 정확히 0인 원소: {zero_cnt}/{tot}");
            let mut wz = 0;
            for p in 136..oh * ow {
                for c in 48..oc {
                    if want[p * oc + c].abs() < 1e-6 { wz += 1; }
                }
            }
            println!("    CPU가 0인 원소: {wz}/{tot}");
            // 이 op의 가중치를 blob에서 직접 읽어 문제 채널(ng,j)이 0인지 본다.
            // GPU 두 변형이 같은 오답을 내면 커널이 아니라 데이터(가중치)가 의심된다.
            if let Conv { w, cout, .. } = op {
                let bytes = &blob[w.off as usize..(w.off + w.len) as usize];
                let f: &[f32] = bytemuck::cast_slice(bytes);
                let ng_cnt = cout.div_ceil(4) as usize;
                let kgp = 32usize; // 이 op 기준
                let mut per_ch_nonzero = vec![0usize; *cout as usize];
                for tap in 0..9usize {
                    for kg in 0..kgp {
                        for ng in 0..ng_cnt {
                            for j in 0..4usize {
                                let base = ((tap * kgp + kg) * ng_cnt + ng) * 4 + j;
                                for lane in 0..4usize {
                                    if f[base * 4 + lane] != 0.0 {
                                        per_ch_nonzero[ng * 4 + j] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                let dead: Vec<usize> =
                    (0..*cout as usize).filter(|c| per_ch_nonzero[*c] == 0).collect();
                println!("  blob 가중치: 전부 0인 출력채널 {}개 {:?}", dead.len(), &dead[..dead.len().min(10)]);
                println!("  ch54 비영 계수 {}개 (ch53 {}개, ch55 {}개)",
                    per_ch_nonzero[54], per_ch_nonzero[53], per_ch_nonzero[55]);
            }
            // 문제 셀이 속한 vec4(ch 52..55)를 통째로 본다.
            // 4레인이 전부 0이면 그 셀이 아예 안 쓰인 것(커버리지 구멍),
            // 한 레인만 0이면 계산 문제다.
            for c in 52..56usize {
                let k = 137 * oc + c;
                println!("    px137 ch{c}: got {:.5} want {:.5}", got[k], want[k]);
            }
            // 실제 데이터(GPU t88, zeros, blob 가중치)로 이 커널만 단독 재현.
            // 재현되면 커널, 안 되면 그래프 바인딩 문제다.
            if let Conv { srcs, cout, w, b, act: a2, .. } = op {
                use ai_gpu::kernels::common::source::SrcView;
                use ai_gpu::kernels::conv_igemm::ConvIgemmSpec;
                let backing = pollster::block_on(model.debug_read_tensor(&ctx, 88)).unwrap();
                let (bh, bw, bc2) = { let t = &sw.tensors[88]; (t.h, t.w, t.c) };
                let bbuf = ai_gpu::testsuite::storage_in(
                    &ctx, &ai_core::pack::pack_nhwc(&backing, &ai_core::TensorDesc::new(bh, bw, bc2, ai_core::DType::F32)));
                let zeros = vec![0f32; (bh * bw * 64) as usize];
                let zbuf = ai_gpu::testsuite::storage_in(
                    &ctx, &ai_core::pack::pack_nhwc(&zeros, &ai_core::TensorDesc::new(bh, bw, 64, ai_core::DType::F32)));
                let wbuf = ai_gpu::testsuite::storage_in(&ctx, &blob[w.off as usize..(w.off + w.len) as usize]);
                let bibuf = ai_gpu::testsuite::storage_in(&ctx, &blob[b.off as usize..(b.off + b.len) as usize]);
                let convop = ai_core::ops::Conv2d {
                    cin: 128, cout: *cout, kh: 3, kw: 3, sh: 1, sw: 1,
                    pad: [1; 4], dil: 1, groups: 1,
                    // AI_DIAG_NOACT=1: 활성화를 빼고 원시 누산값을 본다 (오버플로/NaN 확인)
                    act: if std::env::var("AI_DIAG_NOACT").is_ok() { ai_core::Activation::None } else { *a2 },
                };
                // 같은 데이터를 (a) 뷰로, (b) 평범한 64채널 버퍼로 각각 줘서 비교
                let mut sliced = vec![0f32; (bh * bw * 64) as usize];
                for px in 0..(bh * bw) as usize {
                    for c in 0..64usize {
                        sliced[px * 64 + c] = backing[px * 128 + 64 + c];
                    }
                }
                let slicedbuf = ai_gpu::testsuite::storage_in(
                    &ctx, &ai_core::pack::pack_nhwc(&sliced, &ai_core::TensorDesc::new(bh, bw, 64, ai_core::DType::F32)));
                let use_view = std::env::var("AI_DIAG_PLAIN").is_err();
                let mut spec = ConvIgemmSpec::from_op(&convop, bh, bw, false, ai_core::DType::F32);
                spec.srcs[0] = if use_view { SrcView::view(64, 128, 64) } else { SrcView::plain(64) };
                spec.srcs[1] = SrcView::plain(64);
                let dout = ai_core::TensorDesc::new(bh, bw, *cout, ai_core::DType::F32);
                let outb = ai_gpu::testsuite::storage_out(&ctx, dout.size_bytes());
                let compiled = pollster::block_on(ai_gpu::kernel::compile(&ctx, &spec)).unwrap();
                let pb2 = ctx.device.create_buffer(&ai_gpu::wgpu::BufferDescriptor {
                    label: None, size: 256,
                    usage: ai_gpu::wgpu::BufferUsages::UNIFORM | ai_gpu::wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false });
                ctx.queue.write_buffer(&pb2, 0, &[0u8; 16]);
                let mut ents = vec![ai_gpu::wgpu::BindGroupEntry { binding: 0,
                    resource: ai_gpu::wgpu::BindingResource::Buffer(ai_gpu::wgpu::BufferBinding {
                        buffer: &pb2, offset: 0, size: Some(std::num::NonZeroU64::new(256).unwrap()) }) }];
                let src0 = if use_view { &bbuf } else { &slicedbuf };
                for (bi, bufr) in [src0, &zbuf, &wbuf, &bibuf, &outb].iter().enumerate() {
                    ents.push(ai_gpu::wgpu::BindGroupEntry { binding: (bi + 1) as u32, resource: bufr.as_entire_binding() });
                }
                let bg = ctx.device.create_bind_group(&ai_gpu::wgpu::BindGroupDescriptor {
                    label: None, layout: &compiled.bgl, entries: &ents });
                let stg = ctx.device.create_buffer(&ai_gpu::wgpu::BufferDescriptor {
                    label: None, size: dout.size_bytes(),
                    usage: ai_gpu::wgpu::BufferUsages::MAP_READ | ai_gpu::wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false });
                let mut e2 = ctx.device.create_command_encoder(&ai_gpu::wgpu::CommandEncoderDescriptor { label: None });
                { let mut ps = e2.begin_compute_pass(&ai_gpu::wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                  ps.set_pipeline(&compiled.pipeline); ps.set_bind_group(0, &bg, &[0]);
                  let g = <ConvIgemmSpec as ai_gpu::kernel::KernelSpec>::workgroups(&spec);
                  ps.dispatch_workgroups(g[0], g[1], g[2]); }
                e2.copy_buffer_to_buffer(&outb, 0, &stg, 0, dout.size_bytes());
                ctx.queue.submit([e2.finish()]);
                let by = pollster::block_on(ai_gpu::readback::read_buffers(&ctx, &[&stg])).unwrap().remove(0);
                let so = ai_core::pack::unpack_nhwc(&by, &dout);
                let se = so.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
                let v = so[137 * oc + 54];
                println!(
                    "  단독 재현(뷰={use_view}): max_err {se:.3e}, px137 ch54 = {v:.5} (nan={} inf={})",
                    v.is_nan(), v.is_infinite()
                );
                let _ = srcs;
            }
            worst.push((max_err, i, name));
        }
    }
    if worst.is_empty() {
        println!("발산 없음");
    }
}
