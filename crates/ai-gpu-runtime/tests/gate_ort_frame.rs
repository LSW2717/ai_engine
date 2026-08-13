//! **정확도 게이트** — 실제 사람 프레임 1장에 대해 GPU pha를 ORT 기준값과 대조.
//!
//! 이 테스트가 존재하는 이유: `rvm_e2e`는 엔진 자체 CPU 실행기와 랜덤 노이즈만 쓴다.
//! GPU와 CPU가 같은 lowering을 공유하므로 **둘이 같이 틀리면 통과**한다. 실제로
//! tanh 오버플로가 그렇게 빠져나가 마스크를 반토막 냈고 rvm_e2e는 5.8e-6으로
//! 통과했다. 랜덤 입력은 활성화 전 값이 작아 그런 버그를 만들지도 못한다.
//! 커널·변형 정책을 건드렸다면 **반드시 이걸 돌린다**.
//!
//! 기준값 갱신: `tools/rvm_ref_frame.py` (onnxruntime 필요) 참고.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

const W: usize = 256;
const H: usize = 144;
/// ORT 대비 평균오차 상한. 실측 기준선: fp32/가중치fp16 모두 0.0016
/// (tanh 버그 시절엔 0.084였다 — 50배 차이라 경계가 넉넉하다).
const MEAN_ERR_MAX: f32 = 0.005;
/// 전경 픽셀 비율이 이만큼 이상 어긋나면 마스크가 무너진 것이다
/// (tanh 버그 때 18.7% → 9.8%).
const FG_RATIO_TOL: f32 = 0.02;

fn data(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data").join(name)
}

fn run_gpu(ctx: &GpuContext, fp16_weights: bool, src_nhwc: &[f32]) -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fp32.onnx");
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
        size: Some((W as u32, H as u32)),
        set_inputs: vec![("downsample_ratio".into(), 1.0)],
        states: (1..=4).map(|i| (format!("r{i}i"), format!("r{i}o"))).collect(),
        fp16_weights,
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let mut model = pollster::block_on(Model::load(ctx, &container)).unwrap();
    model.upload_input(ctx, "src", src_nhwc).unwrap();
    // 상태 0에서 1프레임 — ORT 기준값과 같은 조건
    pollster::block_on(model.infer(ctx)).unwrap();
    pollster::block_on(model.read_output(ctx, "pha")).unwrap()
}

#[test]
fn gpu_pha_matches_ort_on_real_frame() {
    let raw = std::fs::read(data("frame_256x144.rgb")).expect("tests/data/frame_256x144.rgb 없음");
    assert_eq!(raw.len(), W * H * 3);
    let refbytes = std::fs::read(data("pha_ref_256x144.f32")).expect("ORT 기준값 없음");
    let pha_ref: Vec<f32> = refbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(pha_ref.len(), W * H);

    // 엔진 입력은 논리 NHWC 0..1
    let src: Vec<f32> = raw.iter().map(|b| *b as f32 / 255.0).collect();
    let fg_ref = pha_ref.iter().filter(|v| **v > 0.5).count() as f32 / pha_ref.len() as f32;

    let ctx = GpuContext::new_blocking().unwrap();
    let mut failures = Vec::new();
    for fp16_weights in [false, true] {
        if fp16_weights && !ctx.caps.f16 {
            continue;
        }
        let pha = run_gpu(&ctx, fp16_weights, &src);
        assert_eq!(pha.len(), W * H);
        let mean_err: f32 =
            pha.iter().zip(&pha_ref).map(|(a, b)| (a - b).abs()).sum::<f32>() / pha.len() as f32;
        let max_err =
            pha.iter().zip(&pha_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let fg = pha.iter().filter(|v| **v > 0.5).count() as f32 / pha.len() as f32;
        let tag = if fp16_weights { "가중치 f16" } else { "fp32    " };
        println!(
            "{tag}  mean_err {mean_err:.5}  max_err {max_err:.4}  전경 {:.1}% (ORT {:.1}%)",
            fg * 100.0,
            fg_ref * 100.0
        );
        if mean_err > MEAN_ERR_MAX {
            failures.push(format!("{tag}: mean_err {mean_err:.5} > {MEAN_ERR_MAX}"));
        }
        if (fg - fg_ref).abs() > FG_RATIO_TOL {
            failures.push(format!(
                "{tag}: 전경비율 {:.3} vs ORT {:.3} — 마스크가 무너졌다",
                fg, fg_ref
            ));
        }
    }
    assert!(failures.is_empty(), "ORT 대조 실패:\n{}", failures.join("\n"));
}
