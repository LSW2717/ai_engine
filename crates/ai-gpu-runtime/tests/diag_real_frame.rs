//! 실제 사람 프레임으로 pha를 뽑아 PGM으로 덤프 (진단용, #[ignore])
//!
//! 데모가 "마스크가 아예 안 생긴다"고 할 때 원인이 (a) 입력 레이아웃 규약,
//! (b) 정규화 범위, (c) 엔진 자체 중 어디인지 가른다. GPU vs CPU 오라클 비교는
//! **같은 배열을 같은 규약으로 해석**하므로 이걸 못 잡는다.
//!
//! 입력: AI_FRAME_RGB=<256×144 rgb24 raw> (ffmpeg -pix_fmt rgb24)
//! 출력: AI_PHA_OUT=<경로.pgm>

use ai_convert::ir::Graph;
use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

fn rvm_ctx(g: &Graph, size: (u32, u32)) -> (Ctx, &'static str) {
    let official = g.inputs.iter().any(|n| n == "src");
    let (input, suffix): (&'static str, &str) =
        if official { ("src", "i") } else { ("input_1", "") };
    let ctx = Ctx {
        size: Some(size),
        set_inputs: if official { vec![("downsample_ratio".into(), 1.0)] } else { vec![] },
        states: (1..=4).map(|i| (format!("r{i}{suffix}"), format!("r{i}o"))).collect(),
        ..Default::default()
    };
    (ctx, input)
}

#[test]
#[ignore]
fn diag_real_frame() {
    let raw_path = std::env::var("AI_FRAME_RGB").expect("AI_FRAME_RGB 필요");
    let out_path = std::env::var("AI_PHA_OUT").unwrap_or_else(|_| "/tmp/pha.pgm".into());
    let (w, h) = (256usize, 144usize);
    let raw = std::fs::read(&raw_path).unwrap();
    assert_eq!(raw.len(), w * h * 3, "rgb24 256×144가 아님");

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(std::env::var("AI_ONNX").unwrap_or_else(|_| "../../models/rvm_fp32.onnx".into()));
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let (mut cctx, in_name) = rvm_ctx(&g, (256, 144));
    // AI_FP16W=1: 가중치만 f16으로 구워 ORT 대비 오차를 본다 (데모 기본 구성)
    cctx.fp16_weights = std::env::var("AI_FP16W").is_ok();
    let cctx = cctx;
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();

    // 두 가지 레이아웃 규약을 모두 시도해 어느 쪽이 사람 마스크를 내는지 본다
    let nhwc: Vec<f32> = raw.iter().map(|b| *b as f32 / 255.0).collect(); // 이미 인터리브
    let mut nchw = vec![0f32; w * h * 3];
    for c in 0..3 {
        for p in 0..w * h {
            nchw[c * w * h + p] = raw[p * 3 + c] as f32 / 255.0;
        }
    }
    for (tag, data) in [("nhwc", &nhwc), ("nchw", &nchw)] {
        // ⚠ CPU/ORT는 제로 상태 1프레임이다. GPU도 **같은 조건**이어야 비교가 성립한다
        // (여러 프레임 돌리면 상태가 쌓여 다른 걸 재게 된다 — 한 번 여기서 헛짚었다).
        let mut model = pollster::block_on(Model::load(&ctx, &container)).unwrap();
        model.upload_input(&ctx, in_name, data).unwrap();
        pollster::block_on(model.infer(&ctx)).unwrap();
        let pha = pollster::block_on(model.read_output(&ctx, "pha")).unwrap();
        let mean = pha.iter().sum::<f32>() / pha.len() as f32;
        let max = pha.iter().cloned().fold(0f32, f32::max);
        let fg = pha.iter().filter(|v| **v > 0.5).count();
        println!(
            "[{tag}] pha 평균 {mean:.4} 최대 {max:.4} | 전경픽셀(>0.5) {fg}/{} ({:.1}%)",
            pha.len(),
            fg as f32 / pha.len() as f32 * 100.0
        );
        std::fs::write(
            format!("{out_path}.{tag}.f32"),
            pha.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
        pgm.extend(pha.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
        std::fs::write(format!("{out_path}.{tag}.pgm"), pgm).unwrap();
    }
    // 같은 lowering을 CPU 실행기로도 돌린다 — GPU와 같으면 버그는 변환기(lowering),
    // 다르면 GPU 커널이다. ORT 오라클(전경 18.7%)이 정답 기준.
    let mut cpu = ai_convert::verify::CpuExec::new(&sw, &blob);
    let in_tid = sw.tensors.iter().position(|t| t.name == in_name).unwrap() as u32;
    let pha_tid = sw.tensors.iter().position(|t| t.name == "pha").unwrap() as u32;
    cpu.set_input(in_tid, nhwc.clone());
    cpu.run().unwrap();
    let cpha = cpu.read(pha_tid).unwrap();
    let cfg = cpha.iter().filter(|v| **v > 0.5).count();
    println!(
        "[CPU lowering] pha 평균 {:.4} 최대 {:.4} | 전경 {:.1}%",
        cpha.iter().sum::<f32>() / cpha.len() as f32,
        cpha.iter().cloned().fold(0f32, f32::max),
        cfg as f32 / cpha.len() as f32 * 100.0
    );
    std::fs::write(
        format!("{out_path}.cpu.f32"),
        cpha.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    println!("덤프: {out_path}.*.f32");
}
