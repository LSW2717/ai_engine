//! **GPU 게이트+벤치 (범용)** — onnxruntime 기준값과 GPU 전 출력 대조 + 프레임타임.
//! gate_ort_models(CPU판)와 같은 규약: tools/ort_dump.py 산출물을 소비한다.
//!
//! 사용:
//!   AI_ONNX=<onnx> AI_ORACLE=<dir> [AI_REPS=20] \
//!     cargo test --release -p ai-gpu-runtime --test gate_models_gpu -- --ignored --nocapture

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_gpu::GpuContext;
use ai_gpu_runtime::Model;

fn read_f32s(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p:?}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore]
fn gpu_outputs_match_ort() {
    let onnx = std::env::var("AI_ONNX").expect("AI_ONNX=<onnx> 필요");
    let oracle = std::path::PathBuf::from(std::env::var("AI_ORACLE").expect("AI_ORACLE=<dir>"));

    let mut g = import(&std::fs::read(&onnx).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "gate").unwrap();
    let container = sw.write_container(&blob).unwrap();

    let gctx = GpuContext::new_blocking().unwrap();
    let mut model = pollster::block_on(Model::load(&gctx, &container)).unwrap();

    let in_name = sw.tensors[sw.inputs[0] as usize].name.clone();
    let input = read_f32s(&oracle.join("input_nhwc.f32"));
    model.upload_input(&gctx, &in_name, &input).unwrap();
    pollster::block_on(model.infer(&gctx)).unwrap();

    // AI_BISECT=1: GPU vs CpuExec 층별 이등분 — 첫 발산 op을 찾는다
    if std::env::var("AI_BISECT").is_ok() {
        let mut cpu = ai_convert::verify::CpuExec::new(&sw, &blob);
        cpu.set_input(sw.inputs[0], input.clone());
        cpu.run().unwrap();
        use ai_core::format::SwOp::*;
        for op in sw.ops.iter() {
            let out_tid = match op {
                Conv { out, .. } | Binary { out, .. } | Gpool { out, .. }
                | Avgpool { out, .. } | Maxpool { out, .. } | Resize { out, .. }
                | Concat { out, .. } | Chcopy { out, .. } | SeGate { out, .. }
                | Act { out, .. } | Mix { out, .. } => *out,
            };
            let got = pollster::block_on(model.debug_read_tensor(&gctx, out_tid)).unwrap();
            let want = cpu.read(out_tid).unwrap();
            let max_err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
                .fold(0f32, f32::max);
            let t = &sw.tensors[out_tid as usize];
            if max_err > 1e-3 {
                println!(
                    "발산 tid {out_tid} {:?} max_err {max_err:.3e} ({}x{}x{}) {}",
                    std::mem::discriminant(op), t.h, t.w, t.c,
                    &t.name[t.name.len().saturating_sub(60)..]
                );
                // 첫 불일치 좌표·값 샘플 (패턴 판독용)
                let (w_, c_) = (t.w as usize, t.c as usize);
                let mut shown = 0;
                for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                    if (g - wv).abs() / wv.abs().max(1.0) > 1e-3 {
                        let (px, ch) = (i / c_, i % c_);
                        println!(
                            "  [y{} x{} c{}] gpu {g:.4} cpu {wv:.4}",
                            px / w_, px % w_, ch
                        );
                        shown += 1;
                        if shown >= 6 {
                            break;
                        }
                    }
                }
                return;
            }
        }
        return;
    }

    let mut checked = 0;
    let mut worst = (0f32, String::new());
    for entry in std::fs::read_dir(&oracle).unwrap() {
        let path = entry.unwrap().path();
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(name) = fname.strip_prefix("out__").and_then(|s| s.strip_suffix(".f32"))
        else {
            continue;
        };
        let want = read_f32s(&path);
        let got = pollster::block_on(model.read_output(&gctx, name))
            .unwrap_or_else(|e| panic!("출력 {name} 읽기 실패: {e:?}"));
        assert_eq!(want.len(), got.len(), "{name} 길이");
        let mut max_err = 0f32;
        for (a, g) in want.iter().zip(&got) {
            max_err = max_err.max((a - g).abs() / a.abs().max(1.0));
        }
        println!("{name}: max_err {max_err:.3e} ({}elem)", want.len());
        if max_err > worst.0 {
            worst = (max_err, name.to_string());
        }
        let tol: f32 =
            std::env::var("AI_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-3);
        assert!(max_err <= tol, "{name} max_err {max_err} 초과");
        checked += 1;
    }
    assert!(checked > 0, "오라클 출력 파일 없음: {oracle:?}");

    // 프레임타임 (infer는 완료 동기 — 보수적 벽시계)
    let reps: usize = std::env::var("AI_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    for _ in 0..3 {
        pollster::block_on(model.infer(&gctx)).unwrap();
    }
    // 마지막에 출력 리드백 1회로 GPU 완료를 강제 — 제출 시간만 재는 허수 방지
    let sync_name = sw.tensors[sw.outputs[0] as usize].name.clone();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        pollster::block_on(model.infer(&gctx)).unwrap();
    }
    pollster::block_on(model.read_output(&gctx, &sync_name)).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("worst: {} {:.3e} ({checked}출력) | GPU 프레임타임 **{ms:.2}ms**", worst.1, worst.0);
}
