//! 층별 발산 지점 추적용 — CpuExec의 전 텐서를 raw .f32로 덤프.
//! tools/ort_dump.py --intermediates 산출물과 tools/diff_dump.py로 대조한다.
//!
//! 사용: AI_ONNX=<onnx> AI_DUMP=<dir> \
//!   cargo test --release -p ai-cpu --test dump_all -- --ignored --nocapture

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_convert::verify::CpuExec;

#[test]
#[ignore]
fn dump_all_tensors() {
    let onnx = std::env::var("AI_ONNX").expect("AI_ONNX");
    let dump = std::path::PathBuf::from(std::env::var("AI_DUMP").expect("AI_DUMP"));
    std::fs::create_dir_all(&dump).unwrap();

    let mut g = import(&std::fs::read(&onnx).unwrap()).unwrap().graph;
    let ctx = Ctx::default();
    run_full(&mut g, &ctx).unwrap();
    let (sw, blob) = lower(&g, &ctx, "dump").unwrap();

    // 게이트와 같은 입력 (ort_dump.py input_nhwc.f32)
    let oracle = std::env::var("AI_ORACLE").expect("AI_ORACLE=<ort_dump 디렉토리>");
    let ib = std::fs::read(std::path::Path::new(&oracle).join("input_nhwc.f32")).unwrap();
    let input: Vec<f32> =
        ib.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let in_tid = sw.inputs[0];
    let mut ex = CpuExec::new(&sw, &blob);
    ex.set_input(in_tid, input);
    ex.run().unwrap();

    for (tid, t) in sw.tensors.iter().enumerate() {
        let Ok(v) = ex.read(tid as u32) else { continue };
        let safe: String = t
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' { c } else { '_' })
            .collect();
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dump.join(format!("{tid:04}__{safe}.f32")), bytes).unwrap();
    }
    println!("덤프 완료: {} 텐서", sw.tensors.len());
}
