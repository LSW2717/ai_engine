//! ai-convert CLI — ONNX → .sw.

use ai_convert::cli;
use ai_convert::onnx::import;
use ai_convert::passes::run_full;
use ai_convert::plan::lower;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match cli::parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}\n\n{}", cli::usage());
            std::process::exit(2);
        }
    };
    if opts.ctx.strip_refiner {
        eprintln!("--strip-refiner: 아직 미구현 (골든 대조용 옵션)");
        std::process::exit(2);
    }

    let run = || -> Result<(), Box<dyn std::error::Error>> {
        let bytes = std::fs::read(&opts.input)?;
        let imported = import::import(&bytes)?;
        let mut g = imported.graph;
        run_full(&mut g, &opts.ctx)?;

        if opts.summary {
            println!("== op 히스토그램 ==");
            for (op, n) in g.op_histogram() {
                println!("{op:>12} {n}");
            }
        }

        let name = opts
            .name
            .clone()
            .unwrap_or_else(|| {
                std::path::Path::new(&opts.input)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".into())
            });
        let (model, blob) = lower::lower(&g, &opts.ctx, &name)?;

        if opts.summary {
            println!(
                "== .sw == 텐서 {} | op {} | 블롭 {:.2}MB",
                model.tensors.len(),
                model.ops.len(),
                blob.len() as f64 / 1e6
            );
        }
        if opts.dump_json {
            println!("{}", String::from_utf8_lossy(&model.to_json()?));
        }
        if let Some(out) = &opts.output {
            let container = model.write_container(&blob)?;
            std::fs::write(out, &container)?;
            println!("{out} ({:.2}MB) 작성 완료", container.len() as f64 / 1e6);
        }
        Ok(())
    };

    if let Err(e) = run() {
        eprintln!("변환 실패: {e}");
        std::process::exit(1);
    }
}
