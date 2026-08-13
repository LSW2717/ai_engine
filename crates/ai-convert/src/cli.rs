//! 수제 CLI 파서 — 프로젝트 최소 의존 문화(clap 없음).

use crate::passes::Ctx;

pub struct Opts {
    pub input: String,
    pub output: Option<String>,
    pub ctx: Ctx,
    pub name: Option<String>,
    pub summary: bool,
    pub dump_json: bool,
}

pub fn usage() -> &'static str {
    "사용법: ai-convert <input.onnx> -o <out.sw> [--size WxH] [--fp16] [--fp16-weights]\n  \
     [--set-input NAME=FLOAT]... [--state IN=OUT]... [--name NAME]\n  \
     [--summary] [--dump-json]"
}

pub fn parse(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        input: String::new(),
        output: None,
        ctx: Ctx::default(),
        name: None,
        summary: false,
        dump_json: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => opts.output = Some(it.next().ok_or("-o 값 필요")?.clone()),
            "--size" => {
                let v = it.next().ok_or("--size 값 필요 (WxH)")?;
                let (w, h) = v.split_once('x').ok_or("--size는 WxH 형식")?;
                opts.ctx.size = Some((
                    w.parse().map_err(|_| "너비 파싱 실패")?,
                    h.parse().map_err(|_| "높이 파싱 실패")?,
                ));
            }
            "--set-input" => {
                let v = it.next().ok_or("--set-input 값 필요 (NAME=FLOAT)")?;
                let (n, f) = v.split_once('=').ok_or("--set-input은 NAME=FLOAT")?;
                opts.ctx
                    .set_inputs
                    .push((n.to_string(), f.parse().map_err(|_| "float 파싱 실패")?));
            }
            "--state" => {
                let v = it.next().ok_or("--state 값 필요 (IN=OUT)")?;
                let (i, o) = v.split_once('=').ok_or("--state는 IN=OUT")?;
                opts.ctx.states.push((i.to_string(), o.to_string()));
            }
            "--fp16" => opts.ctx.fp16 = true,
            "--fp16-weights" => opts.ctx.fp16_weights = true,
            "--strip-refiner" => opts.ctx.strip_refiner = true,
            "--name" => opts.name = Some(it.next().ok_or("--name 값 필요")?.clone()),
            "--summary" => opts.summary = true,
            "--dump-json" => opts.dump_json = true,
            other if !other.starts_with('-') && opts.input.is_empty() => {
                opts.input = other.to_string()
            }
            other => return Err(format!("알 수 없는 인자: {other}")),
        }
    }
    if opts.input.is_empty() {
        return Err("입력 .onnx 필요".into());
    }
    Ok(opts)
}
