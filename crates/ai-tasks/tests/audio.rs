//! 오디오(fastenhancer) 게이트 —
//!  ① 그래프 오라클: 미니 실행기 vs ORT (prep --export가 덤프한 시드 입출력)
//!  ② e2e SNR: Enhancer 전체 체인 vs 공개 wav2wav ONNX 출력 (vcx-noise 픽스처)
//!  ③ hop 벤치: 실시간 예산(48k hop = 10.67ms) 대비 여유 확인
//!
//! 자산: make convert-fastenhancer (models/fastenhancer/). 없으면 스킵.

use std::path::Path;

use ai_tasks::features::audio::graph::FeGraph;
use ai_tasks::features::audio::ops::Tens;
use ai_tasks::features::audio::Enhancer;

fn base() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/fastenhancer"))
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[derive(serde::Deserialize)]
struct IoMeta {
    inputs: Vec<IoDecl>,
    outputs: Vec<IoDecl>,
}
#[derive(serde::Deserialize)]
struct IoDecl {
    name: String,
    shape: Vec<usize>,
}

fn oracle_one(dir: &str) {
    let d = base().join(dir);
    if !d.join("graph.json").exists() {
        eprintln!("{dir} 없음 — 스킵 (make convert-fastenhancer)");
        return;
    }
    let g = FeGraph::load(
        &std::fs::read(d.join("graph.json")).unwrap(),
        &std::fs::read(d.join("weights.bin")).unwrap(),
    )
    .unwrap();
    let meta: IoMeta =
        serde_json::from_slice(&std::fs::read(d.join("oracle_io.json")).unwrap()).unwrap();
    let raw = read_f32(&d.join("oracle_io.bin"));
    let mut off = 0usize;
    let mut take = |shape: &[usize]| -> Vec<f32> {
        let n: usize = shape.iter().product();
        let v = raw[off..off + n].to_vec();
        off += n;
        v
    };
    let inputs: Vec<Tens> =
        meta.inputs.iter().map(|i| Tens::new(i.shape.clone(), take(&i.shape))).collect();
    let expected: Vec<Vec<f32>> = meta.outputs.iter().map(|o| take(&o.shape)).collect();

    let outs = g.run(inputs).expect("그래프 실행");
    for ((o, e), m) in outs.iter().zip(&expected).zip(&meta.outputs) {
        let max_err = o
            .data
            .iter()
            .zip(e)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{dir}/{}: max_err {max_err:.2e}", m.name);
        assert!(max_err < 1e-4, "{dir}/{}: {max_err}", m.name);
    }
}

#[test]
fn graph_oracle_fe48() {
    oracle_one("fe48");
}

#[test]
fn graph_oracle_fe16() {
    oracle_one("fe16");
}

/// e2e: 3초 실오디오를 hop 단위로 흘려 공개 wav2wav ONNX 출력과 SNR 비교.
/// vcx-noise(ncnn)는 같은 픽스처에서 ~49dB — 우리는 같은 가중치·같은 수식이라
/// FFT 반올림 차뿐이어야 한다. 게이트 45dB.
#[test]
fn e2e_snr_48k() {
    let d = base();
    let (inp, refp) = (d.join("in_48k.f32"), d.join("ref_48k_wav2wav.f32"));
    if !inp.exists() || !d.join("fe48/graph.json").exists() {
        eprintln!("픽스처 없음 — 스킵");
        return;
    }
    let input = read_f32(&inp);
    let reference = read_f32(&refp);
    let mut e = Enhancer::new(
        &std::fs::read(d.join("fe48/graph.json")).unwrap(),
        &std::fs::read(d.join("fe48/weights.bin")).unwrap(),
    )
    .unwrap();
    let hop = e.frame_len();
    assert_eq!(hop, 512);
    let hops = reference.len() / hop;
    let mut out = vec![0f32; hops * hop];
    let t0 = std::time::Instant::now();
    for h in 0..hops {
        let seg = &input[h * hop..(h + 1) * hop];
        e.process_frame(seg, &mut out[h * hop..(h + 1) * hop]).unwrap();
    }
    let per_hop_ms = t0.elapsed().as_secs_f64() * 1e3 / hops as f64;

    let (mut se, mut sr) = (0f64, 0f64);
    for i in 0..hops * hop {
        let r = reference[i] as f64;
        sr += r * r;
        let d = (out[i] - reference[i]) as f64;
        se += d * d;
    }
    let snr = 10.0 * (sr / se.max(1e-30)).log10();
    eprintln!(
        "e2e SNR vs wav2wav: {snr:.1} dB, {per_hop_ms:.3} ms/hop (예산 10.67ms, {hops} hops)"
    );
    assert!(snr >= 45.0, "SNR {snr:.1} dB < 45");
    assert!(per_hop_ms < 5.0, "hop {per_hop_ms:.2}ms — 실시간 예산의 절반 초과");
}

/// op별 시간 진단 (수동 실행: cargo test -p ai-tasks --release --test audio profile_ops -- --ignored --nocapture)
#[test]
#[ignore]
fn profile_ops() {
    let d = base().join("fe48");
    let g = FeGraph::load(
        &std::fs::read(d.join("graph.json")).unwrap(),
        &std::fs::read(d.join("weights.bin")).unwrap(),
    )
    .unwrap();
    let inputs: Vec<Tens> =
        g.inputs.iter().map(|(_, s)| Tens::zeros(s.clone())).collect();
    // 워밍업 + 20회 누적
    let mut acc: Vec<(String, f64)> = Vec::new();
    for it in 0..21 {
        let (_, prof) = g.run_profiled(inputs.clone()).unwrap();
        if it == 0 {
            continue;
        }
        if acc.is_empty() {
            acc = prof;
        } else {
            for (a, p) in acc.iter_mut().zip(&prof) {
                debug_assert_eq!(a.0, p.0);
                a.1 += p.1;
            }
        }
    }
    let total: f64 = acc.iter().map(|(_, v)| v).sum();
    for (op, ms) in &acc {
        eprintln!("{op:>14}: {:.3} ms/hop ({:.0}%)", ms / 20.0, ms / total * 100.0);
    }
    eprintln!("{:>14}: {:.3} ms/hop", "TOTAL", total / 20.0);
}

/// 진단: 임의 입력/참조로 SNR (env AI_AUDIO_IN / AI_AUDIO_REF)
#[test]
#[ignore]
fn e2e_snr_custom() {
    let inp = read_f32(Path::new(&std::env::var("AI_AUDIO_IN").unwrap()));
    let reference = read_f32(Path::new(&std::env::var("AI_AUDIO_REF").unwrap()));
    let d = base();
    let mut e = Enhancer::new(
        &std::fs::read(d.join("fe48/graph.json")).unwrap(),
        &std::fs::read(d.join("fe48/weights.bin")).unwrap(),
    )
    .unwrap();
    let hop = e.frame_len();
    let hops = reference.len() / hop;
    let mut out = vec![0f32; hops * hop];
    for h in 0..hops {
        e.process_frame(&inp[h * hop..(h + 1) * hop], &mut out[h * hop..(h + 1) * hop])
            .unwrap();
    }
    let sr = 48000usize;
    for sec in 0..hops * hop / sr + 1 {
        let a = sec * sr;
        let b = ((sec + 1) * sr).min(hops * hop);
        if a >= b {
            break;
        }
        let (mut se, mut sq) = (0f64, 0f64);
        for i in a..b {
            sq += (reference[i] as f64).powi(2);
            se += ((out[i] - reference[i]) as f64).powi(2);
        }
        eprintln!("  {sec}s: {:.1}dB", 10.0 * (sq.max(1e-30) / se.max(1e-30)).log10());
    }
    let (mut se, mut sq) = (0f64, 0f64);
    for i in 0..hops * hop {
        sq += (reference[i] as f64).powi(2);
        se += ((out[i] - reference[i]) as f64).powi(2);
    }
    eprintln!("전체 SNR: {:.1}dB", 10.0 * (sq / se.max(1e-30)).log10());
}
