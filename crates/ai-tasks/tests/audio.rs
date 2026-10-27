//! 오디오(fastenhancer) 게이트 —
//!  ① 그래프 오라클: 미니 실행기 vs ORT (prep --export가 덤프한 시드 입출력)
//!  ② e2e SNR: Enhancer 전체 체인 vs 공개 wav2wav ONNX 출력 (vcx-noise 픽스처)
//!  ③ hop 벤치: 실시간 예산(48k hop = 10.67ms) 대비 여유 확인
//!  ④ RNNoise(기본 모드): 정상 잡음 감쇄 + 프레임 예산
//!  ⑤ Silero VAD: 그래프 오라클 + 원본 ONNX prob 열 대조(16k) + 48k 리샘플 경로 + 잡음 + 예산
//!
//! 자산: make convert-fastenhancer (models/fastenhancer/) · make convert-silero
//! (models/silero/). 없으면 스킵.
//! RNNoise는 가중치 내장이라 ④는 자산 없이 돈다.

use std::path::Path;

use ai_tasks::features::audio::graph::FeGraph;
use ai_tasks::features::audio::ops::Tens;
use ai_tasks::features::audio::{Denoiser, Enhancer, Vad};

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

fn oracle_one(root: &Path, dir: &str) {
    let d = root.join(dir);
    if !d.join("graph.json").exists() {
        eprintln!("{dir} 없음 — 스킵 (make convert-fastenhancer / convert-silero)");
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
    oracle_one(base(), "fe48");
}

#[test]
fn graph_oracle_fe16() {
    oracle_one(base(), "fe16");
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

/// 48k 샘플을 프레임 단위로 흘려 (입력 에너지, 출력 에너지, ms/frame).
/// 에너지는 첫 1초(100프레임 — 적응 구간)를 빼고 잰다.
fn run_denoiser(d: &mut Denoiser, x: &[f32]) -> (f64, f64, f64) {
    let n = d.frame_len();
    let mut out = vec![0f32; n];
    let (mut ein, mut eout) = (0f64, 0f64);
    let t0 = std::time::Instant::now();
    for (i, seg) in x.chunks_exact(n).enumerate() {
        d.process_frame(seg, &mut out);
        if i < 100 {
            continue;
        }
        ein += seg.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        eout += out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / (x.len() / n) as f64;
    (ein, eout, ms)
}

/// RNNoise: −30dBFS 백색 잡음 3초를 6dB 넘게 깎아야.
/// 실측(2026-10-26, M 시리즈 네이티브 release, 첫 1초 제외): 잡음 −14.2dB, 0.05ms/frame (예산 10ms).
#[test]
fn denoiser_rnnoise() {
    let mut d = Denoiser::new();
    assert_eq!(d.frame_len(), 480);

    let mut seed = 12345u32;
    let noise: Vec<f32> = (0..48_000 * 3)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 0.0316
        })
        .collect();
    let (ein, eout, ms) = run_denoiser(&mut d, &noise);
    let atten = 10.0 * (eout / ein.max(1e-30)).log10();
    eprintln!("RNNoise 잡음 감쇄 {atten:.1} dB, {ms:.3} ms/frame");
    assert!(atten < -6.0, "백색 잡음 감쇄 {atten:.1} dB — 6dB 미만");
    assert!(ms < 2.0, "프레임 {ms:.2}ms — 기본 모드 예산(10ms의 20%) 초과");
}

// ───────── ⑤ Silero VAD ─────────

fn silero() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/silero"))
}

fn load_vad(sample_rate: u32) -> Option<Vad> {
    let d = silero().join("vad16");
    if !d.join("graph.json").exists() {
        eprintln!("models/silero 없음 — 스킵 (make convert-silero)");
        return None;
    }
    Some(
        Vad::new(
            &std::fs::read(d.join("graph.json")).unwrap(),
            &std::fs::read(d.join("weights.bin")).unwrap(),
            sample_rate,
        )
        .unwrap(),
    )
}

/// 10ms 조각으로 흘려 청크(32ms)별 prob 열과 ms/청크를 돌려준다 — 조각 < 청크라 호출당 최대 1개.
fn run_vad(v: &mut Vad, x: &[f32], piece: usize) -> (Vec<f32>, f64) {
    let mut probs = Vec::new();
    let t0 = std::time::Instant::now();
    for seg in x.chunks(piece) {
        if let Some(p) = v.process(seg).unwrap() {
            assert!((0.0..=1.0).contains(&p), "prob 범위 밖: {p}");
            probs.push(p);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / probs.len().max(1) as f64;
    (probs, ms)
}

#[test]
fn graph_oracle_silero() {
    oracle_one(silero(), "vad16");
}

/// 16k: 같은 입력열(한국어 음성 6.7초)에서 원본 ONNX + 공식 래퍼 루프의 prob 열과 1e-4 일치.
/// 실측(2026-10-26, M 시리즈 네이티브 release): 208청크 mean 0.928, max|Δ| 4.4e-6, 0.65ms/청크.
#[test]
fn silero_matches_onnx_16k() {
    let Some(mut v) = load_vad(16_000) else { return };
    let input = read_f32(&silero().join("in_16k.f32"));
    let reference = read_f32(&silero().join("ref_probs_16k.f32"));
    let (probs, ms) = run_vad(&mut v, &input, 160);
    assert_eq!(probs.len(), reference.len(), "청크 수");
    let max_err = probs.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean = probs.iter().sum::<f32>() / probs.len() as f32;
    eprintln!("Silero 16k: {} 청크, mean {mean:.3}, max|Δ| {max_err:.2e}, {ms:.3} ms/청크", probs.len());
    assert!(max_err < 1e-4, "원본과 불일치 {max_err}");
    assert!(mean > 0.8, "음성 픽스처 평균 {mean:.2} — 0.8 미만");

    // reset 후 같은 결과 (상태가 남지 않는다)
    v.reset();
    let (again, _) = run_vad(&mut v, &input, 160);
    assert_eq!(again, probs);
}

/// 48k(평소 캡처 레이트): 같은 음성의 48k 판을 리샘플 경로로 — 16k 원본 prob 열과
/// 판정(≥0.5)이 95% 넘게 같고 평균 차가 작아야 한다(리샘플러 차이만큼만 어긋난다).
/// 실측(2026-10-26): 판정 일치 100%, 평균 |Δ| 0.003.
#[test]
fn silero_48k_resampled() {
    let Some(mut v) = load_vad(48_000) else { return };
    let input: Vec<f32> = std::fs::read(silero().join("in_48k.s16"))
        .unwrap()
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    let reference = read_f32(&silero().join("ref_probs_16k.f32"));
    let (probs, ms) = run_vad(&mut v, &input, 480);
    let n = probs.len().min(reference.len());
    assert!(n + 2 >= reference.len(), "청크 수 {} vs {}", probs.len(), reference.len());
    let agree = (0..n).filter(|&i| (probs[i] >= 0.5) == (reference[i] >= 0.5)).count();
    let mad = (0..n).map(|i| (probs[i] - reference[i]).abs()).sum::<f32>() / n as f32;
    eprintln!(
        "Silero 48k→16k: {n} 청크, 판정 일치 {:.1}%, 평균 |Δ| {mad:.3}, {ms:.3} ms/청크",
        agree as f32 / n as f32 * 100.0
    );
    assert!(agree * 100 >= n * 95, "판정 일치 {agree}/{n}");
    assert!(mad < 0.05, "평균 |Δ| {mad:.3}");
}

/// 정상 잡음(백색, −30dBFS)과 무음은 음성이 아니다 — 48k 입력, 실시간 예산(32ms) 대비 여유도 본다.
/// 실측(2026-10-26): 잡음 mean 0.009, ≥0.5 0/93 청크, 0.67ms/청크(리샘플 포함).
#[test]
fn silero_noise_and_budget() {
    let Some(mut v) = load_vad(48_000) else { return };
    let mut seed = 12345u32;
    let noise: Vec<f32> = (0..48_000 * 3)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 0.0316
        })
        .collect();
    let (probs, ms) = run_vad(&mut v, &noise, 480);
    let mean = probs.iter().sum::<f32>() / probs.len() as f32;
    let over = probs.iter().filter(|&&p| p >= 0.5).count();
    eprintln!("Silero 잡음: mean {mean:.3}, ≥0.5 {over}/{} 청크, {ms:.3} ms/청크", probs.len());
    assert!(mean < 0.1, "잡음 평균 {mean:.2}");
    assert!(ms < 5.0, "청크 {ms:.2}ms — 예산 32ms 의 15% 초과");

    v.reset();
    let (silent, _) = run_vad(&mut v, &vec![0.0; 48_000], 480);
    assert!(silent.iter().all(|&p| p < 0.1), "무음에서 음성 판정");
}
