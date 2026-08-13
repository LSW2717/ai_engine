//! 파이프라인 캐시 키 덤프 (진단용, #[ignore])
//!
//! 콜드 로드는 전부 파이프라인 컴파일이다 (브라우저 실측: fetch 25ms, 컴파일 2000ms,
//! 개당 ~23.5ms). shape을 셰이더 상수로 굽는 설계라 고유 파이프라인 수가 곧 로드 시간.
//! "순수 인덱스 상수(M/OW/IH/IW)만 다른" 키들을 묶으면 몇 개까지 줄어드는지 본다.

use ai_convert::onnx::import::import;
use ai_convert::passes::{run_full, Ctx};
use ai_convert::plan::lower::lower;
use ai_core::format::SwModel;
use ai_gpu::GpuContext;
use ai_gpu_runtime::lowering;
use std::collections::BTreeMap;

#[test]
#[ignore]
fn diag_pipelines() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/rvm_fixed_144x256.onnx");
    let ctx = GpuContext::new_blocking().unwrap();
    let mut g = import(&std::fs::read(&path).unwrap()).unwrap().graph;
    let cctx = Ctx {
        size: Some((256, 144)),
        states: (1..=4).map(|i| (format!("r{i}"), format!("r{i}o"))).collect(),
        ..Default::default()
    };
    run_full(&mut g, &cctx).unwrap();
    let (sw, blob) = lower(&g, &cctx, "rvm").unwrap();
    let container = sw.write_container(&blob).unwrap();
    let (sw, _) = SwModel::parse_container(&container).unwrap();

    let mut keys: BTreeMap<String, usize> = BTreeMap::new();
    for op in &sw.ops {
        let lo = match lowering::lower_op(&sw, op, &|t| t, &Default::default()) {
            Ok(lo) => lo,
            Err(_) => continue,
        };
        *keys.entry(lo.spec.cache_key(&ctx.caps)).or_default() += 1;
    }
    println!("고유 파이프라인 {}개 (op {}개)\n", keys.len(), sw.ops.len());

    // 커널 계열별 집계 — 어디에 고유 키가 몰려 있나
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for k in keys.keys() {
        let kind = k.split_whitespace().next().unwrap_or("?");
        *by_kind.entry(kind).or_default() += 1;
    }
    for (kind, n) in &by_kind {
        println!("{kind:>14}: 고유 {n}개");
    }

    // 반사실 집계: 특정 shape 토큰을 셰이더 상수에서 uniform으로 내리면
    // 키가 몇 개로 합쳐지는가. 브라우저 콜드 컴파일이 개당 ~23.5ms 균일이므로
    // (줄어든 개수 × 23.5ms) = 콜드 로드 절감액이다.
    let strip = |k: &str, drop: &[&str]| -> String {
        k.split_whitespace()
            .filter(|tok| {
                // "M144", "KG32", "c120", "18x32" 같은 shape 토큰 제거
                !drop.iter().any(|d| {
                    if *d == "HW" {
                        tok.contains('x') && tok.chars().next().is_some_and(|c| c.is_ascii_digit())
                    } else {
                        tok.starts_with(d)
                            && tok[d.len()..].chars().all(|c| c.is_ascii_digit() || c == '-')
                            && tok.len() > d.len()
                    }
                })
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let count = |drop: &[&str]| -> usize {
        keys.keys().map(|k| strip(k, drop)).collect::<std::collections::HashSet<_>>().len()
    };
    let base = keys.len();
    println!("\n== shape 상수를 uniform으로 내렸을 때 고유 파이프라인 수 ==");
    for (label, drop) in [
        ("현재", vec![]),
        ("해상도(HxW)만", vec!["HW"]),
        ("채널 c만", vec!["c"]),
        ("M/KG/NG만", vec!["M", "KG", "NG"]),
        ("해상도+채널", vec!["HW", "c"]),
        ("전부(HW+c+M/KG/NG)", vec!["HW", "c", "M", "KG", "NG"]),
    ] {
        let n = count(&drop);
        println!(
            "{label:>22}: {n:3}개  (-{:2}, 콜드 -{:.0}ms)",
            base - n,
            (base - n) as f64 * 23.5
        );
    }

    println!("\n계열별 (현재 → 전부 uniform):");
    let mut kinds: BTreeMap<&str, (usize, std::collections::HashSet<String>)> = BTreeMap::new();
    for k in keys.keys() {
        let kind = k.split_whitespace().next().unwrap_or("?");
        let e = kinds.entry(kind).or_default();
        e.0 += 1;
        e.1.insert(strip(k, &["HW", "c", "M", "KG", "NG"]));
    }
    for (kind, (now, merged)) in &kinds {
        println!("{kind:>14}: {now:3} → {:3}", merged.len());
    }
}
