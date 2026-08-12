//! Phase 1 종료 시험 — 풀링/리사이즈 커널 + MNv3 inverted-residual+SE 블록 E2E.

use ai_gpu::{testsuite, GpuContext};

#[test]
fn pool_resize_gpu_matches_cpu() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let results = pollster::block_on(testsuite::run_pool_resize(&ctx)).unwrap();
    let mut failed = 0;
    for r in &results {
        if !r.passed {
            failed += 1;
            eprintln!("FAIL {} (max_err {:.3e}, tol {:.1e})", r.name, r.max_err, r.tol);
        }
    }
    println!("pool/resize: {}/{} 통과", results.len() - failed, results.len());
    assert_eq!(failed, 0, "{failed}개 케이스 실패");
}

#[test]
fn mobilenet_block_e2e() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let results = pollster::block_on(testsuite::run_mobilenet_block(&ctx)).unwrap();
    for r in &results {
        println!(
            "{} → {} (max_err {:.3e})",
            r.name,
            if r.passed { "PASS" } else { "FAIL" },
            r.max_err
        );
        assert!(r.passed, "블록 E2E 실패: max_err {:.3e}", r.max_err);
    }
}
