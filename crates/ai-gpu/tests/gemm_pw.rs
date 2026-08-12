//! pointwise GEMM 커널 GPU vs CPU — small/tiled 변형, 에필로그 융합 그리드.

use ai_gpu::{testsuite, GpuContext};

#[test]
fn gemm_pw_gpu_matches_cpu() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let results = pollster::block_on(testsuite::run_gemm_pw(&ctx)).unwrap();
    let mut failed = 0;
    for r in &results {
        if !r.passed {
            failed += 1;
            eprintln!("FAIL {} (max_err {:.3e}, tol {:.1e})", r.name, r.max_err, r.tol);
        }
    }
    println!("gemm_pw: {}/{} 통과", results.len() - failed, results.len());
    assert_eq!(failed, 0, "{failed}개 케이스 실패");
}
