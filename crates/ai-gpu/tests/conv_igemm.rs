//! 일반 conv(implicit GEMM) GPU vs CPU — direct/tiled 변형, 비대칭 pad, residual.

use ai_gpu::{testsuite, GpuContext};

#[test]
fn conv_igemm_gpu_matches_cpu() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let results = pollster::block_on(testsuite::run_conv_igemm(&ctx)).unwrap();
    let mut failed = 0;
    for r in &results {
        if !r.passed {
            failed += 1;
            eprintln!("FAIL {} (max_err {:.3e}, tol {:.1e})", r.name, r.max_err, r.tol);
        }
    }
    println!("conv_igemm: {}/{} 통과", results.len() - failed, results.len());
    assert_eq!(failed, 0, "{failed}개 케이스 실패");
}
