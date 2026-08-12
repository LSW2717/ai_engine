//! depthwise conv 커널 GPU vs CPU — k3/k5 × s1/s2 × 에필로그 그리드.

use ai_gpu::{testsuite, GpuContext};

#[test]
fn conv_dw_gpu_matches_cpu() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let results = pollster::block_on(testsuite::run_conv_dw(&ctx)).unwrap();
    let mut failed = 0;
    for r in &results {
        if !r.passed {
            failed += 1;
            eprintln!("FAIL {} (max_err {:.3e}, tol {:.1e})", r.name, r.max_err, r.tol);
        }
    }
    println!("conv_dw: {}/{} 통과", results.len() - failed, results.len());
    assert_eq!(failed, 0, "{failed}개 케이스 실패");
}
