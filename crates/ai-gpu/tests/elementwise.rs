//! elementwise 커널 GPU vs CPU E2E — codegen→컴파일→바인딩→디스패치→리드백 체인 최초 증명.

use ai_gpu::{testsuite, GpuContext};

#[test]
fn elementwise_gpu_matches_cpu() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: GPU 없음 ({e})");
            return;
        }
    };
    let mut results = pollster::block_on(testsuite::run_elementwise(&ctx)).unwrap();
    results.extend(pollster::block_on(testsuite::run_elementwise_arena(&ctx)).unwrap());
    let mut failed = 0;
    for r in &results {
        if !r.passed {
            failed += 1;
            eprintln!("FAIL {} (max_err {:.3e}, tol {:.1e})", r.name, r.max_err, r.tol);
        }
    }
    println!("elementwise: {}/{} 통과", results.len() - failed, results.len());
    assert_eq!(failed, 0, "{failed}개 케이스 실패");
}
