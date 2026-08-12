//! onnx.proto(벤더링, onnx 1.17) → prost 코드 생성.
//! protox가 순수 Rust로 proto를 컴파일하므로 시스템 protoc이 필요 없다.
//! 생성 코드가 문제되면 OUT_DIR의 onnx.rs를 저장소에 체크인하는 폴백도 가능.

fn main() {
    let fds = protox::compile(["proto/onnx.proto"], ["proto/"])
        .expect("onnx.proto 컴파일 실패");
    prost_build::Config::new()
        .compile_fds(fds)
        .expect("prost 코드 생성 실패");
    println!("cargo:rerun-if-changed=proto/onnx.proto");
}
