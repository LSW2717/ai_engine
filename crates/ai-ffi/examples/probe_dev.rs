//! 기기에서 GPU 프로브 시퀀스를 앱 없이 실행하는 진단 바이너리.
//!
//! VbGpuProbeService.probe() 와 동일한 순서로 부르고, 각 단계 결과와 rust 로그를
//! stderr 로 뱉는다. adb push 해서 `adb shell /data/local/tmp/probe_dev <seg.sw>`.
//! (앱 프로세스가 아니라 로거 없는 릴리스 .so 로도 원인을 볼 수 있다.)

use std::ffi::CString;

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Info).init();

    let seg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: probe_dev <seg_model.sw>");
        std::process::exit(2);
    });
    eprintln!("[probe_dev] seg={seg}");

    let rc = unsafe { ai_ffi::set_video_stream_info(seg.as_ptr(), seg.len()) };
    eprintln!("[probe_dev] set_video_stream_info -> {rc:?}");
    if rc as i32 != 0 {
        std::process::exit(1);
    }

    let cfg = CString::new(r#"{"blur":0.5}"#).unwrap();
    let rc = unsafe { ai_ffi::update_effects_config(cfg.as_ptr()) };
    eprintln!("[probe_dev] update_effects_config -> {rc:?}");
    if rc as i32 != 0 {
        std::process::exit(1);
    }

    let (w, h) = (128usize, 96usize);
    let (sy, su, sv) = (128usize, 64usize, 64usize);
    let mut y = vec![128u8; sy * h];
    let mut u = vec![128u8; su * h.div_ceil(2)];
    let mut v = vec![128u8; sv * h.div_ceil(2)];
    let rc = unsafe {
        ai_ffi::render_mask(
            y.as_mut_ptr(),
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            w as i32,
            h as i32,
            sy as i32,
            su as i32,
            sv as i32,
        )
    };
    eprintln!("[probe_dev] render_mask -> {rc:?}");
    eprintln!("[probe_dev] verdict ok={}", rc as i32 == 0);
}
