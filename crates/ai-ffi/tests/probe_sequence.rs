//! VbGpuProbeService.probe() 와 동일한 호출 시퀀스를 호스트에서 재현한다.
//! 안드로이드 프로브가 ok=false 를 돌려주는 원인이 로직인지 드라이버인지 가른다.

use std::ffi::CString;

#[test]
fn probe_sequence_matches_android() {
    let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Info).try_init();

    let seg = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/models/rvm_256x144.sw");
    assert!(std::path::Path::new(seg).exists(), "모델 없음: {seg}");

    let rc = unsafe { ai_ffi::set_video_stream_info(seg.as_ptr(), seg.len()) };
    println!("set_video_stream_info rc={rc:?}");
    assert_eq!(rc as i32, 0, "set_video_stream_info 실패");

    let cfg = CString::new(r#"{"blur":0.5}"#).unwrap();
    let rc = unsafe { ai_ffi::update_effects_config(cfg.as_ptr()) };
    println!("update_effects_config rc={rc:?}");
    assert_eq!(rc as i32, 0, "update_effects_config 실패");

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
    println!("render_mask rc={rc:?}");
    assert_eq!(rc as i32, 0, "render_mask 실패 — 프로브 ok=false 의 원인");
}
