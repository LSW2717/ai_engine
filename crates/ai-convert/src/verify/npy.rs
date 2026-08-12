//! 최소 .npy 리더 — C-order f32만 (오라클 덤프 소비 전용, 외부 의존 없음).

pub fn read_npy_f32(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f32>), String> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("npy 매직 불일치".into());
    }
    let (major, _minor) = (bytes[6], bytes[7]);
    let (header_len, data_start) = if major == 1 {
        let l = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (l, 10 + l)
    } else {
        let l = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (l, 12 + l)
    };
    let header = std::str::from_utf8(&bytes[data_start - header_len..data_start])
        .map_err(|_| "헤더 utf8 아님")?;
    if !header.contains("'<f4'") {
        return Err(format!("f32 아님: {header}"));
    }
    if header.contains("'fortran_order': True") {
        return Err("fortran order 미지원".into());
    }
    let shape_str = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("shape 파싱 실패")?;
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let elems: usize = shape.iter().product::<usize>().max(1);
    let data = &bytes[data_start..];
    if data.len() < elems * 4 {
        return Err("데이터 부족".into());
    }
    let vals: Vec<f32> = data[..elems * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((shape, vals))
}

/// NCHW(오라클) → NHWC(우리 논리 레이아웃)
pub fn nchw_to_nhwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ch] = data[(ch * h + y) * w + x];
            }
        }
    }
    out
}

/// NHWC(우리) → NCHW(오라클 비교용)
pub fn nhwc_to_nchw(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(ch * h + y) * w + x] = data[(y * w + x) * c + ch];
            }
        }
    }
    out
}
