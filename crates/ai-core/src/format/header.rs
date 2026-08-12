//! .sw 컨테이너 바이너리 헤더 — 순수 슬라이스 파싱(파일 I/O 없음, wasm-safe).
//!
//! 레이아웃 (little-endian):
//! ```text
//! 0   4  magic "SW1\0"
//! 4   4  version u32 = 1
//! 8   4  flags u32 = 0 (예약; bit0 = 향후 JSON 압축)
//! 12  4  blob_align u32 = 256
//! 16  8  json_off u64 = 32
//! 24  8  json_len u64
//! 32  …  JSON 그래프 (UTF-8)
//! …      0 패딩 → blob_off = (json_off+json_len).next_multiple_of(blob_align)
//! …      가중치 블롭 (모든 세그먼트 오프셋이 blob_align 배수)
//! ```

use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"SW1\0";
pub const VERSION: u32 = 1;
/// WebGPU min_storage_buffer_offset_alignment 상한(256)에 맞춘 사전 정렬 —
/// 런타임은 블롭 전체를 단일 버퍼로 memcpy하고 오프셋 바인딩만 한다.
pub const BLOB_ALIGN: u32 = 256;
pub const HEADER_LEN: usize = 32;

#[derive(Error, Debug)]
pub enum FormatError {
    #[error(".sw 아님 (매직 불일치)")]
    BadMagic,
    #[error("지원하지 않는 버전 {0} (지원: {VERSION})")]
    BadVersion(u32),
    #[error("컨테이너 절단됨: {0}")]
    Truncated(&'static str),
    #[error("JSON 파싱 실패: {0}")]
    Json(String),
}

/// json + blob → 컨테이너 바이트
pub fn write_container(json: &[u8], blob: &[u8]) -> Vec<u8> {
    let json_off = HEADER_LEN as u64;
    let json_len = json.len() as u64;
    let blob_off = (json_off + json_len).next_multiple_of(BLOB_ALIGN as u64);
    let mut out = Vec::with_capacity(blob_off as usize + blob.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&BLOB_ALIGN.to_le_bytes());
    out.extend_from_slice(&json_off.to_le_bytes());
    out.extend_from_slice(&json_len.to_le_bytes());
    out.extend_from_slice(json);
    out.resize(blob_off as usize, 0);
    out.extend_from_slice(blob);
    out
}

/// 컨테이너 바이트 → (json 슬라이스, blob 슬라이스)
pub fn parse_container(bytes: &[u8]) -> Result<(&[u8], &[u8]), FormatError> {
    if bytes.len() < HEADER_LEN {
        return Err(FormatError::Truncated("헤더"));
    }
    if bytes[0..4] != MAGIC {
        return Err(FormatError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let version = u32_at(4);
    if version != VERSION {
        return Err(FormatError::BadVersion(version));
    }
    let blob_align = u32_at(12) as u64;
    let json_off = u64_at(16);
    let json_len = u64_at(24);
    let json_end = json_off
        .checked_add(json_len)
        .ok_or(FormatError::Truncated("json 범위"))?;
    if json_end > bytes.len() as u64 {
        return Err(FormatError::Truncated("json"));
    }
    let blob_off = json_end.next_multiple_of(blob_align.max(1));
    if blob_off > bytes.len() as u64 {
        return Err(FormatError::Truncated("blob"));
    }
    Ok((
        &bytes[json_off as usize..json_end as usize],
        &bytes[blob_off as usize..],
    ))
}
