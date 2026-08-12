//! 가중치 블롭 빌더 — 256정렬 세그먼트, 콘텐츠 dedup (RVM refiner의 공유 box filter).

use std::collections::HashMap;

use ai_core::format::{WRef, BLOB_ALIGN};

#[derive(Default)]
pub struct BlobBuilder {
    data: Vec<u8>,
    dedup: HashMap<Vec<u8>, WRef>,
}

impl BlobBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// 바이트 추가 (동일 콘텐츠는 기존 세그먼트 재사용)
    pub fn push(&mut self, bytes: &[u8]) -> WRef {
        if let Some(r) = self.dedup.get(bytes) {
            return *r;
        }
        let off = (self.data.len() as u64).next_multiple_of(BLOB_ALIGN as u64);
        self.data.resize(off as usize, 0);
        self.data.extend_from_slice(bytes);
        let r = WRef { off, len: bytes.len() as u64 };
        self.dedup.insert(bytes.to_vec(), r);
        r
    }

    pub fn finish(self) -> Vec<u8> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_and_dedups() {
        let mut b = BlobBuilder::new();
        let r1 = b.push(&[1u8; 100]);
        let r2 = b.push(&[2u8; 50]);
        let r3 = b.push(&[1u8; 100]); // dedup
        assert_eq!(r1.off, 0);
        assert_eq!(r2.off, 256);
        assert_eq!(r3, r1);
        assert_eq!(b.finish().len(), 256 + 50);
    }
}
