//! 텐서 기술자와 NHWC-C4 레이아웃 수학.
//!
//! 레이아웃: 채널을 4개씩 vec4로 묶고(C4), 공간은 NHWC 순서.
//! vec4 단위 선형 인덱스는 `idx(h, w, cg) = (h*W + w) * cg_count + cg`.
//! C만 4의 배수로 제로패딩되며 W/H는 무제약이다(구 WebGL2 엔진의 W%4 제약 제거).
//! 행렬로 보면 row-major `[H*W, cg_count]`이므로 1×1 conv GEMM이 재배열 없이 동작한다.

/// 텐서 원소의 저장 정밀도. 커널 codegen의 정식 축이며 캐시 키에 포함된다.
/// 누산은 dtype과 무관하게 항상 f32.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    /// f16 스토리지 (셰이더는 `enable f16;` + `vec4<f16>`, 누산 f32)
    F16,
}

impl DType {
    /// vec4 하나가 차지하는 바이트 수
    pub fn vec4_bytes(self) -> u64 {
        match self {
            DType::F32 => 16,
            DType::F16 => 8,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TensorDesc {
    /// 배치. Phase 1에서는 항상 1 (필드는 향후 확장용).
    pub n: u32,
    pub h: u32,
    pub w: u32,
    /// 논리 채널 수 (패딩 전)
    pub c: u32,
    pub dt: DType,
}

impl TensorDesc {
    pub fn new(h: u32, w: u32, c: u32, dt: DType) -> Self {
        Self { n: 1, h, w, c, dt }
    }

    /// 채널 그룹 수 = ceil(C/4)
    pub fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }

    /// 전체 vec4 개수 = H * W * cg
    pub fn vec4_len(&self) -> u64 {
        self.h as u64 * self.w as u64 * self.cg() as u64
    }

    /// 패킹된 바이트 크기
    pub fn size_bytes(&self) -> u64 {
        self.vec4_len() * self.dt.vec4_bytes()
    }

    /// align 배수로 올림한 바이트 크기 (arena 오프셋 정렬용)
    pub fn size_bytes_aligned(&self, align: u64) -> u64 {
        self.size_bytes().div_ceil(align) * align
    }

    /// vec4 단위 선형 인덱스
    pub fn idx(&self, h: u32, w: u32, cg: u32) -> u64 {
        (h as u64 * self.w as u64 + w as u64) * self.cg() as u64 + cg as u64
    }

    /// 논리 원소 수 (패딩 제외)
    pub fn elems(&self) -> usize {
        (self.h * self.w * self.c) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_math() {
        // c=6 → cg=2, 홀수 W 허용
        let d = TensorDesc::new(3, 5, 6, DType::F32);
        assert_eq!(d.cg(), 2);
        assert_eq!(d.vec4_len(), 3 * 5 * 2);
        assert_eq!(d.size_bytes(), 3 * 5 * 2 * 16);
        // idx: (h*W + w)*cg + cg_i
        assert_eq!(d.idx(0, 0, 0), 0);
        assert_eq!(d.idx(0, 0, 1), 1);
        assert_eq!(d.idx(0, 1, 0), 2);
        assert_eq!(d.idx(1, 0, 0), 10);
        assert_eq!(d.idx(2, 4, 1), (2 * 5 + 4) * 2 + 1);
    }

    #[test]
    fn f16_sizes_and_alignment() {
        let d = TensorDesc::new(9, 16, 64, DType::F16);
        assert_eq!(d.cg(), 16);
        assert_eq!(d.size_bytes(), 9 * 16 * 16 * 8);
        // 256 정렬 올림
        let d2 = TensorDesc::new(1, 1, 1, DType::F32);
        assert_eq!(d2.size_bytes(), 16);
        assert_eq!(d2.size_bytes_aligned(256), 256);
    }
}
