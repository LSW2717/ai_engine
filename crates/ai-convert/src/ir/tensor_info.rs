//! IR 텐서 메타데이터 — shape(NCHW 의미론, lowering 전까지) + dtype + 상수 데이터.

use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OnnxDtype {
    F32,
    F16,
    I64,
    I32,
    Bool,
    Unknown(i32),
}

impl OnnxDtype {
    pub fn byte_size(self) -> Option<usize> {
        match self {
            OnnxDtype::F32 | OnnxDtype::I32 => Some(4),
            OnnxDtype::F16 => Some(2),
            OnnxDtype::I64 => Some(8),
            OnnxDtype::Bool => Some(1),
            OnnxDtype::Unknown(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TensorInfo {
    /// None = 완전 미상. Some 안의 -1 = 심볼릭 차원(미해석).
    pub shape: Option<Vec<i64>>,
    pub dtype: OnnxDtype,
    /// Some = 상수(이니셜라이저/Constant/폴딩 결과), 원시 바이트
    pub data: Option<Arc<Vec<u8>>>,
}

impl Default for OnnxDtype {
    fn default() -> Self {
        OnnxDtype::F32
    }
}

impl TensorInfo {
    pub fn is_const(&self) -> bool {
        self.data.is_some()
    }

    pub fn static_shape(&self) -> Option<&[i64]> {
        match &self.shape {
            Some(s) if s.iter().all(|d| *d >= 0) => Some(s),
            _ => None,
        }
    }

    /// 상수 f32 값들 (dtype F32일 때)
    pub fn as_f32s(&self) -> Option<&[f32]> {
        match (&self.data, self.dtype) {
            (Some(d), OnnxDtype::F32) => Some(bytemuck::cast_slice(d)),
            _ => None,
        }
    }

    /// 상수 i64 값들
    pub fn as_i64s(&self) -> Option<&[i64]> {
        match (&self.data, self.dtype) {
            (Some(d), OnnxDtype::I64) => Some(bytemuck::cast_slice(d)),
            _ => None,
        }
    }

    /// 스칼라 f32 (shape [] 또는 [1])
    pub fn as_scalar_f32(&self) -> Option<f32> {
        let v = self.as_f32s()?;
        if v.len() == 1 { Some(v[0]) } else { None }
    }
}
