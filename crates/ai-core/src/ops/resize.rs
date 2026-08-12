//! bilinear 리사이즈 속성.

/// ONNX coordinate_transformation_mode의 부분집합 (RVM/세그 모델이 쓰는 두 가지)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordMode {
    /// src = (dst + 0.5) * (in/out) - 0.5
    HalfPixel,
    /// src = dst * (in/out)
    Asymmetric,
}

impl CoordMode {
    pub fn tag(self) -> &'static str {
        match self {
            CoordMode::HalfPixel => "half_pixel",
            CoordMode::Asymmetric => "asymmetric",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizeBilinear {
    pub oh: u32,
    pub ow: u32,
    pub mode: CoordMode,
}
