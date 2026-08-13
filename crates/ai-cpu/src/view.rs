//! 채널 스트라이드 뷰 — alias(채널 슬라이스)와 concat 융합 파트의 공용 표현.
//!
//! NHWC 밀집 레이아웃에서는 한 픽셀의 채널이 연속이므로, 백킹 텐서의 채널
//! 부분구간은 (기준 오프셋, 픽셀 스트라이드)만으로 복사 없이 읽을 수 있다.
//! 커널은 이 뷰만 알고, alias 해석·슬롯 관리는 plan/exec이 소유한다.

#[derive(Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [f32],
    /// 픽셀 내 시작 채널 (alias cg_off*4)
    pub c_off: usize,
    /// 픽셀 간 스트라이드 (= 백킹 텐서의 채널 수)
    pub stride: usize,
    /// 이 뷰의 논리 채널 수
    pub c: usize,
}

impl<'a> View<'a> {
    pub fn dense(data: &'a [f32], c: usize) -> Self {
        Self { data, c_off: 0, stride: c, c }
    }

    /// 선형 픽셀 인덱스 → data 내 시작 오프셋
    #[inline(always)]
    pub fn base(&self, lin_px: usize) -> usize {
        lin_px * self.stride + self.c_off
    }
}
