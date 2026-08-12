//! 풀링 속성. GlobalAveragePool은 별도 타입 없이 커널/레퍼런스 함수로 직접 표현한다.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvgPool2d {
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    /// [top, left, bottom, right] — 패딩 영역은 평균 분모에서 제외하지 않음(count_include_pad=false 아님).
    /// RVM의 AveragePool은 k==s, pad=0이므로 Phase 1 커널은 그 경로만 최적화한다.
    pub pad: [u32; 4],
}

impl AvgPool2d {
    pub fn out_hw(&self, ih: u32, iw: u32) -> (u32, u32) {
        let oh = (ih + self.pad[0] + self.pad[2] - self.kh) / self.sh + 1;
        let ow = (iw + self.pad[1] + self.pad[3] - self.kw) / self.sw + 1;
        (oh, ow)
    }
}
