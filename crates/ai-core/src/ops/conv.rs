//! 2D 컨볼루션 속성. groups로 일반/pointwise/depthwise를 모두 표현한다.

use crate::activation::Activation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2d {
    pub cin: u32,
    pub cout: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    /// [top, left, bottom, right]
    pub pad: [u32; 4],
    /// 등방 dilation (RVM 인코더 마지막 스테이지가 d=2 사용)
    pub dil: u32,
    /// 1 = 일반/pointwise, cin(=cout) = depthwise
    pub groups: u32,
    /// 에필로그 융합 활성화 (bias → act → residual 순으로 적용)
    pub act: Activation,
}

impl Conv2d {
    pub fn pointwise(cin: u32, cout: u32, act: Activation) -> Self {
        Self { cin, cout, kh: 1, kw: 1, sh: 1, sw: 1, pad: [0; 4], dil: 1, groups: 1, act }
    }

    pub fn depthwise(c: u32, k: u32, stride: u32, act: Activation) -> Self {
        let p = (k - 1) / 2;
        Self {
            cin: c,
            cout: c,
            kh: k,
            kw: k,
            sh: stride,
            sw: stride,
            pad: [p; 4],
            dil: 1,
            groups: c,
            act,
        }
    }

    pub fn is_depthwise(&self) -> bool {
        self.groups == self.cin && self.cin == self.cout && self.groups > 1
    }

    /// 출력 공간 크기 (유효 커널 = dil*(k-1)+1)
    pub fn out_hw(&self, ih: u32, iw: u32) -> (u32, u32) {
        let ekh = self.dil * (self.kh - 1) + 1;
        let ekw = self.dil * (self.kw - 1) + 1;
        let oh = (ih + self.pad[0] + self.pad[2] - ekh) / self.sh + 1;
        let ow = (iw + self.pad[1] + self.pad[3] - ekw) / self.sw + 1;
        (oh, ow)
    }
}
