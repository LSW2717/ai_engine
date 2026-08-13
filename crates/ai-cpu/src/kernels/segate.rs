//! SE 게이트 — gpool(채널 평균) → FC1(act1) [→ FC2] 를 한 op로.
//! 출력은 [1,1,C] 채널 벡터 (후속 cvec-mul이 게이트로 읽는다).
//!
//! FC 가중치는 로드 시 OIHW 1×1에서 `[cout][cin]` 행-major로 언팩해 둔다 —
//! 채널 수십 규모라 스칼라 gemv로 충분하다.

use ai_core::Activation;

use crate::view::View;

pub struct SeFc {
    /// `[cout][cin]` 행-major
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub act: Activation,
}

impl SeFc {
    fn apply(&self, x: &[f32], out: &mut Vec<f32>) {
        let cin = x.len();
        let cout = self.b.len();
        debug_assert_eq!(self.w.len(), cin * cout);
        out.clear();
        for o in 0..cout {
            let mut acc = self.b[o];
            let row = &self.w[o * cin..(o + 1) * cin];
            for i in 0..cin {
                acc += row[i] * x[i];
            }
            out.push(self.act.apply(acc));
        }
    }
}

/// scratch: (mean 버퍼, fc 중간 버퍼) — 호출자가 재사용해 프레임 중 할당 0
pub fn se_gate(
    input: View,
    px: usize,
    fc1: &SeFc,
    fc2: Option<&SeFc>,
    scratch: &mut (Vec<f32>, Vec<f32>),
    out: &mut [f32],
) {
    let (mean, mid) = scratch;
    mean.clear();
    mean.resize(input.c, 0.0);
    for p in 0..px {
        let b = input.base(p);
        for ch in 0..input.c {
            mean[ch] += input.data[b + ch];
        }
    }
    let inv = 1.0 / px as f32;
    for v in mean.iter_mut() {
        *v *= inv;
    }

    match fc2 {
        Some(f2) => {
            fc1.apply(mean, mid);
            let mut fin = std::mem::take(mean);
            f2.apply(mid, &mut fin);
            out.copy_from_slice(&fin);
            *mean = fin;
        }
        None => {
            fc1.apply(mean, mid);
            out.copy_from_slice(mid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 손계산: 2px c=2 입력 평균 [2, 3], FC1(relu): w=[[1,1],[1,-2]], b=[0,0]
    /// → [5, relu(2-6)=0], FC2(sigmoid 없이 None 경로 확인은 위에서)
    #[test]
    fn hand_computed_fc1_only() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // px0=[1,2], px1=[3,4] → mean=[2,3]
        let fc1 = SeFc {
            w: vec![1.0, 1.0, 1.0, -2.0],
            b: vec![0.0, 0.0],
            act: Activation::Relu,
        };
        let mut out = vec![0f32; 2];
        let mut scratch = (vec![], vec![]);
        se_gate(View::dense(&x, 2), 2, &fc1, None, &mut scratch, &mut out);
        assert_eq!(out, vec![5.0, 0.0]);
    }

    /// FC2 경로: FC1 항등 → FC2 스케일 2배
    #[test]
    fn hand_computed_fc2() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // mean=[2,3]
        let fc1 = SeFc {
            w: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![0.0, 0.0],
            act: Activation::None,
        };
        let fc2 = SeFc {
            w: vec![2.0, 0.0, 0.0, 2.0],
            b: vec![1.0, 1.0],
            act: Activation::None,
        };
        let mut out = vec![0f32; 2];
        let mut scratch = (vec![], vec![]);
        se_gate(View::dense(&x, 2), 2, &fc1, Some(&fc2), &mut scratch, &mut out);
        assert_eq!(out, vec![5.0, 7.0]); // [2*2+1, 3*2+1]
    }
}
