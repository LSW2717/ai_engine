//! 프레임타임 링버퍼 — GPU/CPU 세그멘터가 공유하는 강등 판정 입력.
//!
//! 평균이 아니라 **p90**을 쓴다: 평균은 가끔 튀는 프레임을 감춘다. 강등 기준은
//! "프레임 예산"이 아니라 "마스크 갱신율 하한"이어야 한다 — 벽시계엔 GPU 큐
//! 대기와 이벤트루프 대기가 섞이기 때문이다 (v-ai의 66ms/2윈도우 규칙과 동일 근거).

use std::collections::VecDeque;

/// 프레임타임 분포 — 런타임 강등 판정의 입력
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub frames: u32,
    pub p50_ms: f32,
    pub p90_ms: f32,
    pub last_ms: f32,
}

/// 통계 창 길이 — 30fps에서 4초
const WINDOW: usize = 120;

#[derive(Default)]
pub struct FrameClock {
    times: VecDeque<f32>,
    frames: u32,
}

impl FrameClock {
    pub fn new() -> Self {
        Self { times: VecDeque::with_capacity(WINDOW), frames: 0 }
    }

    pub fn record(&mut self, ms: f32) {
        if self.times.len() == WINDOW {
            self.times.pop_front();
        }
        self.times.push_back(ms);
        self.frames = self.frames.wrapping_add(1);
    }

    pub fn stats(&self) -> Stats {
        if self.times.is_empty() {
            return Stats::default();
        }
        let mut v: Vec<f32> = self.times.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f32| v[((v.len() as f32 - 1.0) * q).round() as usize];
        Stats {
            frames: self.frames,
            p50_ms: pick(0.5),
            p90_ms: pick(0.9),
            last_ms: *self.times.back().unwrap(),
        }
    }
}
