//! 형태 연산(Concat/Split/Slice) — Phase 2 이음새.
//!
//! WebGL2 엔진의 교훈대로 Split은 뷰(제로카피), Concat은 가능하면 소비자에 융합한다.
//! Phase 1 커널은 없으며 타입만 정의한다.

/// 채널 축(C) concat. NHWC-C4에서 소스 채널이 4의 배수면 그룹 오프셋 뷰로 융합 가능.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcatC {
    /// 각 입력의 채널 수
    pub parts: Vec<u32>,
}

/// 채널 축(C) split. 4의 배수 경계면 제로카피 뷰.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitC {
    pub parts: Vec<u32>,
}
