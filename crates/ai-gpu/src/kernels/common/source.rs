//! 입력 소스 기술자 — "이 커널이 읽는 채널 구간이 백킹 버퍼 어디에 있는가".
//!
//! 두 가지를 하나의 개념으로 흡수한다:
//! 1. **concat 융합 파트** — 여러 버퍼를 채널 축으로 이어 붙여 읽는다 (파트마다 SrcView 하나).
//! 2. **채널 뷰(비복사 Split)** — 백킹 텐서의 부분 채널 구간을 그대로 읽는다.
//!    (`stride_c` = 백킹의 채널 수, `off_c` = 시작 채널)
//!
//! 둘 다 결국 "행 stride와 시작 오프셋이 자기 채널 수와 다를 수 있다"는 같은 문제라,
//! 커널 codegen은 `q = pix * stride_cg + off_cg` 한 줄로 양쪽을 처리한다.
//! 뷰를 흡수하면 실체화 복사 디스패치가 통째로 사라진다 (디스패치 하나 = 실측 ~4µs).
//!
//! 채널 오프셋은 항상 4의 배수다 — 변환기가 비정렬 Split은 chcopy(실복사)로 보내고
//! 4배수만 chview(별칭)로 남긴다.

/// 커널 입력 한 파트의 위치. `c`는 이 파트가 기여하는 논리 채널 수.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SrcView {
    /// 이 파트의 논리 채널 수 (0 = 미사용 슬롯)
    pub c: u32,
    /// 백킹 텐서의 채널 수 (= 행 stride). 뷰가 아니면 `c`와 같다.
    pub stride_c: u32,
    /// 백킹 텐서 안에서 이 파트가 시작하는 채널 (4의 배수)
    pub off_c: u32,
}

impl SrcView {
    /// 미사용 슬롯 / "평범한 단일 입력" 표식
    pub const NONE: Self = Self { c: 0, stride_c: 0, off_c: 0 };

    /// 뷰 아님 — 버퍼 전체가 이 파트다
    pub fn plain(c: u32) -> Self {
        Self { c, stride_c: c, off_c: 0 }
    }

    /// 백킹의 `off_c..off_c+c` 구간을 가리키는 뷰
    pub fn view(c: u32, stride_c: u32, off_c: u32) -> Self {
        debug_assert_eq!(off_c % 4, 0, "채널 오프셋은 4배수여야 한다 (chview 규약)");
        debug_assert!(off_c + c <= stride_c, "뷰가 백킹을 벗어남");
        Self { c, stride_c, off_c }
    }

    pub fn is_used(&self) -> bool {
        self.c > 0
    }

    /// 인덱싱이 항등이 아닌가 (stride/offset 보정이 필요한가)
    pub fn is_offset(&self) -> bool {
        self.c > 0 && (self.stride_c != self.c || self.off_c != 0)
    }

    pub fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }

    pub fn stride_cg(&self) -> u32 {
        self.stride_c.div_ceil(4)
    }

    pub fn off_cg(&self) -> u32 {
        self.off_c / 4
    }

    /// 캐시 키 조각 — 평범한 파트는 채널 수만, 뷰는 구간까지 (codegen이 달라진다)
    pub fn key(&self) -> String {
        if self.is_offset() {
            format!("{}@{}/{}", self.c, self.off_c, self.stride_c)
        } else {
            self.c.to_string()
        }
    }
}

/// 소스 배열의 캐시 키 조각 — 전부 미사용이면 빈 문자열
pub fn key_of(srcs: &[SrcView]) -> String {
    if !srcs.iter().any(|s| s.is_used()) {
        return String::new();
    }
    let parts: Vec<String> = srcs.iter().filter(|s| s.is_used()).map(|s| s.key()).collect();
    format!(" src[{}]", parts.join(","))
}
