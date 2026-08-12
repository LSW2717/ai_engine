//! WGSL 템플릿 슬롯 치환 + 코드 조각 빌더.
//!
//! 템플릿(.wgsl)은 정적 골격을 담고, `//@NAME` 마커 라인이 Rust가 생성한
//! 블록으로 치환된다. 치환되지 않은 마커는 빈 줄이 된다(선택적 슬롯).

/// 템플릿의 `//@NAME` 마커 라인을 대응 블록으로 치환.
/// 마커의 들여쓰기가 블록 전 라인에 적용된다.
pub fn fill(template: &str, slots: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(template.len() * 2);
    'line: for line in template.lines() {
        let trimmed = line.trim_start();
        if let Some(name) = trimmed.strip_prefix("//@") {
            let indent = &line[..line.len() - trimmed.len()];
            for (slot, body) in slots {
                if name == *slot {
                    for bline in body.lines() {
                        out.push_str(indent);
                        out.push_str(bline);
                        out.push('\n');
                    }
                    continue 'line;
                }
            }
            continue 'line; // 매칭 없는 슬롯은 제거
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// 여러 줄 코드 조각 빌더 (언롤 루프 생성용)
#[derive(Default)]
pub struct W {
    s: String,
}

impl W {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn line(&mut self, l: impl AsRef<str>) -> &mut Self {
        self.s.push_str(l.as_ref());
        self.s.push('\n');
        self
    }

    pub fn done(self) -> String {
        self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_replaces_marker_with_indent() {
        let t = "a\n    //@BODY\nb\n//@GONE\n";
        let out = fill(t, &[("BODY", "x = 1;\ny = 2;".to_string())]);
        assert_eq!(out, "a\n    x = 1;\n    y = 2;\nb\n");
    }
}
