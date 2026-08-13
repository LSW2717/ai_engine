//! 이펙트 파라미터 — vcxrust_ai `EffectsPatch` 머지 규약(사용자 채택, INTEGRATION.md D3):
//! JSON 패치에서 **필드 없음 = 유지 / null = 해제 / 값 = 설정**.
//!
//! 호스트(웹 JS·모바일 호스트)는 항상 부분 패치 JSON 문자열 하나만 보낸다 —
//! 옵션 스키마가 바인딩마다 갈라지는 사고(웹 VBOptions/네이티브 flat 3중 수정)를
//! 원천 차단한다. 파생 상수(coverage·bilateral σ 등 blur 종속값)는 `derived()`
//! 한 곳에서만 계산한다 (웹 `_computePostProcessingConfig` 등가).

use serde::Deserialize;

fn double<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    // 필드가 존재하면 여기로 온다: null → Some(None), 값 → Some(Some(v))
    Ok(Some(Option::<T>::deserialize(d)?))
}

/// 배경 지정 — 색상 hex("#rrggbb") 또는 "image"(별도 업로드된 이미지 사용)
#[derive(Clone, Debug, PartialEq)]
pub enum Background {
    None,
    Color([f32; 3]),
    Image,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StudioLight {
    pub enabled: bool,
    pub x: f32,
    pub y: f32,
    /// "#rrggbb"
    pub color: String,
    pub intensity: f32,
    pub radius: f32,
    /// "person" | "background" | "all"
    pub target: String,
}

impl Default for StudioLight {
    fn default() -> Self {
        StudioLight {
            enabled: true, x: 0.5, y: 0.3, color: "#ffffff".into(),
            intensity: 0.8, radius: 0.5, target: "all".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StudioLightOptions {
    pub enabled: bool,
    pub ambient: f32,
    /// 최대 2개 (초과분 무시 — v-ai 규약)
    pub lights: Vec<StudioLight>,
}

impl Default for StudioLightOptions {
    fn default() -> Self {
        StudioLightOptions { enabled: false, ambient: 1.0, lights: Vec::new() }
    }
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EffectsPatch {
    /// null=배경 해제, "#rrggbb"=단색, "image"=업로드된 이미지
    #[serde(deserialize_with = "double")]
    pub background: Option<Option<String>>,
    /// 0..1 (배경 블러 강도, 0=off)
    pub blur: Option<f32>,
    /// 1.0 = 원본 (배경에만 적용 — 웹 규약)
    pub brightness: Option<f32>,
    /// 0..1 (배경에만 적용)
    pub grayscale: Option<f32>,
    /// null=조명 해제
    #[serde(deserialize_with = "double")]
    pub studio_light: Option<Option<StudioLightOptions>>,
}

/// 해석된 현재 상태 — 파이프라인이 프레임마다 읽는 단일 진실
#[derive(Clone, Debug)]
pub struct EffectsState {
    pub background: Background,
    pub blur: f32,
    pub brightness: f32,
    pub grayscale: f32,
    pub studio_light: Option<StudioLightOptions>,
}

impl Default for EffectsState {
    fn default() -> Self {
        EffectsState {
            background: Background::None,
            blur: 0.0,
            brightness: 1.0,
            grayscale: 0.0,
            studio_light: None,
        }
    }
}

fn parse_hex(s: &str) -> Option<[f32; 3]> {
    let v = u32::from_str_radix(s.strip_prefix('#')?, 16).ok()?;
    Some([
        ((v >> 16) & 255) as f32 / 255.0,
        ((v >> 8) & 255) as f32 / 255.0,
        (v & 255) as f32 / 255.0,
    ])
}

impl EffectsState {
    pub fn apply(&mut self, patch: &EffectsPatch) {
        if let Some(bg) = &patch.background {
            self.background = match bg.as_deref() {
                None => Background::None,
                Some("image") => Background::Image,
                Some(hex) => parse_hex(hex).map(Background::Color).unwrap_or(Background::None),
            };
        }
        if let Some(v) = patch.blur {
            self.blur = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.brightness {
            self.brightness = v.clamp(0.0, 2.0);
        }
        if let Some(v) = patch.grayscale {
            self.grayscale = v.clamp(0.0, 1.0);
        }
        if let Some(sl) = &patch.studio_light {
            self.studio_light = sl.clone().filter(|o| o.enabled);
        }
    }

    /// "#rrggbb" → [r,g,b] (0..1)
    pub fn hex_rgb(s: &str) -> [f32; 3] {
        parse_hex(s).unwrap_or([1.0, 1.0, 1.0])
    }

    pub fn apply_json(&mut self, json: &str) -> Result<(), String> {
        let patch: EffectsPatch =
            serde_json::from_str(json).map_err(|e| format!("EffectsPatch 파싱: {e}"))?;
        self.apply(&patch);
        Ok(())
    }

    /// blur 종속 파생 상수 — 웹 `_computePostProcessingConfig` 등가 (v-ai 파리티 스펙)
    pub fn derived(&self) -> Derived {
        let b = self.blur;
        let image = matches!(self.background, Background::Image);
        Derived {
            sigma_space: 2.0 + b * 3.2,
            sigma_color: 0.1 + b * 0.36,
            coverage: [(0.3 - b * 0.05).max(0.01), (0.7 + b * 0.05).min(0.99)],
            light_wrapping: 0.05 + b * 0.1,
            spill: if image { 0.18 } else { 0.14 },
            edge_darkening: if image { 0.24 } else { 0.2 },
        }
    }
}

pub struct Derived {
    pub sigma_space: f32,
    pub sigma_color: f32,
    pub coverage: [f32; 2],
    pub light_wrapping: f32,
    pub spill: f32,
    pub edge_darkening: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_semantics() {
        let mut s = EffectsState::default();
        s.apply_json(r##"{"background":"#ff0000","blur":0.5}"##).unwrap();
        assert_eq!(s.background, Background::Color([1.0, 0.0, 0.0]));
        assert_eq!(s.blur, 0.5);
        // 필드 없음 = 유지
        s.apply_json(r#"{"brightness":1.2}"#).unwrap();
        assert_eq!(s.background, Background::Color([1.0, 0.0, 0.0]));
        assert_eq!(s.blur, 0.5);
        // null = 해제
        s.apply_json(r#"{"background":null}"#).unwrap();
        assert_eq!(s.background, Background::None);
        // 파생 상수는 blur를 따라간다
        assert!((s.derived().sigma_space - (2.0 + 0.5 * 3.2)).abs() < 1e-6);
    }
}
