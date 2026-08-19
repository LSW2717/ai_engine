//! 최종 합성 — 배경 4모드 + coverage/spill/edge + 밝기/흑백(배경만)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::params::{Background, EffectsState};
use crate::features::vb::stages::{ubo, Bind, Fullscreen, FULLSCREEN_VS};

const SRC: &str = include_str!("../shaders/compose.wgsl");

pub(crate) struct Compose {
    pub fs: Fullscreen,
    pub params: wgpu::Buffer,
}

/// 터치업/메이크업 uniform 값 (compose.wgsl tu_map/tu_par/mk_map) — 전부 0 = off.
/// 프레임 정규화 좌표 → 128² 오버레이 uv 매핑 + 강도/블러 스트라이드.
#[derive(Clone, Copy, Default)]
pub(crate) struct FaceFxParams {
    pub tu_map: [f32; 4],
    pub tu_par: [f32; 4],
    pub mk_map: [f32; 4],
}

impl Compose {
    pub fn new(ctx: &GpuContext, target: wgpu::TextureFormat) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let fs = Fullscreen::new(
            ctx,
            "video-compose",
            &src,
            target,
            &[Bind::Tex, Bind::Tex, Bind::Tex, Bind::Tex, Bind::Sampler, Bind::Uniform,
                Bind::Tex, Bind::Tex, Bind::Tex],
        );
        Compose { fs, params: ubo(ctx, "video-compose", 256) }
    }

    /// bg_dims: 업로드된 배경 이미지 크기 (cover 크롭 계산용).
    /// framing: 인물 중앙 프레이밍 (scale, cx, cy — scale 1 = off)
    #[allow(clippy::too_many_arguments)]
    pub fn write_params(
        &self,
        ctx: &GpuContext,
        st: &EffectsState,
        (fw, fh): (u32, u32),
        bg_dims: Option<(u32, u32)>,
        framing: (f32, f32, f32),
        face_fx: &FaceFxParams,
    ) {
        let d = st.derived();
        let (mode, color) = match &st.background {
            Background::None if st.blur > 0.0 => (1u32, [0.0; 3]),
            Background::None => (0, [0.0; 3]),
            Background::Color(c) => (2, *c),
            Background::Image => (3, [0.0; 3]),
        };
        // cover 크롭 — 웹 updateBackgroundImage 등가 (f64 — JS와 비트 정합)
        let (mut sx, mut sy) = (1.0f64, 1.0f64);
        if let Some((iw, ih)) = bg_dims {
            let (ir, cr) = (iw as f64 / ih as f64, fw as f64 / fh as f64);
            if ir > cr {
                sx = cr / ir;
            } else {
                sy = ir / cr;
            }
        }
        // mirror/degree 배경 보정 — 웹 updateTransform/applyScaleAndOffset/
        // updateAspectComp 1:1. 프레임 자체는 호스트가 추론 전에 변환(계약) —
        // 여기선 이미지 배경 샘플 좌표만 같은 변환을 따라간다.
        let rad = (st.degree as f64).rem_euclid(360.0) * std::f64::consts::PI / 180.0;
        let ms = if st.mirror { -1.0f64 } else { 1.0 };
        let (sin_r, cos_r) = rad.sin_cos();
        // GL uniformMatrix2fv 열우선 [c0x, c0y, c1x, c1y]
        let mut mat = [cos_r * ms, -sin_r, sin_r * ms, cos_r];
        let quarter = (rad - std::f64::consts::FRAC_PI_2).abs() < 0.001
            || (rad - 3.0 * std::f64::consts::FRAC_PI_2).abs() < 0.001;
        if st.mirror && quarter {
            for v in &mut mat {
                *v = -*v;
            }
        }
        // contain 스케일 — 회전된 콘텐츠가 화면을 넘지 않게
        let (content_w, content_h) = (sx * fw as f64, sy * fh as f64);
        let rot_w = cos_r.abs() * content_w + sin_r.abs() * content_h;
        let rot_h = sin_r.abs() * content_w + cos_r.abs() * content_h;
        let contain = if rot_w > 0.0 && rot_h > 0.0 {
            (fw as f64 / rot_w).min(fh as f64 / rot_h)
        } else {
            1.0
        };
        let mult = contain.min(1.0);
        let (esx, esy) = (sx * mult, sy * mult);
        let (offx, offy) =
            ((1.0 - sx) * 0.5 + (sx - esx) * 0.5, (1.0 - sy) * 0.5 + (sy - esy) * 0.5);
        // aspect 보정 (baseScale 비율 기반 — 웹 updateAspectComp)
        let (aw, ah) = (sx.max(1e-4), sy.max(1e-4));
        let (aspect_x, aspect_y) =
            if aw > ah { (aw / ah, 1.0) } else if ah > aw { (1.0, ah / aw) } else { (1.0, 1.0) };
        let (sx, sy) = (esx as f32, esy as f32);
        let (offx, offy) = (offx as f32, offy as f32);
        let mut b = Vec::with_capacity(80);
        b.extend_from_slice(&mode.to_le_bytes());
        for v in [st.blur, st.brightness, st.grayscale, d.coverage[0], d.coverage[1], d.spill,
            d.edge_darkening]
        {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in [color[0], color[1], color[2], 1.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        // scale/offset은 유효값(contain 배율 반영), texel = (scale/W, scale/H) —
        // v-ai applyScaleAndOffset 규약 (이미지 배경 자체 블러 오프셋)
        for v in [sx, sy, offx, offy, d.light_wrapping,
            0.0 /* _pad0 (구 use_fgr 슬롯) */, sx / fw as f32, sy / fh as f32]
        {
            b.extend_from_slice(&v.to_le_bytes());
        }
        // 스튜디오 조명: relight vec4 + lights 4×vec4 (2광원 × [pos/radius/int, color/target])
        let sl = st.studio_light.as_ref();
        let aspect = fw as f32 / fh.max(1) as f32;
        for v in [if sl.is_some() { 1.0 } else { 0.0 }, sl.map(|o| o.ambient).unwrap_or(1.0), aspect, 0.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..2 {
            let l = sl.and_then(|o| o.lights.get(i)).filter(|l| l.enabled);
            let (pr, ct) = match l {
                Some(l) => {
                    let c = crate::features::vb::params::EffectsState::hex_rgb(&l.color);
                    let t = match l.target.as_str() {
                        "person" => 0.0,
                        "background" => 1.0,
                        _ => 2.0,
                    };
                    ([l.x, l.y, l.radius.max(1e-4), l.intensity], [c[0], c[1], c[2], t])
                }
                None => ([0.0, 0.0, 1e-4, 0.0], [0.0, 0.0, 0.0, 2.0]),
            };
            for v in pr.into_iter().chain(ct) {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [framing.0, framing.1, framing.2, 0.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in mat {
            b.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for v in [aspect_x as f32, aspect_y as f32, 0.0, 0.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in face_fx.tu_map.into_iter().chain(face_fx.tu_par).chain(face_fx.mk_map) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        ctx.queue.write_buffer(&self.params, 0, &b);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_valid() {
        let src = format!("{}\n{}", super::FULLSCREEN_VS, super::SRC);
        let m = naga::front::wgsl::parse_str(&src).unwrap();
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap();
    }
}
