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

impl Compose {
    pub fn new(ctx: &GpuContext, target: wgpu::TextureFormat) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let fs = Fullscreen::new(
            ctx,
            "video-compose",
            &src,
            target,
            &[Bind::Tex, Bind::Tex, Bind::Tex, Bind::Tex, Bind::Sampler, Bind::Uniform],
        );
        Compose { fs, params: ubo(ctx, "video-compose", 160) }
    }

    /// bg_dims: 업로드된 배경 이미지 크기 (cover 크롭 계산용)
    pub fn write_params(
        &self,
        ctx: &GpuContext,
        st: &EffectsState,
        (fw, fh): (u32, u32),
        bg_dims: Option<(u32, u32)>,
    ) {
        let d = st.derived();
        let (mode, color) = match &st.background {
            Background::None if st.blur > 0.0 => (1u32, [0.0; 3]),
            Background::None => (0, [0.0; 3]),
            Background::Color(c) => (2, *c),
            Background::Image => (3, [0.0; 3]),
        };
        // cover 크롭 — 웹 updateBackgroundImage 등가
        let (mut sx, mut sy) = (1.0f32, 1.0f32);
        if let Some((iw, ih)) = bg_dims {
            let (ir, cr) = (iw as f32 / ih as f32, fw as f32 / fh as f32);
            if ir > cr {
                sx = cr / ir;
            } else {
                sy = ir / cr;
            }
        }
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
        for v in [sx, sy, (1.0 - sx) * 0.5, (1.0 - sy) * 0.5, d.light_wrapping, 0.0, 0.0, 0.0] {
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
