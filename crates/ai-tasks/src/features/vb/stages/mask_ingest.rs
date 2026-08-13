//! 마스크 인제스트 — 모델 출력 → r8 마스크 + 시간 EMA (핑퐁)

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

use crate::features::vb::stages::{ubo, Bind, Fullscreen, FULLSCREEN_VS};

const SRC: &str = include_str!("../shaders/mask_ingest.wgsl");

/// v-ai 파리티: diff>0.3 ? 0.9 : 0.03 (2ch 로짓), RVM(pha)류는 0.6/0.9
pub(crate) struct MaskIngest {
    pub fs: Fullscreen,
    pub params: wgpu::Buffer,
}

#[derive(Clone, Copy)]
pub(crate) enum MaskKind {
    /// 2채널 로짓 [bg, person] → sigmoid(person-bg)
    Logits2,
    /// 알파 직출 (RVM pha)
    Alpha,
}

impl MaskIngest {
    pub fn new(ctx: &GpuContext) -> Self {
        let src = format!("{FULLSCREEN_VS}\n{SRC}");
        let fs = Fullscreen::new(
            ctx,
            "video-mask-ingest",
            &src,
            wgpu::TextureFormat::R8Unorm,
            &[Bind::TexNf, Bind::TexNf, Bind::Uniform],
        );
        MaskIngest { fs, params: ubo(ctx, "video-mask-ingest", 32) }
    }

    /// ema=false: min_a=max_a=1 → 출력=현재 프레임 (게이트가 시간 상태를 끊을 때)
    pub fn write_params(&self, ctx: &GpuContext, kind: MaskKind, ema: bool) {
        let (mode, mut min_a, mut max_a) = match kind {
            MaskKind::Logits2 => (0u32, 0.03f32, 0.9f32),
            MaskKind::Alpha => (1, 0.6, 0.9),
        };
        if !ema {
            min_a = 1.0;
            max_a = 1.0;
        }
        let mut b = [0u8; 32];
        b[0..4].copy_from_slice(&mode.to_le_bytes());
        b[16..20].copy_from_slice(&0.3f32.to_le_bytes()); // diff 문턱
        b[20..24].copy_from_slice(&min_a.to_le_bytes());
        b[24..28].copy_from_slice(&max_a.to_le_bytes());
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
