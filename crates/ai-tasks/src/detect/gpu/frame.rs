//! 프레임 텍스처 홀더 — 카메라 프레임의 GPU 상주 사본 (Rgba8Unorm).
//!
//! 웹은 `copy_external_image_to_texture`(무복사 임포트), 네이티브·ffi·테스트는
//! `upload_rgb`(write_texture)로 채운다. RENDER_ATTACHMENT는 웹 임포트의 필수
//! usage (WebGPU copyExternalImageToTexture 규약 — vb frame_tex와 동일).

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

pub struct FrameTex {
    t: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
}

impl FrameTex {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        FrameTex { t: None }
    }

    /// w×h 텍스처 보장 (크기 다르면 재생성) — 반환 텍스처에 임포트/업로드한다
    pub fn ensure(&mut self, ctx: &GpuContext, w: u32, h: u32) -> &wgpu::Texture {
        let stale = !matches!(&self.t, Some((_, _, tw, th)) if *tw == w && *th == h);
        if stale {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("det-frame"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.t = Some((tex, view, w, h));
        }
        &self.t.as_ref().unwrap().0
    }

    /// 현재 텍스처 뷰 + 크기 (ensure/upload 이전엔 None)
    pub fn view(&self) -> Option<(&wgpu::TextureView, u32, u32)> {
        self.t.as_ref().map(|(_, v, w, h)| (v, *w, *h))
    }

    /// u8 RGB(타이트) 프레임 업로드 — RGBA 패딩 후 write_texture.
    /// 네이티브 테스트·ffi용 (웹은 캔버스 임포트가 이 일을 한다).
    pub fn upload_rgb(&mut self, ctx: &GpuContext, rgb: &[u8], w: u32, h: u32) {
        assert_eq!(rgb.len(), (w * h * 3) as usize);
        let mut rgba = vec![255u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
        }
        let tex = self.ensure(ctx, w, h);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }
}
