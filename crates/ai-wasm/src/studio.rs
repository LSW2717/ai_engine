//! studio — VideoPipeline의 웹 서피스 바인딩 (플랫폼 몫: 서피스 획득 + 프레임 임포트).
//! 파이프라인 로직은 전부 ai_tasks::video — 이 파일이 커지면 로직이 샌 것이다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::face::items3d::ItemsOverlay;
use ai_tasks::features::vb::VideoPipeline;
use ai_tasks::GpuSession;

pub struct Studio {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pub pipeline: VideoPipeline,
    /// 3D 아이템 오버레이 (wgpu — three.js 대체, 웹·모바일 렌더러 통일).
    /// studio_items가 첫 사용 시 생성. ⚠ 프레이밍 크롭 좌표 보정은 P3 이월.
    pub items: Option<ItemsOverlay>,
}

impl Studio {
    /// 파이프라인 출력 위에 아이템 오버레이 (그릴 게 없으면 no-op)
    fn overlay(&mut self, ctx: &GpuContext, view: &wgpu::TextureView) {
        if let Some(items) = &mut self.items {
            items.draw(ctx, view, self.config.format, self.config.width, self.config.height);
        }
    }
}

impl Studio {
    pub fn new(ctx: &GpuContext, canvas: web_sys::HtmlCanvasElement) -> Result<Self, String> {
        let (w, h) = (canvas.width().max(1), canvas.height().max(1));
        let surface = ctx
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("서피스 생성 실패: {e:?}"))?;
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            color_space: Default::default(),
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctx.device, &config);
        Ok(Studio { surface, config, pipeline: VideoPipeline::new(ctx, format), items: None })
    }

    /// 프레임 1장: 소스 캔버스 → GPU 텍스처(무복사 임포트) → 파이프라인 → 서피스
    pub async fn frame(
        &mut self,
        ctx: &GpuContext,
        seg: &mut GpuSession,
        source: &web_sys::HtmlCanvasElement,
    ) -> Result<(), String> {
        let (fw, fh) = (source.width().max(1), source.height().max(1));
        let src = wgpu::wgt::CopyExternalImageSourceInfo {
            source: wgpu::wgt::ExternalImageSource::HTMLCanvasElement(source.clone()),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        };
        self.pipeline
            .with_frame_texture(ctx, seg, fw, fh, |tex| {
                ctx.queue.copy_external_image_to_texture(
                    &src,
                    wgpu::wgt::CopyExternalImageDestInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::wgt::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
                );
            })
            .map_err(|e| e.to_string())?;
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                log::warn!("[studio] 서피스 {other:?} — 재구성 후 스킵");
                self.surface.configure(&ctx.device, &self.config);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.pipeline
            .process_gpu(ctx, seg, fw, fh, &view)
            .await
            .map_err(|e| e.to_string())?;
        self.overlay(ctx, &view);
        drop(frame); // present
        Ok(())
    }

    /// B티어 프레임: 소스 캔버스 + **외부(CPU 추론) 마스크** → 이펙트 스택 → 서피스.
    /// GPU 추론이 없다 — 마스크 업로드(수십 KB)가 프레임당 CPU→GPU 트래픽 전부.
    #[allow(clippy::too_many_arguments)]
    pub fn frame_mask(
        &mut self,
        ctx: &GpuContext,
        seg: &GpuSession,
        source: &web_sys::HtmlCanvasElement,
        mask: &[f32],
        ch: u32,
        mask_w: u32,
        mask_h: u32,
    ) -> Result<(), String> {
        let (fw, fh) = (source.width().max(1), source.height().max(1));
        let src = wgpu::wgt::CopyExternalImageSourceInfo {
            source: wgpu::wgt::ExternalImageSource::HTMLCanvasElement(source.clone()),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        };
        self.pipeline
            .with_frame_texture(ctx, seg, fw, fh, |tex| {
                ctx.queue.copy_external_image_to_texture(
                    &src,
                    wgpu::wgt::CopyExternalImageDestInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::wgt::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
                );
            })
            .map_err(|e| e.to_string())?;
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                log::warn!("[studio] 서피스 {other:?} — 재구성 후 스킵");
                self.surface.configure(&ctx.device, &self.config);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.pipeline
            .process_gpu_mask(ctx, seg, mask, ch, mask_w, mask_h, true, fw, fh, &view)
            .map_err(|e| e.to_string())?;
        self.overlay(ctx, &view);
        drop(frame); // present
        Ok(())
    }
}
