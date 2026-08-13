//! studio — VideoPipeline의 웹 서피스 바인딩 (플랫폼 몫: 서피스 획득 + 프레임 임포트).
//! 파이프라인 로직은 전부 ai_tasks::video — 이 파일이 커지면 로직이 샌 것이다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::features::vb::VideoPipeline;
use ai_tasks::GpuSession;

pub struct Studio {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pub pipeline: VideoPipeline,
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
        Ok(Studio { surface, config, pipeline: VideoPipeline::new(ctx, format) })
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
        drop(frame); // present
        Ok(())
    }
}
