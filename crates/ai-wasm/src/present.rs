//! 웹 서피스 바인딩 — **여기 있는 건 캔버스 관련뿐이다.**
//!
//! 합성 셰이더·업샘플·리소스 캐시는 `ai_tasks::Compositor`로 내려갔다.
//! 모바일(`ai-ffi`)은 같은 Compositor에 CAMetalLayer/ANativeWindow 뷰만 물리면 된다.
//! 이 파일이 다시 커지면 로직이 새어 올라온 것이다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;
use ai_tasks::{CompositeOpts, Compositor};

pub struct Presenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    compositor: Compositor,
    size: (u32, u32),
}

impl Presenter {
    pub fn new(ctx: &GpuContext, canvas: web_sys::HtmlCanvasElement) -> Result<Self, String> {
        let (w, h) = (canvas.width().max(1), canvas.height().max(1));
        let surface = ctx
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("서피스 생성 실패: {e:?}"))?;
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps.formats[0];
        // 알파가 페이지 합성까지 살아야 destination-in 마스킹에 쓸 수 있다.
        // Opaque로 떨어지면 알파가 1로 고정돼 마스크가 무의미해진다 — 실제로 한 번 그랬다.
        log::info!("[ai-wasm] present alpha_modes={:?} formats={:?}", caps.alpha_modes, caps.formats);
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            log::warn!("[ai-wasm] PreMultiplied 미지원 — 알파 마스킹이 동작하지 않는다");
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            color_space: Default::default(),
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctx.device, &config);
        log::info!("[ai-wasm] present 준비 ({w}x{h}, {format:?}, {alpha_mode:?})");
        Ok(Self { surface, config, compositor: Compositor::new(ctx, format), size: (w, h) })
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// 캔버스/비디오 등 브라우저 이미지 소스를 GPU 프레임 텍스처로 (CPU 왕복 없음).
    /// **웹 전용 임포트 경로** — 모바일은 CVPixelBuffer/AHardwareBuffer를 쓴다.
    pub fn upload_frame(
        &self,
        ctx: &GpuContext,
        src: &wgpu::wgt::CopyExternalImageSourceInfo,
        w: u32,
        h: u32,
    ) {
        self.compositor.with_frame_texture(ctx, w, h, |tex| {
            ctx.queue.copy_external_image_to_texture(
                src,
                wgpu::wgt::CopyExternalImageDestInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                    color_space: wgpu::wgt::PredefinedColorSpace::Srgb,
                    premultiplied_alpha: false,
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        });
    }

    /// 마스크 + 프레임을 합성해 캔버스에 그린다.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        mask: &wgpu::Buffer,
        desc: &ai_core::TensorDesc,
        opts: CompositeOpts,
    ) -> Result<(), String> {
        // wgpu 30: get_current_texture는 Result가 아니라 상태 enum을 준다.
        // Outdated/Lost는 정상적으로 발생할 수 있다 — 재구성하고 이번 프레임은 건너뛴다.
        // (여기서 하드 에러를 내면 호스트 루프가 죽는다. 실제로 그렇게 만들었었다.)
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                log::warn!("[ai-wasm] 서피스 상태 {other:?} — 재구성 후 이 프레임 건너뜀");
                self.surface.configure(&ctx.device, &self.config);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.compositor.draw(ctx, &mut enc, mask, desc, &view, opts)?;
        ctx.queue.submit([enc.finish()]);
        drop(frame); // wgpu 30은 SurfaceTexture drop 시점에 present 한다
        Ok(())
    }
}
