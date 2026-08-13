//! 출력 텐서를 캔버스에 바로 그리는 경로 — CPU 리드백 없이.
//!
//! **왜 필요한가**: 마스크를 CPU로 꺼내면(map_async / readPixels) 그 왕복이 추론보다
//! 비싸다. 실측(144×256 알파): webgl2의 readPixels×4가 17~18ms, 순추론은 2.3ms였다.
//! 실제 파이프라인은 마스크를 GPU에 둔 채 합성해야 하고, 이 모듈이 그 경로다.
//!
//! 출력은 **불투명 그레이스케일** (a,a,a,1). wgpu WebGPU 백엔드가 Opaque 서피스만
//! 노출해서 캔버스 알파를 쓸 수 없기 때문이다 — 호스트가 휘도로 합성한다.

use ai_core::TensorDesc;
use ai_gpu::wgpu;
use ai_gpu::GpuContext;

const SHADER: &str = r#"
struct P { w: u32, h: u32, cg: u32, ch: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> SRC: array<vec4<f32>>;

// 풀스크린 삼각형 (버텍스 버퍼 없음)
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(i & 2u) * 2.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), p.w - 1u);
    let y = min(u32(pos.y), p.h - 1u);
    // NHWC-C4: idx(h,w,cg) = (y*W + x)*cg_count + cg
    let v4 = SRC[(y * p.w + x) * p.cg + (p.ch >> 2u)];
    let a = clamp(v4[p.ch & 3u], 0.0, 1.0);
    // 불투명 그레이스케일. wgpu의 WebGPU 백엔드는 alpha_modes=[Opaque]만 주므로
    // 캔버스 알파는 못 쓴다 — 호스트가 휘도로 합성한다(out = bg(1-m) + fg·m).
    return vec4<f32>(a, a, a, 1.0);
}
"#;

pub struct Presenter {
    surface: wgpu::Surface<'static>,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    size: (u32, u32),
}

impl Presenter {
    pub fn new(
        ctx: &GpuContext,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<Self, String> {
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
        surface.configure(
            &ctx.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                color_space: Default::default(),
                desired_maximum_frame_latency: 2,
            },
        );

        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { surface, pipeline, bgl, params, size: (w, h) })
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// `src`(NHWC-C4 스토리지)의 채널 `ch`를 알파로 캔버스에 그린다.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        src: &wgpu::Buffer,
        desc: &TensorDesc,
        ch: u32,
    ) -> Result<(), String> {
        let p = [desc.w, desc.h, desc.cg(), ch];
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            ],
        });
        // wgpu 30: get_current_texture는 Result가 아니라 상태 enum을 준다
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => return Err(format!("서피스 텍스처 획득 실패: {other:?}")),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit([enc.finish()]);
        drop(frame); // wgpu 30은 SurfaceTexture drop 시점에 present 한다
        Ok(())
    }
}
