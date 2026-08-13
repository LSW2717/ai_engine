//! 마스크 + 카메라 프레임 합성기 — **플랫폼 무관**.
//!
//! 서피스(캔버스/CAMetalLayer/ANativeWindow) 획득과 프레임 임포트만 플랫폼마다
//! 다르다. 그 둘은 바인딩 크레이트가 하고, 여기는 "주어진 타깃 뷰에 그린다"까지만
//! 안다. 합성 수식·업샘플·리소스 캐시가 한 벌로 공유되는 지점.

use std::cell::RefCell;

use ai_core::TensorDesc;
use ai_gpu::wgpu;
use ai_gpu::GpuContext;

const SHADER: &str = include_str!("shaders/compositor.wgsl");

/// 합성 파라미터 (셰이더 uniform과 1:1)
#[derive(Clone, Copy, Debug)]
pub struct CompositeOpts {
    /// 마스크로 쓸 출력 채널
    pub channel: u32,
    /// 0 = 배경 합성, 1 = 마스크만 보기
    pub mode: u32,
    /// 0 = 그라디언트, 1 = 검정, 2 = 프레임 블러
    pub bg: u32,
}

impl Default for CompositeOpts {
    fn default() -> Self {
        Self { channel: 0, mode: 0, bg: 0 }
    }
}

/// 모델 출력 버퍼를 옮겨 담는 중간 텍스처 + 바인드그룹.
struct Staged {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    /// 바인드그룹은 두 텍스처가 그대로면 재사용한다.
    /// 매 프레임 만들면 30fps에서 분당 1800개가 쌓여 시간이 계속 늘어난다
    /// (실측: 추론 2.59 → 6.76ms로 단조 증가).
    bind: Option<wgpu::BindGroup>,
    frame_gen: u32,
}

struct FrameTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    /// 재생성될 때마다 증가 — 캐시된 바인드그룹을 무효화하는 신호
    gen_id: u32,
}

pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    staging: RefCell<Option<Staged>>,
    frame: RefCell<Option<FrameTex>>,
}

impl Compositor {
    /// `target_format` = 최종적으로 그려질 서피스/텍스처 포맷.
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
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
                targets: &[Some(target_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            pipeline,
            bgl,
            params,
            sampler,
            staging: RefCell::new(None),
            frame: RefCell::new(None),
        }
    }

    /// 카메라 프레임 텍스처를 `w×h`로 확보하고 클로저에 넘긴다.
    ///
    /// 채우는 방법은 플랫폼마다 다르다(웹 `copy_external_image_to_texture`,
    /// 네이티브 `CVPixelBuffer`/`AHardwareBuffer` 임포트) — 그래서 채우기는
    /// 호출자에게 맡기고, **생성·크기 추적·바인드그룹 무효화만** 여기서 한다.
    /// 그 세 가지가 프레임당 리소스 누수의 진원지였다.
    pub fn with_frame_texture<R>(
        &self,
        ctx: &GpuContext,
        w: u32,
        h: u32,
        f: impl FnOnce(&wgpu::Texture) -> R,
    ) -> R {
        let mut ft = self.frame.borrow_mut();
        if ft.as_ref().map(|t| (t.w, t.h)) != Some((w, h)) {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("composite-frame"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let gen_id = ft.as_ref().map(|t| t.gen_id.wrapping_add(1)).unwrap_or(0);
            *ft = Some(FrameTex { tex, view, w, h, gen_id });
        }
        f(&ft.as_ref().unwrap().tex)
    }

    /// 마스크 버퍼(NHWC-C4 스토리지)와 프레임 텍스처를 합성해 `target`에 그린다.
    /// 커맨드는 호출자가 준 인코더에 기록한다 — 제출/서피스 present는 호출자 몫.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        mask: &wgpu::Buffer,
        desc: &TensorDesc,
        target: &wgpu::TextureView,
        opts: CompositeOpts,
    ) -> Result<(), String> {
        let p = [desc.w, desc.h, desc.cg(), opts.channel, opts.mode, opts.bg, 0u32, 0u32];
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));

        // NHWC-C4 버퍼 → rgba32float 텍스처. 텍스처 폭 = W * cg (채널그룹 인터리브).
        // copy_buffer_to_texture는 bytes_per_row가 256의 배수여야 한다.
        let (tw, th) = (desc.w * desc.cg(), desc.h);
        let bytes_per_row = tw * 16;
        if bytes_per_row % 256 != 0 {
            return Err(format!(
                "composite: 행 바이트 {bytes_per_row}가 256 정렬이 아님 (W={} cg={})",
                desc.w,
                desc.cg()
            ));
        }
        let mut st = self.staging.borrow_mut();
        if st.as_ref().map(|s| (s.w, s.h)) != Some((tw, th)) {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("composite-staging"),
                size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            *st = Some(Staged { tex, view, w: tw, h: th, bind: None, frame_gen: u32::MAX });
        }
        let ft = self.frame.borrow();
        let frame = ft.as_ref().ok_or_else(|| "with_frame_texture() 먼저".to_string())?;
        {
            let staged = st.as_mut().unwrap();
            if staged.bind.is_none() || staged.frame_gen != frame.gen_id {
                staged.bind = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite-bind"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&staged.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&frame.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }));
                staged.frame_gen = frame.gen_id;
            }
        }
        let staged = st.as_ref().unwrap();
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: mask,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(th),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &staged.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
        );
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_bind_group(0, staged.bind.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
