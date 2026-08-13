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
// ⚠ 프래그먼트 스테이지에서 **스토리지 버퍼를 읽지 않는다**.
// WebGPU는 maxStorageBuffersInFragmentStage가 구현마다 다르고 0일 수 있다
// (compat 모드/일부 브라우저). 검증 오류는 비동기라 조용히 실패하고, 캔버스가
// 지워진 채 남아 "검은 마스크"가 된다 — 사파리에서 실제로 그랬다.
// 대신 출력 버퍼를 rgba32float 텍스처로 복사해 textureLoad로 읽는다.
// textureLoad는 필터링을 안 하므로 float32-filterable도 필요 없다.
// 합성까지 여기서 끝낸다: out = bg*(1-m) + fg*m.
// 캔버스 2D 블렌드(multiply/difference/lighter)로 하면 브라우저마다 결과가
// 달라진다 — 사파리에서 합성만 깨졌던 원인이 그것이다. 셰이더에서 하면 없다.
struct P {
    w: u32, h: u32, cg: u32, ch: u32,
    mode: u32,      // 0 = 합성, 1 = 마스크만
    bg: u32,        // 0 = 그라디언트, 1 = 검정, 2 = 프레임 블러(근사)
    _pad0: u32, _pad1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var SRC: texture_2d<f32>;   // 모델 출력 (NHWC-C4)
@group(0) @binding(2) var FRAME: texture_2d<f32>; // 카메라 프레임 (rgba8)
@group(0) @binding(3) var SAMP: sampler;

// 풀스크린 삼각형 (버텍스 버퍼 없음)
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(i & 2u) * 2.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// 마스크는 모델 해상도라 표시 해상도로 확대된다. textureLoad는 보간이 없으니
// 이웃 4텍셀을 직접 바이리니어로 섞어 경계 계단을 없앤다.
fn maskAt(uv: vec2<f32>) -> f32 {
    let fw = f32(p.w);
    let fh = f32(p.h);
    let t = vec2<f32>(uv.x * fw - 0.5, uv.y * fh - 0.5);
    let b = floor(t);
    let f = t - b;
    let cg = i32(p.cg);
    let lane = i32(p.ch & 3u);
    let grp = i32(p.ch >> 2u);
    var acc = 0.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let xi = clamp(i32(b.x) + dx, 0, i32(p.w) - 1);
            let yi = clamp(i32(b.y) + dy, 0, i32(p.h) - 1);
            let v4 = textureLoad(SRC, vec2<i32>(xi * cg + grp, yi), 0);
            let wgt = (select(1.0 - f.x, f.x, dx == 1)) * (select(1.0 - f.y, f.y, dy == 1));
            acc = acc + clamp(v4[lane], 0.0, 1.0) * wgt;
        }
    }
    return acc;
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(FRAME));
    let uv = pos.xy / dim;
    let a = maskAt(uv);
    if (p.mode == 1u) {
        return vec4<f32>(a, a, a, 1.0);   // 마스크만 보기
    }
    let fg = textureSampleLevel(FRAME, SAMP, uv, 0.0).rgb;
    var bg: vec3<f32>;
    if (p.bg == 1u) {
        bg = vec3<f32>(0.0);
    } else if (p.bg == 2u) {
        // 프레임 블러 근사 — 넓게 흩뿌린 9탭 (배경 이미지 없이도 가상배경 느낌)
        var sum = vec3<f32>(0.0);
        let r = 6.0 / dim;
        for (var i = -1; i <= 1; i = i + 1) {
            for (var j = -1; j <= 1; j = j + 1) {
                sum = sum + textureSampleLevel(
                    FRAME, SAMP, uv + vec2<f32>(f32(i), f32(j)) * r, 0.0).rgb;
            }
        }
        bg = sum / 9.0;
    } else {
        // 그라디언트
        let t = clamp((uv.x + uv.y) * 0.5, 0.0, 1.0);
        bg = mix(vec3<f32>(0.118, 0.227, 0.372), vec3<f32>(0.482, 0.176, 0.369), t);
    }
    let outc = bg * (1.0 - a) + fg * a;
    return vec4<f32>(outc, 1.0);
}
"#;



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

pub struct Presenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    size: (u32, u32),
    /// 출력 버퍼를 옮겨 담는 중간 텍스처 + 뷰 + 바인드그룹.
    /// 매 프레임 새로 만들면 WebGPU 백엔드에서 JS 객체가 쌓여 GC 압력이 된다.
    staging: std::cell::RefCell<Option<Staged>>,
    sampler: wgpu::Sampler,
    /// 카메라 프레임 텍스처 (표시 해상도)
    frame_tex: std::cell::RefCell<Option<(wgpu::Texture, wgpu::TextureView, u32, u32, u32)>>,
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
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        log::info!("[ai-wasm] present 준비 ({}x{}, {format:?}, {alpha_mode:?})", w, h);
        Ok(Self {
            surface,
            config,
            pipeline,
            bgl,
            params,
            size: (w, h),
            staging: std::cell::RefCell::new(None),
            sampler: ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("present-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            frame_tex: std::cell::RefCell::new(None),
        })
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    fn reconfigure(&self, ctx: &GpuContext) {
        self.surface.configure(&ctx.device, &self.config);
    }

    /// `src`(NHWC-C4 스토리지)의 채널 `ch`를 알파로 캔버스에 그린다.
    /// 카메라 프레임을 GPU 텍스처로 올린다 (CPU 왕복 없음).
    pub fn upload_frame(
        &self,
        ctx: &GpuContext,
        src: &wgpu::wgt::CopyExternalImageSourceInfo,
        w: u32,
        h: u32,
    ) -> Result<(), String> {
        let mut ft = self.frame_tex.borrow_mut();
        if ft.as_ref().map(|(_, _, fw, fh, _)| (*fw, *fh)) != Some((w, h)) {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("present-frame"),
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
            let genv = ft.as_ref().map(|(_, _, _, _, g)| g.wrapping_add(1)).unwrap_or(0);
            *ft = Some((tex, view, w, h, genv));
        }
        let (tex, _, _, _, _) = ft.as_ref().unwrap();
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
        Ok(())
    }

    /// 마스크 + 프레임을 합성해 캔버스에 그린다. mode 1 = 마스크만, bg 0/1/2.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        src: &wgpu::Buffer,
        desc: &TensorDesc,
        ch: u32,
        mode: u32,
        bg: u32,
    ) -> Result<(), String> {
        let p = [desc.w, desc.h, desc.cg(), ch, mode, bg, 0u32, 0u32];
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));

        // NHWC-C4 버퍼 → rgba32float 텍스처. 텍스처 폭 = W * cg (채널그룹 인터리브).
        // copy_buffer_to_texture는 bytes_per_row가 256의 배수여야 한다.
        let (tw, th) = (desc.w * desc.cg(), desc.h);
        let bytes_per_row = tw * 16;
        if bytes_per_row % 256 != 0 {
            return Err(format!(
                "present: 행 바이트 {bytes_per_row}가 256 정렬이 아님 (W={} cg={})",
                desc.w,
                desc.cg()
            ));
        }
        let mut st = self.staging.borrow_mut();
        if st.as_ref().map(|s2| (s2.w, s2.h)) != Some((tw, th)) {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("present-staging"),
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
        let ft = self.frame_tex.borrow();
        let (_, frame_view, _, _, genv) =
            ft.as_ref().ok_or_else(|| "upload_frame() 먼저".to_string())?;
        {
            let staged = st.as_mut().unwrap();
            if staged.bind.is_none() || staged.frame_gen != *genv {
                staged.bind = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("present-bind"),
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
                            resource: wgpu::BindingResource::TextureView(frame_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }));
                staged.frame_gen = *genv;
            }
        }
        let staged = st.as_ref().unwrap();
        let tex = &staged.tex;
        let bind = staged.bind.as_ref().unwrap();
        // wgpu 30: get_current_texture는 Result가 아니라 상태 enum을 준다.
        // Outdated/Lost는 정상적으로 발생할 수 있다 — 재구성하고 이번 프레임은 건너뛴다.
        // (여기서 하드 에러를 내면 호스트 루프가 죽는다. 실제로 그렇게 만들었었다.)
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                log::warn!("[ai-wasm] 서피스 상태 {other:?} — 재구성 후 이 프레임 건너뜀");
                self.reconfigure(ctx);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: src,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(th),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
        );
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
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit([enc.finish()]);
        drop(frame); // wgpu 30은 SurfaceTexture drop 시점에 present 한다
        Ok(())
    }
}
