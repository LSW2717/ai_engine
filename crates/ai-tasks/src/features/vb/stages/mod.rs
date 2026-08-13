//! 스테이지 공통 골격 — 새 영상처리 스테이지 추가 절차 (확장 지점):
//!
//! 1. `video/shaders/<이름>.wgsl` — vs_fullscreen + fs (자체 완결, naga 테스트 대상)
//! 2. `video/stages/<이름>.rs` — 이 파일의 `Fullscreen`/`ubo`로 파이프라인 구성
//!    + `#[cfg(test)] wgsl_valid` (naga 검증 — GPU 없이 0.2초)
//! 3. `video/mod.rs`의 `Resources`에 입출력 텍스처·바인드그룹, `encode_effects`에
//!    호출 순서 한 줄
//!
//! dyn Stage 트레이트를 두지 않는 이유: 스테이지는 6~12개 고정 순서의
//! 이미지 패스라 동적 DAG가 주는 게 없고, 리소스 수명·재사용(성능의 본체)을
//! 파이프라인이 한눈에 소유하는 쪽이 저사양 최적화(패스 융합·텍스처 재사용)에
//! 유리하다. 확장 비용은 위 3단계 = 파일 1개 + 배선 몇 줄.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

/// 풀스크린 삼각형 render 파이프라인 (draw 0..3) — 바인딩 종류만 선언하면 된다
pub(crate) struct Fullscreen {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
}

/// 바인딩 선언 축약 — 순서가 곧 binding 번호
#[derive(Clone, Copy)]
pub(crate) enum Bind {
    Uniform,
    /// 필터링 가능 텍스처 (rgba8/r8 — 선형 샘플)
    Tex,
    /// 필터링 불가 텍스처 (rgba32float 등 — texelFetch/nearest)
    TexNf,
    Sampler,
    SamplerNf,
}

impl Fullscreen {
    pub fn new(
        ctx: &GpuContext,
        label: &str,
        wgsl: &str,
        target: wgpu::TextureFormat,
        binds: &[Bind],
    ) -> Self {
        Self::new_entry(ctx, label, wgsl, target, binds, "fs")
    }

    /// fs 엔트리포인트를 지정하는 변형 — 한 wgsl에 패스 여러 개일 때 (mask_refine)
    pub fn new_entry(
        ctx: &GpuContext,
        label: &str,
        wgsl: &str,
        target: wgpu::TextureFormat,
        binds: &[Bind],
        entry: &str,
    ) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: match b {
                    Bind::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Bind::Tex => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    Bind::TexNf => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    Bind::Sampler => {
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                    }
                    Bind::SamplerNf => {
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)
                    }
                },
                count: None,
            })
            .collect();
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(entry),
                targets: &[Some(target.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Fullscreen { pipeline, bgl }
    }

    /// 단일 패스 기록 — LoadOp은 항상 덮어쓰기(Clear 불필요한 풀스크린)
    pub fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        bind: &wgpu::BindGroup,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
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
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub(crate) fn ubo(ctx: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(crate) fn tex2d(
    ctx: &GpuContext,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    extra: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let t = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | extra,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}

/// 공용 풀스크린 VS — 각 스테이지 wgsl 앞에 이어 붙인다 (자체 완결 유지)
pub(crate) const FULLSCREEN_VS: &str = r#"
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(xy.x, 1.0 - xy.y);
    return out;
}
"#;

// ── 스테이지 목록 (1스테이지 = 1파일 + 1wgsl) ──
pub(crate) mod bbox;
pub(crate) mod bg_blur;
pub(crate) mod mask_refine;
pub(crate) mod compose;
pub(crate) mod mask_ingest;
pub(crate) mod mask_upsample;
pub(crate) mod preprocess;
