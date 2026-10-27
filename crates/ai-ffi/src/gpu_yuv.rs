//! YUV ↔ RGBA 를 GPU 에서 — render_mask 의 프레임 왕복을 CPU(yuv.rs) 대신 GPU 가 한다.
//!
//! - 올리기: 평면을 그대로 텍스처로 올리고(I420 = R8 세 장 · NV12 = R8 + RG8 두 장 —
//!   RGBA 의 3/8 크기), 셰이더가 RGBA 로 바꿔 프레임 텍스처에 그린다. 카메라 회전(90/180/270 —
//!   libyuv I420Rotate 와 같은 시계 방향)과 mirror/180 뒤집기도 여기서 한다.
//! - 내리기: 합성 타깃을 셰이더가 YUV 바이트로 바꿔 버퍼 한 줄(Y 다음 크로마)에 싣고, 그 줄만
//!   읽어 호출자 평면에 stride 대로 복사한다 — RGBA 리드백과 CPU 변환이 없다. 버퍼는 크기·
//!   형식이 바뀔 때만 다시 만든다.
//!
//! NV12 는 iOS 카메라 버퍼(CVPixelBuffer)의 형식이다 — 그대로 받아 그대로 내보내면 호스트의
//! NV12→I420 변환과 되돌리는 인터리브 복사가 없어진다. 크로마 값은 I420 과 같다(배치만 다름).
//!
//! 변환식은 yuv.rs 와 같다(BT.601 full-range · 절사 · 크로마 = 2×2 블록 좌상단). 셰이더
//! 컴파일러가 곱셈·덧셈을 합치는(FMA) 정도에 따라 경계값에서 드물게 ±1 이 날 수 있다 —
//! 아래 테스트가 상한을 잰다. CPU 픽셀이 필요한 프레임(B 티어 세그 입력 · 태스크 픽셀 틱)은
//! 호출자가 yuv.rs 로 간다.

use ai_gpu::wgpu;
use ai_gpu::GpuContext;

const UPLOAD_WGSL: &str = r#"
struct Params { w: u32, h: u32, flip_h: u32, flip_v: u32, nv12: u32, rot: u32, sw: u32, sh: u32 };
@group(0) @binding(0) var ty: texture_2d<f32>;
@group(0) @binding(1) var tu: texture_2d<f32>;
@group(0) @binding(2) var tv: texture_2d<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // 화면을 덮는 삼각형 하나
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

fn to_byte(v: f32) -> f32 {
    return round(v * 255.0);
}

// yuv.rs 의 `clamp(0, 255) as u8` — 음수가 없으니 절사 = floor
fn to_unorm(v: f32) -> f32 {
    return floor(clamp(v, 0.0, 255.0)) / 255.0;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var x = u32(pos.x);
    var y = u32(pos.y);
    if (p.flip_h != 0u) { x = p.w - 1u - x; }
    if (p.flip_v != 0u) { y = p.h - 1u - y; }
    // 세운 좌표(w×h) → 원본 평면(sw×sh). libyuv I420Rotate(시계 방향)의 역사상 —
    // 크로마는 원본 좌표의 절반이라 짝수 크기에서 평면 회전과 같다(홀수는 호출자가 거른다).
    var sx = x;
    var sy = y;
    if (p.rot == 90u) {
        sx = y;
        sy = p.sh - 1u - x;
    } else if (p.rot == 180u) {
        sx = p.sw - 1u - x;
        sy = p.sh - 1u - y;
    } else if (p.rot == 270u) {
        sx = p.sw - 1u - y;
        sy = x;
    }
    let yv = to_byte(textureLoad(ty, vec2<u32>(sx, sy), 0).r);
    let c = vec2<u32>(sx / 2u, sy / 2u);
    // NV12 는 tu = tv = UV 텍스처(r = U, g = V)
    let uv = to_byte(textureLoad(tu, c, 0).r) - 128.0;
    let cv = textureLoad(tv, c, 0);
    let vv = to_byte(select(cv.r, cv.g, p.nv12 != 0u)) - 128.0;
    return vec4<f32>(
        to_unorm(yv + 1.402 * vv),
        to_unorm(yv - 0.344136 * uv - 0.714136 * vv),
        to_unorm(yv + 1.772 * uv),
        1.0,
    );
}
"#;

const PACK_WGSL: &str = r#"
struct Params { w: u32, h: u32, cw: u32, ch: u32, y_words: u32, c_words: u32, row: u32, nv12: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn rgb_at(x: u32, y: u32) -> vec3<f32> {
    return round(textureLoad(src, vec2<u32>(x, y), 0).rgb * 255.0);
}

// yuv.rs 의 `clamp(0, 255) as u8` — f32 → u32 변환은 0 쪽으로 절사
fn to_byte(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 255.0));
}

fn y_of(c: vec3<f32>) -> u32 {
    return to_byte(0.299 * c.r + 0.587 * c.g + 0.114 * c.b);
}

fn u_of(c: vec3<f32>) -> u32 {
    return to_byte(-0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b + 128.0);
}

fn v_of(c: vec3<f32>) -> u32 {
    return to_byte(0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b + 128.0);
}

// 워드 하나 = 출력 바이트 4개(리틀 엔디언). 줄 배치는 Y 워드들 다음 크로마 —
// I420 은 U 워드들 · V 워드들, NV12 는 UV 인터리브 워드들.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.row + gid.x;
    var word = 0u;
    if (i < p.y_words) {
        let n = p.w * p.h;
        for (var k = 0u; k < 4u; k++) {
            let px = i * 4u + k;
            if (px < n) {
                word |= y_of(rgb_at(px % p.w, px / p.w)) << (8u * k);
            }
        }
    } else if (p.nv12 != 0u) {
        if (i >= p.y_words + p.c_words) {
            return;
        }
        let j = i - p.y_words;
        let rw = 2u * p.cw;
        let n = rw * p.ch;
        for (var k = 0u; k < 4u; k++) {
            let q = j * 4u + k;
            if (q < n) {
                let col = q % rw;
                let c = rgb_at((col / 2u) * 2u, (q / rw) * 2u);
                word |= select(u_of(c), v_of(c), (col & 1u) == 1u) << (8u * k);
            }
        }
    } else if (i < p.y_words + 2u * p.c_words) {
        let j = i - p.y_words;
        let is_v = j >= p.c_words;
        let base = select(j, j - p.c_words, is_v);
        let n = p.cw * p.ch;
        for (var k = 0u; k < 4u; k++) {
            let q = base * 4u + k;
            if (q < n) {
                let c = rgb_at((q % p.cw) * 2u, (q / p.cw) * 2u);
                word |= select(u_of(c), v_of(c), is_v) << (8u * k);
            }
        }
    } else {
        return;
    }
    out[i] = word;
}
"#;

const PACK_WG: u32 = 256;
const MAX_GROUPS: u32 = 65_535;

/// 평면 형식 — I420(Y·U·V 세 장) / NV12(Y + UV 인터리브 두 장).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fmt {
    I420,
    Nv12,
}

/// 올릴 프레임 — 평면과 stride(바이트).
pub enum Src<'a> {
    I420 { y: &'a [u8], u: &'a [u8], v: &'a [u8], strides: [u32; 3] },
    Nv12 { y: &'a [u8], uv: &'a [u8], strides: [u32; 2] },
}

/// 내려받을 평면 — stride 밖 여백은 건드리지 않는다.
pub enum Dst<'a> {
    I420 { y: &'a mut [u8], u: &'a mut [u8], v: &'a mut [u8], strides: [usize; 3] },
    Nv12 { y: &'a mut [u8], uv: &'a mut [u8], strides: [usize; 2] },
}

/// 한 줄 배치 — Y 다음 크로마. 각 평면은 4바이트 워드 경계에서 시작한다.
#[derive(Clone, Copy)]
struct Layout {
    cw: u32,
    ch: u32,
    y_words: u32,
    /// 크로마 평면 하나의 워드 수 (I420 은 이런 평면이 둘)
    c_words: u32,
    fmt: Fmt,
}

impl Layout {
    fn new(w: u32, h: u32, fmt: Fmt) -> Self {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let c_bytes = match fmt {
            Fmt::I420 => cw * ch,
            Fmt::Nv12 => 2 * cw * ch,
        };
        Self { cw, ch, y_words: (w * h).div_ceil(4), c_words: c_bytes.div_ceil(4), fmt }
    }

    fn bytes(&self) -> u64 {
        let planes = match self.fmt {
            Fmt::I420 => 2,
            Fmt::Nv12 => 1,
        };
        (self.y_words + planes * self.c_words) as u64 * 4
    }
}

struct Planes {
    w: u32,
    h: u32,
    fmt: Fmt,
    y: wgpu::Texture,
    /// I420 = [U, V] · NV12 = [UV]
    c: Vec<wgpu::Texture>,
    bind: wgpu::BindGroup,
}

struct Packed {
    w: u32,
    h: u32,
    fmt: Fmt,
    storage: wgpu::Buffer,
    staging: wgpu::Buffer,
}

pub struct GpuYuv {
    up_pipe: wgpu::RenderPipeline,
    up_bgl: wgpu::BindGroupLayout,
    up_params: wgpu::Buffer,
    planes: Option<Planes>,
    pack_pipe: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    pack_params: wgpu::Buffer,
    packed: Option<Packed>,
}

fn le_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn uniform_buffer(ctx: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// 평면 `pw×ph`(촘촘한 줄)를 호출자 평면에 stride 대로 옮긴다.
fn copy_plane(src: &[u8], pw: usize, ph: usize, dst: &mut [u8], stride: usize) {
    for r in 0..ph {
        dst[r * stride..r * stride + pw].copy_from_slice(&src[r * pw..r * pw + pw]);
    }
}

/// 평면 하나를 텍스처로 — `bytes_per_row` 는 호출자 stride 그대로.
fn write_plane(ctx: &GpuContext, tex: &wgpu::Texture, data: &[u8], stride: u32) {
    let size = tex.size();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride),
            rows_per_image: Some(size.height),
        },
        size,
    );
}

impl GpuYuv {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let up_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ai-ffi-yuv-upload"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                texture_entry(2, wgpu::ShaderStages::FRAGMENT),
                buffer_entry(3, wgpu::ShaderStages::FRAGMENT, wgpu::BufferBindingType::Uniform),
            ],
        });
        let up_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ai-ffi-yuv-upload"),
            source: wgpu::ShaderSource::Wgsl(UPLOAD_WGSL.into()),
        });
        let up_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ai-ffi-yuv-upload"),
            bind_group_layouts: &[Some(&up_bgl)],
            ..Default::default()
        });
        let up_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ai-ffi-yuv-upload"),
            layout: Some(&up_layout),
            vertex: wgpu::VertexState {
                module: &up_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &up_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ai-ffi-yuv-pack"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::COMPUTE),
                buffer_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                buffer_entry(2, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ai-ffi-yuv-pack"),
            source: wgpu::ShaderSource::Wgsl(PACK_WGSL.into()),
        });
        let pack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ai-ffi-yuv-pack"),
            bind_group_layouts: &[Some(&pack_bgl)],
            ..Default::default()
        });
        let pack_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ai-ffi-yuv-pack"),
            layout: Some(&pack_layout),
            module: &pack_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            up_pipe,
            up_bgl,
            up_params: uniform_buffer(ctx, "ai-ffi-yuv-upload", 32),
            planes: None,
            pack_pipe,
            pack_bgl,
            pack_params: uniform_buffer(ctx, "ai-ffi-yuv-pack", 32),
            packed: None,
        }
    }

    fn ensure_planes(&mut self, ctx: &GpuContext, w: u32, h: u32, fmt: Fmt) {
        if self.planes.as_ref().is_some_and(|p| p.w == w && p.h == h && p.fmt == fmt) {
            return;
        }
        let make = |label: &str, pw: u32, ph: u32, format| {
            ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: pw, height: ph, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = make("ai-ffi-yuv-y", w, h, wgpu::TextureFormat::R8Unorm);
        let c = match fmt {
            Fmt::I420 => vec![
                make("ai-ffi-yuv-u", cw, ch, wgpu::TextureFormat::R8Unorm),
                make("ai-ffi-yuv-v", cw, ch, wgpu::TextureFormat::R8Unorm),
            ],
            Fmt::Nv12 => vec![make("ai-ffi-yuv-uv", cw, ch, wgpu::TextureFormat::Rg8Unorm)],
        };
        let yv = y.create_view(&Default::default());
        let uv = c[0].create_view(&Default::default());
        // NV12 는 U·V 를 같은 텍스처에서 읽는다 (셰이더가 채널로 가른다)
        let vv = c.last().unwrap().create_view(&Default::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ai-ffi-yuv-upload"),
            layout: &self.up_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&yv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&vv) },
                wgpu::BindGroupEntry { binding: 3, resource: self.up_params.as_entire_binding() },
            ],
        });
        self.planes = Some(Planes { w, h, fmt, y, c, bind });
    }

    fn ensure_packed(&mut self, ctx: &GpuContext, w: u32, h: u32, lay: Layout) {
        if self.packed.as_ref().is_some_and(|p| p.w == w && p.h == h && p.fmt == lay.fmt) {
            return;
        }
        let size = lay.bytes();
        let make = |label: &str, usage| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        self.packed = Some(Packed {
            w,
            h,
            fmt: lay.fmt,
            storage: make(
                "ai-ffi-yuv-packed",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: make(
                "ai-ffi-yuv-staging",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        });
    }

    /// 프레임 평면(`sw×sh`) → `frame`(Rgba8Unorm, RENDER_ATTACHMENT — 회전 90/270 이면 `sh×sw`).
    /// (호스트 회전 +) yuv::i420_to_rgba + flip_rgba 와 같은 결과를 GPU 에서 낸다.
    /// `rot` 은 0/90/180/270 — 0 이 아니면 sw·sh 가 짝수여야 한다.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        frame: &wgpu::Texture,
        src: Src,
        sw: u32,
        sh: u32,
        rot: u32,
        flip_h: bool,
        flip_v: bool,
    ) {
        let fmt = match src {
            Src::I420 { .. } => Fmt::I420,
            Src::Nv12 { .. } => Fmt::Nv12,
        };
        let (w, h) = (frame.width(), frame.height());
        self.ensure_planes(ctx, sw, sh, fmt);
        let planes = self.planes.as_ref().unwrap();
        match src {
            Src::I420 { y, u, v, strides } => {
                write_plane(ctx, &planes.y, y, strides[0]);
                write_plane(ctx, &planes.c[0], u, strides[1]);
                write_plane(ctx, &planes.c[1], v, strides[2]);
            }
            Src::Nv12 { y, uv, strides } => {
                write_plane(ctx, &planes.y, y, strides[0]);
                write_plane(ctx, &planes.c[0], uv, strides[1]);
            }
        }
        let params = [w, h, flip_h as u32, flip_v as u32, (fmt == Fmt::Nv12) as u32, rot, sw, sh];
        ctx.queue.write_buffer(&self.up_params, 0, &le_words(&params));

        let view = frame.create_view(&Default::default());
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ai-ffi-yuv-upload"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ai-ffi-yuv-upload"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.up_pipe);
            pass.set_bind_group(0, &planes.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit([enc.finish()]);
    }

    /// `src`(Rgba8Unorm, `w×h`, TEXTURE_BINDING) → 호출자 평면.
    /// yuv::rgba_to_i420 (NV12 면 그 크로마를 인터리브)와 같은 결과를 GPU 에서 낸다 —
    /// GPU 가 끝날 때까지 기다린다.
    pub fn download(
        &mut self,
        ctx: &GpuContext,
        src: &wgpu::TextureView,
        w: u32,
        h: u32,
        dst: Dst,
    ) -> Result<(), String> {
        let fmt = match dst {
            Dst::I420 { .. } => Fmt::I420,
            Dst::Nv12 { .. } => Fmt::Nv12,
        };
        let lay = Layout::new(w, h, fmt);
        self.ensure_packed(ctx, w, h, lay);
        let result = self.pack_and_read(ctx, src, w, h, lay, dst);
        if result.is_err() {
            // 매핑이 걸린 채 남았을 수 있다 — 다음 프레임은 새 버퍼로
            self.packed = None;
        }
        result
    }

    fn pack_and_read(
        &self,
        ctx: &GpuContext,
        src: &wgpu::TextureView,
        w: u32,
        h: u32,
        lay: Layout,
        dst: Dst,
    ) -> Result<(), String> {
        let packed = self.packed.as_ref().unwrap();
        let size = lay.bytes();
        let groups = (size / 4).div_ceil(PACK_WG as u64) as u32;
        let gx = groups.min(MAX_GROUPS);
        let gy = groups.div_ceil(gx);
        let nv12 = (lay.fmt == Fmt::Nv12) as u32;
        ctx.queue.write_buffer(
            &self.pack_params,
            0,
            &le_words(&[w, h, lay.cw, lay.ch, lay.y_words, lay.c_words, gx * PACK_WG, nv12]),
        );
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ai-ffi-yuv-pack"),
            layout: &self.pack_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: packed.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.pack_params.as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ai-ffi-yuv-pack"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ai-ffi-yuv-pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&packed.storage, 0, &packed.staging, 0, size);
        ctx.queue.submit([enc.finish()]);

        let slice = packed.staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| format!("device.poll 실패: {e:?}"))?;
        rx.recv()
            .map_err(|_| "map_async 콜백 유실".to_string())?
            .map_err(|e| format!("buffer map 실패: {e:?}"))?;
        {
            let data = slice
                .get_mapped_range()
                .map_err(|e| format!("get_mapped_range 실패: {e:?}"))?;
            let (cw, ch) = (lay.cw as usize, lay.ch as usize);
            let c_off = lay.y_words as usize * 4;
            match dst {
                Dst::I420 { y, u, v, strides } => {
                    let v_off = (lay.y_words + lay.c_words) as usize * 4;
                    copy_plane(&data[..], w as usize, h as usize, y, strides[0]);
                    copy_plane(&data[c_off..], cw, ch, u, strides[1]);
                    copy_plane(&data[v_off..], cw, ch, v, strides[2]);
                }
                Dst::Nv12 { y, uv, strides } => {
                    copy_plane(&data[..], w as usize, h as usize, y, strides[0]);
                    copy_plane(&data[c_off..], 2 * cw, ch, uv, strides[1]);
                }
            }
        }
        packed.staging.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yuv;

    fn noise(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn rgba_texture(ctx: &GpuContext, w: u32, h: u32) -> wgpu::Texture {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-rgba"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn read_rgba(ctx: &GpuContext, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
        let bpr = (w * 4).next_multiple_of(256);
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        ctx.queue.submit([enc.finish()]);
        let bytes = pollster::block_on(ai_gpu::readback::read_buffers(ctx, &[&staging]))
            .unwrap()
            .remove(0);
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for r in 0..h as usize {
            out.extend_from_slice(&bytes[r * bpr as usize..r * bpr as usize + (w * 4) as usize]);
        }
        out
    }

    /// (다른 바이트 수, 최대 차이)
    fn diff(a: &[u8], b: &[u8]) -> (usize, u8) {
        a.iter().zip(b).fold((0, 0), |(n, m), (x, y)| {
            let d = x.abs_diff(*y);
            (n + (d != 0) as usize, m.max(d))
        })
    }

    fn assert_close(label: &str, want: &[u8], got: &[u8]) {
        let (n, m) = diff(want, got);
        eprintln!("{label}: 다른 바이트 {n}/{} 최대 {m}", want.len());
        assert!(m <= 1, "{label} 최대 차이 {m}");
        assert!(n * 1000 <= want.len(), "{label} 다른 바이트 {n} 이 0.1% 초과");
    }

    /// I420 크로마 두 장 → NV12 UV 한 장 (stride `suv`, 여백은 `fill`)
    fn interleave(u: &[u8], v: &[u8], sc: usize, cw: usize, ch: usize, suv: usize, fill: u8) -> Vec<u8> {
        let mut uv = vec![fill; suv * ch];
        for r in 0..ch {
            for c in 0..cw {
                uv[r * suv + 2 * c] = u[r * sc + c];
                uv[r * suv + 2 * c + 1] = v[r * sc + c];
            }
        }
        uv
    }

    fn context() -> Option<GpuContext> {
        let _ = env_logger::builder().is_test(true).try_init();
        match GpuContext::new_blocking() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("GPU 없음 — 건너뜀: {e}");
                None
            }
        }
    }

    // (w, h, Y stride, I420 크로마 stride, NV12 UV stride) — 짝수/홀수 크기 · stride > 폭
    const SIZES: [(usize, usize, usize, usize, usize); 3] = [
        (1280, 720, 1280, 640, 1280),
        (641, 359, 704, 352, 704),
        (64, 48, 96, 40, 80),
    ];

    #[test]
    fn upload_matches_cpu() {
        let Some(ctx) = context() else { return };
        let mut g = GpuYuv::new(&ctx);
        for (w, h, sy, sc, suv) in SIZES {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let (y, u, v) = (noise(sy * h, 1), noise(sc * ch, 2), noise(sc * ch, 3));
            let uv = interleave(&u, &v, sc, cw, ch, suv, 0xCD);
            let frame = rgba_texture(&ctx, w as u32, h as u32);
            for (fh, fv) in [(false, false), (true, false), (true, true)] {
                let mut want = vec![0u8; w * h * 4];
                yuv::i420_to_rgba(&y, &u, &v, w, h, sy, sc, sc, &mut want);
                yuv::flip_rgba(&mut want, w, h, fh, fv);
                let strides = [sy as u32, sc as u32, sc as u32];
                g.upload(&ctx, &frame, Src::I420 { y: &y, u: &u, v: &v, strides }, w as u32, h as u32, 0, fh, fv);
                let got = read_rgba(&ctx, &frame, w as u32, h as u32);
                assert_close(&format!("upload i420 {w}x{h} flip=({fh},{fv})"), &want, &got);
                let strides = [sy as u32, suv as u32];
                g.upload(&ctx, &frame, Src::Nv12 { y: &y, uv: &uv, strides }, w as u32, h as u32, 0, fh, fv);
                let got_nv12 = read_rgba(&ctx, &frame, w as u32, h as u32);
                assert_close(&format!("upload nv12 {w}x{h} flip=({fh},{fv})"), &want, &got_nv12);
                assert_eq!(got, got_nv12, "NV12 와 I420 올리기는 같은 텍스처를 내야 한다");
            }
        }
    }

    /// libyuv RotatePlane 그대로 — 90 = 원본을 아래부터 읽어 전치, 270 = 전치 후 대상을 아래부터,
    /// 180 = 양방향 뒤집기 (ffmpeg transpose 와 대조해 맞춘 식). 반환 = (평면, 폭, 높이), stride = 폭.
    fn rotate_plane(src: &[u8], stride: usize, w: usize, h: usize, rot: u32) -> (Vec<u8>, usize, usize) {
        let at = |x: usize, y: usize| src[y * stride + x];
        match rot {
            90 => {
                let mut out = vec![0u8; w * h];
                for i in 0..w {
                    for j in 0..h {
                        out[i * h + j] = at(i, h - 1 - j);
                    }
                }
                (out, h, w)
            }
            270 => {
                let mut out = vec![0u8; w * h];
                for i in 0..w {
                    for j in 0..h {
                        out[(w - 1 - i) * h + j] = at(i, j);
                    }
                }
                (out, h, w)
            }
            180 => {
                let mut out = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        out[y * w + x] = at(w - 1 - x, h - 1 - y);
                    }
                }
                (out, w, h)
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn upload_rotation_matches_libyuv() {
        let Some(ctx) = context() else { return };
        let mut g = GpuYuv::new(&ctx);
        // 카메라 프레임 모양 — 짝수 크기, stride 여백 있음/없음
        for (w, h, sy, sc) in [(1280usize, 720usize, 1280usize, 640usize), (64, 48, 96, 40)] {
            let (cw, ch) = (w / 2, h / 2);
            let (y, u, v) = (noise(sy * h, 11), noise(sc * ch, 12), noise(sc * ch, 13));
            for rot in [90u32, 180, 270] {
                let (ry, dw, dh) = rotate_plane(&y, sy, w, h, rot);
                let (ru, dcw, _) = rotate_plane(&u, sc, cw, ch, rot);
                let (rv, _, _) = rotate_plane(&v, sc, cw, ch, rot);
                let frame = rgba_texture(&ctx, dw as u32, dh as u32);
                for (fh, fv) in [(false, false), (true, false), (true, true)] {
                    let mut want = vec![0u8; dw * dh * 4];
                    yuv::i420_to_rgba(&ry, &ru, &rv, dw, dh, dw, dcw, dcw, &mut want);
                    yuv::flip_rgba(&mut want, dw, dh, fh, fv);
                    let strides = [sy as u32, sc as u32, sc as u32];
                    g.upload(&ctx, &frame, Src::I420 { y: &y, u: &u, v: &v, strides }, w as u32, h as u32, rot, fh, fv);
                    let got = read_rgba(&ctx, &frame, dw as u32, dh as u32);
                    assert_close(&format!("upload rot{rot} {w}x{h} flip=({fh},{fv})"), &want, &got);
                }
            }
        }
    }

    #[test]
    fn download_matches_cpu() {
        let Some(ctx) = context() else { return };
        let mut g = GpuYuv::new(&ctx);
        for (w, h, sy, sc, suv) in SIZES {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let rgba = {
                let mut px = noise(w * h * 4, 7);
                px.chunks_exact_mut(4).for_each(|p| p[3] = 255);
                px
            };
            let tex = rgba_texture(&ctx, w as u32, h as u32);
            ctx.queue.write_texture(
                tex.as_image_copy(),
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((w * 4) as u32),
                    rows_per_image: Some(h as u32),
                },
                wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            );
            let view = tex.create_view(&Default::default());
            // stride 여백에 쓰레기 — 건드리지 않아야 한다
            let (mut wy, mut wu, mut wv) = (vec![0xAB; sy * h], vec![0xAB; sc * ch], vec![0xAB; sc * ch]);
            yuv::rgba_to_i420(&rgba, w, h, &mut wy, &mut wu, &mut wv, sy, sc, sc);

            let (mut gy, mut gu, mut gv) = (vec![0xAB; sy * h], vec![0xAB; sc * ch], vec![0xAB; sc * ch]);
            let dst = Dst::I420 { y: &mut gy, u: &mut gu, v: &mut gv, strides: [sy, sc, sc] };
            g.download(&ctx, &view, w as u32, h as u32, dst).unwrap();
            for (name, want, got) in [("Y", &wy, &gy), ("U", &wu, &gu), ("V", &wv, &gv)] {
                assert_close(&format!("download i420 {w}x{h} {name}"), want, got);
            }

            let want_uv = interleave(&wu, &wv, sc, cw, ch, suv, 0xAB);
            let (mut ny, mut nuv) = (vec![0xAB; sy * h], vec![0xAB; suv * ch]);
            let dst = Dst::Nv12 { y: &mut ny, uv: &mut nuv, strides: [sy, suv] };
            g.download(&ctx, &view, w as u32, h as u32, dst).unwrap();
            assert_close(&format!("download nv12 {w}x{h} Y"), &wy, &ny);
            assert_close(&format!("download nv12 {w}x{h} UV"), &want_uv, &nuv);
            assert_eq!(gy, ny, "NV12 와 I420 의 Y 는 같아야 한다");
            assert_eq!(interleave(&gu, &gv, sc, cw, ch, suv, 0xAB), nuv, "NV12 UV = I420 U·V 인터리브");
        }
    }

    /// 720p 한 프레임 왕복 시간 — 예전 경로(CPU 변환 + RGBA 올림 + 프레임마다 새 스테이징으로
    /// RGBA 리드백) vs GPU 평면 경로. `cargo test -p ai-ffi --lib --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn roundtrip_timing() {
        let Some(ctx) = context() else { return };
        let (w, h) = (1280usize, 720usize);
        let (mut y, mut u, mut v) = (noise(w * h, 1), noise(w * h / 4, 2), noise(w * h / 4, 3));
        let frame = rgba_texture(&ctx, w as u32, h as u32);
        let view = frame.create_view(&Default::default());
        let mut g = GpuYuv::new(&ctx);
        let n = 60;
        let mut rgba = vec![0u8; w * h * 4];
        let t = std::time::Instant::now();
        for _ in 0..n {
            yuv::i420_to_rgba(&y, &u, &v, w, h, w, w / 2, w / 2, &mut rgba);
            yuv::flip_rgba(&mut rgba, w, h, true, false);
            ctx.queue.write_texture(
                frame.as_image_copy(),
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((w * 4) as u32),
                    rows_per_image: Some(h as u32),
                },
                wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            );
            let back = read_rgba(&ctx, &frame, w as u32, h as u32);
            yuv::rgba_to_i420(&back, w, h, &mut y, &mut u, &mut v, w, w / 2, w / 2);
        }
        let old = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let strides = [w as u32, (w / 2) as u32, (w / 2) as u32];
            g.upload(&ctx, &frame, Src::I420 { y: &y, u: &u, v: &v, strides }, w as u32, h as u32, 0, true, false);
            let dst = Dst::I420 { y: &mut y, u: &mut u, v: &mut v, strides: [w, w / 2, w / 2] };
            g.download(&ctx, &view, w as u32, h as u32, dst).unwrap();
        }
        let gpu = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        eprintln!("720p 왕복: 예전 {old:.2} ms / GPU 평면 {gpu:.2} ms");
    }
}
