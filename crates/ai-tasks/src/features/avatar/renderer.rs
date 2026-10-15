// renderer — VRM 아바타 GPU 렌더 (스키닝 + 모프 + MToon + 아웃라인).
//
// items3d 와 달리 **전체 프레임 대체**다: 카메라 픽셀을 합성하지 않고 배경색
// clear 위에 아바타만 그려 target 에 리졸브한다. 세그/합성 스택이 통째로
// 필요 없어 VB 대비 발열이 오히려 준다.
//
// 프레임 입력은 얼굴 피팅 쿼터니언(items3d Horn 재사용)과 랜드마크 EAR 뿐.
// 손/스프링본은 후속 단계 (rig 가 자리를 갖고 있다).
//
// 파이프라인은 target 포맷에 종속이라 지연 생성(ensure_pipelines) — 서피스
// 포맷은 attach 후에야 안다 (ItemsOverlay.ensure_blit 와 같은 사정).

use std::collections::HashMap;

use ai_gpu::wgpu::util::{BufferInitDescriptor, DeviceExt};
use ai_gpu::wgpu::*;
use ai_gpu::GpuContext;

use super::rig::{
    blink_from_landmarks, mat_compose, mat_mul, mat_perspective, mouth_from_landmarks,
    quat_axis_angle, quat_mul, quat_slerp, BlinkCtl, MouthCtl, Quat, Rig,
};
use super::vrm::{parse_vrm, VrmModel};
use crate::features::face::items3d::face_pose_from_landmarks;

type Result<T> = std::result::Result<T, String>;

const MAX_JOINTS: usize = 96;
const MSAA_SAMPLES: u32 = 4;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24Plus;
const VFOV_DEG: f32 = 30.0;
/// 보유 에셋(VRM 0.x)은 무회전 배치에서 등을 보였다 (2026-08-26 실측) —
/// Y축 180° 로 카메라(원점, -Z 시선) 쪽을 보게 한다. rig 의 머리 회전
/// 켤레변환이 같은 값을 쓰므로 좌우도 함께 맞는다.
const ROOT_YAW: f32 = std::f32::consts::PI;
/// 머리 회전 저역통과 (items3d 스무딩의 고정 α 판 — 피팅 자체가 이미 EMA 라
/// 과한 이중 스무딩을 피한다)
const HEAD_SLERP_ALPHA: f32 = 0.45;
/// 머리 캘리브레이션 트림 — 정면 응시에도 화면 기준 왼쪽으로 살짝 기울어 보임
/// (2026-08-26 실측, Horn 피팅 계통 바이어스). 양수 = 화면 기준 오른쪽으로 기움.
const HEAD_ROLL_TRIM_DEG: f32 = 6.0;
/// 좌우 돌아봄(yaw) 트림 — 양수 = 화면 기준 오른쪽 보기.
/// 정면 응시에 왼쪽으로 ~30° 돌아가 보임 (2026-08-26 실측) — 우로 보정.
const HEAD_YAW_TRIM_DEG: f32 = 30.0;

/// 카메라계 트림 쿼터니언 — 피팅 결과에 선곱한다.
/// ⚠ 부호는 **실측 확정** (+Y 가 화면 왼쪽으로 나옴 — 2026-08-26 두 번 확인).
/// 축 해석으로 다시 뒤집지 말 것.
fn head_trim() -> Quat {
    quat_mul(
        quat_axis_angle([0.0, 1.0, 0.0], -HEAD_YAW_TRIM_DEG.to_radians()),
        quat_axis_angle([0.0, 0.0, 1.0], -HEAD_ROLL_TRIM_DEG.to_radians()),
    )
}
/// 프레임에서 상반신이 차지하는 비율 (거리 산정용 — 줌 수준)
const FRAME_FILL: f32 = 0.80;
/// head 본 위 정수리/머리카락 여유 — (head-hips) 구간 대비 비율 (거리 산정용)
const HEAD_MARGIN: f32 = 0.75;
/// 화면 세로 중앙에 오는 지점 — head 본에서 (head-hips) 구간 대비 위로 이만큼
/// 올라간 곳(≈얼굴 중심). "머리가 중앙" (사용자 지정 2026-08-26).
const HEAD_CENTER_OFFSET: f32 = 0.10;

// WGSL 반사 대조용 (mtoon_wgsl_parses 테스트) — items3d 와 같은 계약:
// 레이아웃·바인드그룹·셰이더 셋이 어긋나면 기기에서만 죽는다.
#[cfg_attr(not(test), allow(dead_code))]
const CAM_BINDING_COUNT: usize = 3;
#[cfg_attr(not(test), allow(dead_code))]
const MAT_BINDING_COUNT: usize = 5;

/// 아바타 배경 지정 — VB 배경 선택(EffectsState.background)을 아바타 렌더가
/// 직접 소비한다 (세그/합성 스택은 아바타 모드에서 안 돌므로).
#[derive(Clone, Copy)]
pub enum AvatarBg {
    /// 기본 스튜디오 단색
    Studio,
    Color([f32; 3]),
    /// set_background_image 로 업로드된 이미지 (미업로드면 Studio 폴백)
    Image,
}

/// 배경 풀스크린 패스 — cover-fit uv 는 유니폼(scale, offset).
const BG_WGSL: &str = r#"
struct BgParams { scale: vec2<f32>, offset: vec2<f32> }
@group(0) @binding(0) var<uniform> params: BgParams;
@group(0) @binding(1) var bg_tex: texture_2d<f32>;
@group(0) @binding(2) var bg_samp: sampler;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs_bg(@builtin(vertex_index) vi: u32) -> VOut {
    var out: VOut;
    let xy = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4<f32>(xy * 2.0 - 1.0, 1.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

fn srgb_oetf_bg(v: vec3<f32>) -> vec3<f32> {
    let hi = pow(v, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
    let lo = v * 12.92;
    return select(hi, lo, v <= vec3<f32>(0.0031308));
}

@fragment
fn fs_bg(in: VOut) -> @location(0) vec4<f32> {
    let uv = in.uv * params.scale + params.offset;
    let c = textureSample(bg_tex, bg_samp, uv).rgb;
    return vec4<f32>(srgb_oetf_bg(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AvatarVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    joints: [u16; 4],
    weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    model: [f32; 16],
    light_dir: [f32; 4],
    light_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    shade_color: [f32; 4],
    outline_color: [f32; 4],
    params: [f32; 4],
    mode: [f32; 4],
}

struct GpuPrim {
    vbuf: Buffer,
    ibuf: Buffer,
    index_count: u32,
    mat_bind: BindGroup,
    outline: bool,
    /// cutout 은 셰이더 discard 로 처리 (mode.x) — 별도 파이프라인 분기는
    /// 반투명(blend 2) 에셋이 생길 때 함께 (보유 6종 전부 opaque).
    #[allow(dead_code)]
    cutout: bool,
}

struct LoadedAvatar {
    prims: Vec<GpuPrim>,
    rig: Rig,
    /// 로드 시 계산한 고정 프레이밍: model 행렬 (yaw + 평행이동)
    base_model: [f32; 16],
    /// 프레이밍 파라미터 (px→모델 공간 역변환용)
    cam_dist: f32,
    center_y: f32,
    blink: BlinkCtl,
    mouth: MouthCtl,
    smooth_quat: Option<Quat>,
}

pub struct AvatarOverlay {
    // 파이프라인 (target 포맷 지연 생성)
    pipes: Option<(RenderPipeline, RenderPipeline, TextureFormat)>,
    shader: Option<ShaderModule>,
    cam_layout: BindGroupLayout,
    mat_layout: BindGroupLayout,
    cam_buf: Buffer,
    joints_buf: Buffer,
    morph_w_buf: Buffer,
    cam_bind: BindGroup,
    sampler: Sampler,
    white_view: TextureView,

    msaa: Option<Texture>,
    depth: Option<Texture>,
    size: (u32, u32),

    // 배경 이미지 패스 (VB 배경 재사용)
    bg_layout: BindGroupLayout,
    bg_params_buf: Buffer,
    bg_pipe: Option<(RenderPipeline, TextureFormat)>,
    bg_bind: Option<BindGroup>,
    bg_dims: (u32, u32),

    models: HashMap<String, LoadedAvatar>,
    active: String,
    /// 배경색 (linear RGB — clear 는 리니어에서, 셰이더 OETF 와 무관하게
    /// clear 값은 그대로 타깃에 쓰이므로 sRGB 값으로 지정한다)
    pub bg_srgb: [f64; 3],
}

impl AvatarOverlay {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let device = &ctx.device;
        let cam_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("avatar_cam_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let mat_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("avatar_mat_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cam_buf = device.create_buffer(&BufferDescriptor {
            label: Some("avatar_cam_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let joints_buf = device.create_buffer(&BufferDescriptor {
            label: Some("avatar_joints_buf"),
            size: (MAX_JOINTS * 64) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let morph_w_buf = device.create_buffer(&BufferDescriptor {
            label: Some("avatar_morph_w_buf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("avatar_cam_bind"),
            layout: &cam_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: joints_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: morph_w_buf.as_entire_binding() },
            ],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("avatar_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        // 1×1 흰 텍스처 — 텍스처 없는 머티리얼 폴백
        let white = ctx.device.create_texture(&TextureDescriptor {
            label: Some("avatar_white"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &white,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white_view = white.create_view(&TextureViewDescriptor::default());

        let bg_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("avatar_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bg_params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("avatar_bg_params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(AvatarOverlay {
            pipes: None,
            shader: None,
            cam_layout,
            mat_layout,
            cam_buf,
            joints_buf,
            morph_w_buf,
            cam_bind,
            sampler,
            white_view,
            msaa: None,
            depth: None,
            size: (0, 0),
            bg_layout,
            bg_params_buf,
            bg_pipe: None,
            bg_bind: None,
            bg_dims: (0, 0),
            models: HashMap::new(),
            active: String::new(),
            bg_srgb: [0.117, 0.129, 0.153], // #1e2127 스튜디오 다크
        })
    }

    /// VB 배경 이미지 업로드 (RGBA8) — background:"image" 일 때 아바타 뒤에 깔린다.
    pub fn set_background_image(&mut self, ctx: &GpuContext, rgba: &[u8], w: u32, h: u32) {
        if w == 0 || h == 0 || rgba.len() < (w * h * 4) as usize {
            return;
        }
        let tex = ctx.device.create_texture(&TextureDescriptor {
            label: Some("avatar_bg_tex"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            rgba,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&TextureViewDescriptor::default());
        self.bg_bind = Some(ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("avatar_bg_bind"),
            layout: &self.bg_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.bg_params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&self.sampler) },
            ],
        }));
        self.bg_dims = (w, h);
    }

    fn ensure_bg_pipe(&mut self, device: &Device, format: TextureFormat) {
        if let Some((_, f)) = &self.bg_pipe {
            if *f == format {
                return;
            }
        }
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("avatar_bg"),
            source: ShaderSource::Wgsl(BG_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("avatar_bg_pipe_layout"),
            bind_group_layouts: &[Some(&self.bg_layout)],
            ..Default::default()
        });
        let pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("avatar_bg_pipe"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_bg"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_bg"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // z=1(원면)에 그림 — 아바타(depth LessEqual)가 자연히 덮는다
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: MSAA_SAMPLES, ..Default::default() },
            multiview_mask: None,
            cache: None,
        });
        self.bg_pipe = Some((pipe, format));
    }

    // ── 모델 로드 ──

    pub fn preload_vrm(&mut self, ctx: &GpuContext, name: &str, bytes: &[u8]) -> Result<()> {
        if self.models.contains_key(name) {
            return Ok(());
        }
        let model = parse_vrm(bytes)?;
        let loaded = self.upload(ctx, &model)?;
        self.models.insert(name.to_string(), loaded);
        Ok(())
    }

    pub fn set_active(&mut self, name: &str) {
        if self.active != name {
            self.active = name.to_string();
            if let Some(m) = self.models.get_mut(name) {
                m.smooth_quat = None;
            }
        }
    }

    pub fn active_loaded(&self) -> bool {
        self.models.contains_key(&self.active)
    }

    fn upload(&self, ctx: &GpuContext, model: &VrmModel) -> Result<LoadedAvatar> {
        let device = &ctx.device;
        if model.skin_joints.len() > MAX_JOINTS {
            return Err(format!(
                "조인트 {}개 — MAX_JOINTS {MAX_JOINTS} 초과",
                model.skin_joints.len()
            ));
        }

        // 텍스처 업로드 (sRGB — 샘플에서 리니어로)
        let mut tex_views: Vec<TextureView> = Vec::with_capacity(model.images.len());
        for (i, img) in model.images.iter().enumerate() {
            let tex = device.create_texture(&TextureDescriptor {
                label: Some(&format!("avatar_tex_{i}")),
                size: Extent3d {
                    width: img.width,
                    height: img.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &img.rgba,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(img.width * 4),
                    rows_per_image: None,
                },
                Extent3d {
                    width: img.width,
                    height: img.height,
                    depth_or_array_layers: 1,
                },
            );
            tex_views.push(tex.create_view(&TextureViewDescriptor::default()));
        }

        let blink_binds = model.blink_binds();
        let aa_binds = model.aa_binds();

        let mut prims = Vec::with_capacity(model.primitives.len());
        for (pi, p) in model.primitives.iter().enumerate() {
            let n = p.positions.len();
            let mut verts = Vec::with_capacity(n);
            for i in 0..n {
                verts.push(AvatarVertex {
                    pos: p.positions[i],
                    normal: p.normals[i],
                    uv: p.uvs[i],
                    joints: p.joints[i],
                    weights: p.weights[i],
                });
            }
            let vbuf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("avatar_vbuf_{pi}")),
                contents: bytemuck::cast_slice(&verts),
                usage: BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("avatar_ibuf_{pi}")),
                contents: bytemuck::cast_slice(&p.indices),
                usage: BufferUsages::INDEX,
            });

            // 모프 스토리지: 정점당 vec4 4개 [blink_dpos, blink_dnrm, aa_dpos, aa_dnrm]
            // — 바인드 가중 합으로 구운 결합 델타 (그룹 가중은 로드 시 상수)
            let mut morph = vec![[0f32; 4]; n * 4];
            let mut accumulate = |binds: &[(usize, f32)], slot_pos: usize, slot_nrm: usize| {
                for &(ti, w) in binds {
                    if let Some(t) = p.targets.get(ti) {
                        for i in 0..n {
                            let d = t.d_pos[i];
                            morph[i * 4 + slot_pos][0] += d[0] * w;
                            morph[i * 4 + slot_pos][1] += d[1] * w;
                            morph[i * 4 + slot_pos][2] += d[2] * w;
                            if let Some(dn) = t.d_normal.get(i) {
                                morph[i * 4 + slot_nrm][0] += dn[0] * w;
                                morph[i * 4 + slot_nrm][1] += dn[1] * w;
                                morph[i * 4 + slot_nrm][2] += dn[2] * w;
                            }
                        }
                    }
                }
            };
            accumulate(&blink_binds, 0, 1);
            accumulate(&aa_binds, 2, 3);
            let morph_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("avatar_morph_{pi}")),
                contents: bytemuck::cast_slice(&morph),
                usage: BufferUsages::STORAGE,
            });

            let mat = model
                .materials
                .get(p.material)
                .or_else(|| model.materials.first())
                .ok_or("머티리얼 없음")?;
            let mat_uniform = MaterialUniform {
                base_color: mat.base_color,
                shade_color: mat.shade_color,
                outline_color: mat.outline_color,
                params: [
                    mat.outline_width * 0.01, // cm → m
                    mat.cutoff,
                    mat.shade_shift,
                    mat.shade_toony,
                ],
                mode: [mat.blend_mode.min(1) as f32, 0.0, 0.0, 0.0],
            };
            let mat_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("avatar_mat_{pi}")),
                contents: bytemuck::bytes_of(&mat_uniform),
                usage: BufferUsages::UNIFORM,
            });
            let main_view = mat
                .main_tex
                .and_then(|i| tex_views.get(i))
                .unwrap_or(&self.white_view);
            let shade_view = mat
                .shade_tex
                .and_then(|i| tex_views.get(i))
                .or_else(|| mat.main_tex.and_then(|i| tex_views.get(i)))
                .unwrap_or(&self.white_view);
            let mat_bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("avatar_mat_bind_{pi}")),
                layout: &self.mat_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: mat_buf.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(main_view) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&self.sampler) },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(shade_view),
                    },
                    BindGroupEntry { binding: 4, resource: morph_buf.as_entire_binding() },
                ],
            });

            prims.push(GpuPrim {
                vbuf,
                ibuf,
                index_count: p.indices.len() as u32,
                mat_bind,
                outline: mat.outline_width > 0.0,
                cutout: mat.blend_mode == 1,
            });
        }

        // ── 프레이밍: rest 포즈에서 상반신(waist-up) 고정 카메라 ──
        let mut rig = Rig::new(model);
        rig.update_globals();
        let head_y = rig.node_world_pos(model.humanoid["head"])[1];
        let hips_y = model
            .humanoid
            .get("hips")
            .map(|&n| rig.node_world_pos(n)[1])
            .unwrap_or(head_y * 0.55);
        // 정수리는 head 본 위로 더 있다 — head-hips 구간 비율로 머리 높이 가산
        let top_y = head_y + (head_y - hips_y) * HEAD_MARGIN;
        let cut_y = hips_y + (head_y - hips_y) * 0.05;
        let span = (top_y - cut_y).max(0.05);
        let dist = span / FRAME_FILL / (2.0 * (VFOV_DEG.to_radians() / 2.0).tan());
        // 머리(얼굴 중심)가 화면 세로 중앙 — 몸통은 아래로 내려가고 위는 배경
        let center_y = head_y + (head_y - hips_y) * HEAD_CENTER_OFFSET;
        let yaw_q = [0.0, (ROOT_YAW / 2.0).sin(), 0.0, (ROOT_YAW / 2.0).cos()];
        let base_model = mat_mul(
            &mat_compose([0.0, -center_y, -dist], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            &mat_compose([0.0, 0.0, 0.0], yaw_q, [1.0; 3]),
        );

        Ok(LoadedAvatar {
            prims,
            rig,
            base_model,
            cam_dist: dist,
            center_y,
            blink: BlinkCtl::default(),
            mouth: MouthCtl::default(),
            smooth_quat: None,
        })
    }

    /// 화면 픽셀 → 모델 공간 (base_model 역변환 — 손목 IK 타깃용).
    /// 깊이는 몸 평면보다 약간 앞(카메라 쪽) 고정 — 2D 관측이라 z 는 추정 불가.
    fn px_to_model(px: [f32; 2], fw: u32, fh: u32, dist: f32, center_y: f32) -> [f32; 3] {
        const FORWARD: f32 = 0.12;
        let tanv = (VFOV_DEG.to_radians() / 2.0).tan();
        let aspect = fw as f32 / fh.max(1) as f32;
        let ndc_x = px[0] / fw as f32 * 2.0 - 1.0;
        let ndc_y = 1.0 - px[1] / fh as f32 * 2.0;
        // 월드(카메라계) → base_model 역변환: T(0,-cy,-d)·RotY(yaw) 의 역
        let w = [ndc_x * tanv * aspect * dist, ndc_y * tanv * dist, -dist + FORWARD];
        let (c, s) = ((-ROOT_YAW).cos(), (-ROOT_YAW).sin());
        let p = [w[0], w[1] + center_y, w[2] + dist]; // T⁻¹
        [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]] // RotY(-yaw)
    }

    // ── 파이프라인/타깃 ──

    fn ensure_pipelines(&mut self, device: &Device, format: TextureFormat) {
        if let Some((_, _, f)) = &self.pipes {
            if *f == format {
                return;
            }
        }
        let shader = self.shader.get_or_insert_with(|| {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("avatar_mtoon"),
                source: ShaderSource::Wgsl(include_str!("mtoon.wgsl").into()),
            })
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("avatar_pipe_layout"),
            bind_group_layouts: &[Some(&self.cam_layout), Some(&self.mat_layout)],
            ..Default::default()
        });
        let vbuf_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<AvatarVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute { format: VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                VertexAttribute { format: VertexFormat::Uint16x4, offset: 32, shader_location: 3 },
                VertexAttribute { format: VertexFormat::Float32x4, offset: 40, shader_location: 4 },
            ],
        };
        let make = |label: &str, vs: &str, fs: &str, cull: Face| -> RenderPipeline {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState {
                    module: shader,
                    entry_point: Some(vs),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(vbuf_layout.clone())],
                },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some(fs),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: Some(cull),
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::LessEqual),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState { count: MSAA_SAMPLES, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let main = make("avatar_main", "vs_main", "fs_main", Face::Back);
        let outline = make("avatar_outline", "vs_outline", "fs_outline", Face::Front);
        self.pipes = Some((main, outline, format));
    }

    fn ensure_targets(&mut self, device: &Device, w: u32, h: u32, format: TextureFormat) {
        let format_changed = self
            .msaa
            .as_ref()
            .map(|t| t.format() != format)
            .unwrap_or(true);
        if self.size == (w, h) && !format_changed {
            return;
        }
        self.size = (w, h);
        let make = |label: &str, fmt: TextureFormat| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLES,
                dimension: TextureDimension::D2,
                format: fmt,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        self.msaa = Some(make("avatar_msaa", format));
        self.depth = Some(make("avatar_depth", DEPTH_FORMAT));
    }

    // ── 프레임 ──

    /// 아바타 한 프레임: 배경 clear + 스킨 메시 렌더 → target 리졸브.
    /// landmarks 는 얼굴 추적 프레임에만 Some — 없으면 마지막 포즈 + auto-blink.
    /// hands 는 손 추적 틱의 21pt 픽셀 랜드마크 (최대 2) — 팔 IK/손가락 컬 입력.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        ctx: &GpuContext,
        target: &TextureView,
        format: TextureFormat,
        fw: u32,
        fh: u32,
        landmarks: Option<&[[f32; 3]]>,
        hands: &[[[f32; 2]; 21]],
        bg: AvatarBg,
        t_ms: f64,
    ) {
        if !self.models.contains_key(&self.active) {
            return;
        }
        self.ensure_pipelines(&ctx.device, format);
        self.ensure_targets(&ctx.device, fw, fh, format);
        let use_bg_image = matches!(bg, AvatarBg::Image) && self.bg_bind.is_some();
        if use_bg_image {
            self.ensure_bg_pipe(&ctx.device, format);
            // cover-fit: 프레임을 가득 채우고 넘치는 축을 크롭
            let (bw, bh) = self.bg_dims;
            let fa = fw as f32 / fh.max(1) as f32;
            let ba = bw as f32 / bh.max(1) as f32;
            let (sx, sy) = if ba > fa { (fa / ba, 1.0) } else { (1.0, ba / fa) };
            let params = [sx, sy, (1.0 - sx) / 2.0, (1.0 - sy) / 2.0];
            ctx.queue.write_buffer(&self.bg_params_buf, 0, bytemuck::cast_slice(&params));
        }

        // 포즈/blink/입 계산 (가변 차용 블록 — 이후는 불변 참조만)
        let active = self.active.clone();
        let (blink_w, mouth_w) = {
            let m = self.models.get_mut(&active).unwrap();
            let fit = landmarks.and_then(|lm| face_pose_from_landmarks(lm, fw, fh));
            if let Some((q, _pos)) = fit {
                let sq = match m.smooth_quat {
                    Some(prev) => quat_slerp(prev, q, HEAD_SLERP_ALPHA),
                    None => q,
                };
                m.smooth_quat = Some(sq);
            }
            m.rig.clear_pose();
            if let Some(q) = m.smooth_quat {
                m.rig.set_head_rotation(quat_mul(head_trim(), q), ROOT_YAW);
            }

            // ── 손 → 팔 IK 타깃 (화면 좌우로 아바타 팔 선택 — 거울 일관) ──
            // 화면 왼쪽 손 = (카메라를 보는) 아바타의 오른팔(1), 오른쪽 = 왼팔(0).
            let mut obs: [Option<([f32; 3], [f32; 5])>; 2] = [None, None];
            for lm in hands.iter() {
                let wrist = lm[0];
                let side = if wrist[0] < fw as f32 / 2.0 { 1 } else { 0 };
                let slot = if obs[side].is_none() { side } else { 1 - side };
                obs[slot] = Some((
                    Self::px_to_model(wrist, fw, fh, m.cam_dist, m.center_y),
                    crate::features::avatar::rig::finger_curls_from_px(lm),
                ));
            }
            for side in 0..2 {
                match obs[side] {
                    Some((t, c)) => m.rig.set_hand_obs(side, Some(t), Some(c)),
                    None => m.rig.set_hand_obs(side, None, None),
                }
            }
            m.rig.tick_arms();
            m.rig.update_globals();
            let blink_obs = landmarks.and_then(blink_from_landmarks);
            let mouth_obs = landmarks.and_then(mouth_from_landmarks);
            (m.blink.update(blink_obs, t_ms), m.mouth.update(mouth_obs))
        };

        // uniform 기록
        let m = self.models.get(&active).unwrap();
        let vrm_palette: Vec<[f32; 16]> = m.rig.joint_palette();
        let mut joints_data = [[0f32; 16]; MAX_JOINTS];
        for (i, p) in vrm_palette.iter().enumerate().take(MAX_JOINTS) {
            joints_data[i] = *p;
        }
        let aspect = fw as f32 / fh.max(1) as f32;
        let proj = mat_perspective(VFOV_DEG.to_radians(), aspect, 0.05, 100.0);
        let ld = {
            let v: [f32; 3] = [-0.3, 0.5, 1.0];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let cam = CameraUniform {
            view_proj: proj,
            model: m.base_model,
            light_dir: [ld[0], ld[1], ld[2], 1.0],
            light_color: [1.0, 0.98, 0.96, 0.55],
        };
        ctx.queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&cam));
        ctx.queue
            .write_buffer(&self.joints_buf, 0, bytemuck::cast_slice(&joints_data));
        ctx.queue.write_buffer(
            &self.morph_w_buf,
            0,
            // x=blink(슬롯 0/1), y=입 'a'(슬롯 2/3) — mtoon.wgsl morphed_pos/normal 규약
            bytemuck::cast_slice(&[blink_w, mouth_w, 0.0, 0.0]),
        );

        // 렌더패스
        let (pipe_main, pipe_outline, _) = self.pipes.as_ref().unwrap();
        let msaa_view = self
            .msaa
            .as_ref()
            .unwrap()
            .create_view(&TextureViewDescriptor::default());
        let depth_view = self
            .depth
            .as_ref()
            .unwrap()
            .create_view(&TextureViewDescriptor::default());
        let clear = match bg {
            AvatarBg::Color(c) => Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: 1.0 },
            _ => {
                let [br, bg_c, bb] = self.bg_srgb;
                Color { r: br, g: bg_c, b: bb, a: 1.0 }
            }
        };
        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("avatar_render") });
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("avatar_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &msaa_view,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: Operations {
                        load: LoadOp::Clear(clear),
                        store: StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            // 배경 이미지 (있으면) — depth 미기록 z=1 이라 아바타가 자연히 덮는다
            if use_bg_image {
                if let (Some((bg_pipe, _)), Some(bg_bind)) = (&self.bg_pipe, &self.bg_bind) {
                    pass.set_pipeline(bg_pipe);
                    pass.set_bind_group(0, bg_bind, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            pass.set_bind_group(0, &self.cam_bind, &[]);
            // 아웃라인 먼저 (인버티드 헐) — 본체가 위에 덮는다
            pass.set_pipeline(pipe_outline);
            for p in &m.prims {
                if !p.outline {
                    continue;
                }
                pass.set_bind_group(1, &p.mat_bind, &[]);
                pass.set_vertex_buffer(0, p.vbuf.slice(..));
                pass.set_index_buffer(p.ibuf.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..p.index_count, 0, 0..1);
            }
            pass.set_pipeline(pipe_main);
            for p in &m.prims {
                pass.set_bind_group(1, &p.mat_bind, &[]);
                pass.set_vertex_buffer(0, p.vbuf.slice(..));
                pass.set_index_buffer(p.ibuf.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..p.index_count, 0, 0..1);
            }
        }
        ctx.queue.submit([enc.finish()]);
    }

    /// GPU 자원 반납 (모델 바이트는 director 가 들고 있어 재로드 가능)
    pub fn reset(&mut self) {
        self.models.clear();
        self.msaa = None;
        self.depth = None;
        self.size = (0, 0);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mtoon_wgsl_parses() {
        // items3d 와 동일 정적 게이트: 파싱/검증 + MSL/SPIR-V 코드젠 + 바인딩 수 대조
        let src = include_str!("mtoon.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("mtoon.wgsl parse error");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("mtoon.wgsl validation error");

        let mut msl_opts = naga::back::msl::Options::default();
        msl_opts.lang_version = (2, 0);
        naga::back::msl::write_string(
            &module,
            &info,
            &msl_opts,
            &naga::back::msl::PipelineOptions::default(),
        )
        .expect("mtoon.wgsl → MSL (iOS)");
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .expect("mtoon.wgsl → SPIR-V (Android)");

        let mut per_group = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        for (_, g) in module.global_variables.iter() {
            if let Some(b) = g.binding.as_ref() {
                per_group.entry(b.group).or_default().push(b.binding);
            }
        }
        assert_eq!(
            per_group.get(&0).map(|v| v.len()),
            Some(super::CAM_BINDING_COUNT),
            "group(0) 바인딩 수 ≠ 레이아웃"
        );
        assert_eq!(
            per_group.get(&1).map(|v| v.len()),
            Some(super::MAT_BINDING_COUNT),
            "group(1) 바인딩 수 ≠ 레이아웃"
        );
    }

    #[test]
    fn vertex_layout_is_56_bytes() {
        assert_eq!(std::mem::size_of::<super::AvatarVertex>(), 56);
    }
}
