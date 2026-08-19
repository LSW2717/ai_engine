// items3d — 3D 아이템(모자/안경/수염) 레이어. 웹 face-3d.ts의 Rust판
// (vcxrust_ai vcx-segmentation face/items3d.rs 에서 이관 — 웹·모바일 렌더러 통일).
// 이관 차이: ①에셋은 파일 IO 대신 **호스트가 bytes 주입** (preload_glb — wasm은
// fetch, ffi는 fs) ②에러 스코프 pollster 제거 (wasm 비호환 — naga 테스트가 정적
// 게이트) ③RGB 프레임 광원 프로브 추가(웹 probeSceneLight 등가; YUV판은 모바일용
// 그대로) ④합성은 ItemsOverlay(파일 하단)가 서피스에 직접 알파-오버.
//
// 배치 원칙(웹과 동일): 아이템은 canonical face model 좌표(cm, y상/+z얼굴전방)에
// 고정 배치하고, 프레임마다 canonical→카메라공간 유사변환(R·s·t)을 루트에 꽂는다.
// 카메라는 MediaPipe metric space 가상 카메라 규약(원점, -z 시선, vfov 63°).
//
// 변환은 관측 랜드마크를 depth 로 역투영한 3D 점군에 canonical 점군을
// Horn 절대방향법(쿼터니언 파워이터레이션)으로 피팅한다. depth 는 동공간격
// (iris 미지원 — 눈꺼풀 중앙 159/386 근사)으로 추정.
// 지터 억제: 위치·스케일 EMA + 회전 slerp 저역통과 (움직임 크면 즉시 추종).
//
// 렌더: wgpu 렌더패스(MSAA4 + depth) → 프레임 해상도 RGBA 리졸브 텍스처.
// pack 셰이더가 src_orig 좌표로 알파-오버 합성한다(웹 drawImage 대응).
// GLB 는 외부 크레이트 없이 직접 파싱 (JSON 청크 serde_json + BIN 청크 accessor).

use std::collections::HashMap;
use std::sync::OnceLock;

use ai_gpu::wgpu::util::{BufferInitDescriptor, DeviceExt};
use ai_gpu::wgpu::*;
use ai_gpu::GpuContext;

use crate::error::TaskError;

type Result<T> = std::result::Result<T, String>;

// ── 카메라/캐노니컬 규약 (웹 face-3d.ts 와 동일 수치) ──
const METRIC_VFOV_DEG: f32 = 63.0;
const METRIC_NEAR: f32 = 1.0;
const METRIC_FAR: f32 = 10000.0;
const CANON_FACE_W: f32 = 15.33;
const CANON_PUPIL_DIST: f32 = 6.3;
const CANON_HAT_ANCHOR_Y: f32 = 8.9;
const CANON_HAT_ANCHOR_Z: f32 = -2.2;
const CANON_EYE_ANCHOR: [f32; 3] = [0.0, 2.6, 4.7];

/// 피팅 점군 15개 — [랜드마크 idx, canonical x, y, z(cm)]. 웹 FIT_PTS 그대로.
const FIT_PTS: [(usize, f32, f32, f32); 15] = [
    (234, -7.66, 0.67, -2.44),
    (454, 7.66, 0.67, -2.44),
    (10, 0.0, 8.26, 4.48),
    (152, 0.0, -9.4, 4.26),
    (1, 0.0, -1.13, 7.48),
    (33, -4.45, 2.66, 3.17),
    (263, 4.45, 2.66, 3.17),
    (61, -2.46, -4.34, 4.28),
    (291, 2.46, -4.34, 4.28),
    (199, 0.0, -7.94, 5.18),
    (168, 0.0, 3.27, 5.24),
    (93, -7.54, -1.05, -2.43),
    (323, 7.54, -1.05, -2.43),
    (8, 0.0, 4.02, 5.28),
    (4, 0.0, -0.46, 7.59),
];

/// GLB 로드 후 곱하는 보정값 — 웹 GLB_HATS/GLB_EYEWEARS 와 동일.
/// (scale, pitch rad, dy, dz, sx, sy)
fn item_adjust(kind: &str) -> (f32, f32, f32, f32, f32, f32) {
    match kind {
        "hat1" => (1.12, 0.2, -0.18, 0.0, 1.0, 1.0),
        // 레더햇 — dy 부양 금지 원칙. 뒤로 젖힘(pitch -0.15) + 후방(dz -0.12)만
        // (웹 GLB_HATS 와 동일 값).
        "hat3" => (1.0, -0.15, -0.05, -0.2, 1.0, 1.0),
        // 다운로드 에셋 실착 보정 — 웹 GLB_HATS/GLB_EYEWEARS 와 동일 값.
        "hat_bicycle" => (1.08, -std::f32::consts::FRAC_PI_2, -0.32, 0.45, 1.0, 1.0),
        "hat_christmas" => (1.0, std::f32::consts::FRAC_PI_6, 0.05, -0.29, 1.0, 1.0),
        "glasses_heart" => (1.0, 0.0, 0.0, 0.14, 1.0, 1.0),
        // 풀 비어드 — canonical cm 직접 배치 GLB, 웹 GLB_BEARDS 와 동일 (dy -3cm)
        "beard_full" => (1.0, 0.0, -3.0, 0.0, 1.0, 1.0),
        _ => (1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    }
}

const HAT_KINDS: [&str; 6] = [
    "hat1",
    "hat3",
    "hat5",
    "hat_bicycle",
    "hat_christmas",
    "hat_cat_ears",
];
const EYEWEAR_KINDS: [&str; 5] = [
    "glasses1",
    "glasses2",
    "glasses3",
    "glasses_heart",
    "glasses_yellow",
];
// 수염류 GLB 는 canonical 얼굴 cm 좌표(원점=얼굴 원점)에 직접 구워져 있다 — 그룹 앵커 없음.
const BEARD_KINDS: [&str; 3] = ["mustache1", "mustache2", "beard_full"];

// ═══════════════════ 미니 행렬/쿼터니언 ═══════════════════
// 열우선(column-major) 4x4 — wgpu/WGSL mat4x4 레이아웃과 일치.

type Mat4 = [f32; 16];
type Quat = [f32; 4]; // (x, y, z, w)

fn mat_identity() -> Mat4 {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + r] * b[c * 4 + k];
            }
            out[c * 4 + r] = s;
        }
    }
    out
}

/// three.js compose 와 동일: T · R · S.
fn mat_compose(pos: [f32; 3], q: Quat, scale: [f32; 3]) -> Mat4 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let (sx, sy, sz) = (scale[0], scale[1], scale[2]);
    [
        (1.0 - (yy + zz)) * sx,
        (xy + wz) * sx,
        (xz - wy) * sx,
        0.0,
        (xy - wz) * sy,
        (1.0 - (xx + zz)) * sy,
        (yz + wx) * sy,
        0.0,
        (xz + wy) * sz,
        (yz - wx) * sz,
        (1.0 - (xx + yy)) * sz,
        0.0,
        pos[0],
        pos[1],
        pos[2],
        1.0,
    ]
}

/// wgpu NDC(z 0..1) 원근 투영 — 원점에서 -z 시선.
fn mat_perspective(vfov_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (vfov_rad / 2.0).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

fn quat_normalize(q: Quat) -> Quat {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len <= 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

/// three.js Quaternion.slerp 와 동일 동작(단축 경로 선택 포함).
fn quat_slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let mut cos_half = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b = b;
    if cos_half < 0.0 {
        cos_half = -cos_half;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    if cos_half >= 0.9995 {
        return quat_normalize([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]);
    }
    let half = cos_half.clamp(-1.0, 1.0).acos();
    let sin_half = half.sin();
    let ra = ((1.0 - t) * half).sin() / sin_half;
    let rb = (t * half).sin() / sin_half;
    [
        a[0] * ra + b[0] * rb,
        a[1] * ra + b[1] * rb,
        a[2] * ra + b[2] * rb,
        a[3] * ra + b[3] * rb,
    ]
}

fn quat_rotate(q: Quat, v: [f32; 3]) -> [f32; 3] {
    // v' = v + 2·q_xyz×(q_xyz×v + w·v)
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    let (vx, vy, vz) = (v[0], v[1], v[2]);
    let tx = 2.0 * (qy * vz - qz * vy);
    let ty = 2.0 * (qz * vx - qx * vz);
    let tz = 2.0 * (qx * vy - qy * vx);
    [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]
}

fn mat_rot_x(rad: f32) -> Mat4 {
    let (s, c) = rad.sin_cos();
    let mut m = mat_identity();
    m[5] = c;
    m[6] = s;
    m[9] = -s;
    m[10] = c;
    m
}

fn mat_translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = mat_identity();
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

fn mat_scale(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = mat_identity();
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m
}

// ═══════════════════ Horn 절대방향 피팅 ═══════════════════

struct FitResult {
    quat: Quat,
    pos: [f32; 3],
    scale: f32,
}

/// canonical(FIT_PTS) → obs 유사변환 피팅 (Horn 1987 쿼터니언법, 웹 fitPose 이식).
fn fit_pose(obs: &[[f32; 3]; 15]) -> FitResult {
    const N: usize = 15;
    // canonical centroid
    let mut cent_src = [0.0f32; 3];
    for f in &FIT_PTS {
        cent_src[0] += f.1;
        cent_src[1] += f.2;
        cent_src[2] += f.3;
    }
    for v in &mut cent_src {
        *v /= N as f32;
    }
    // obs centroid
    let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
    for o in obs {
        cx += o[0];
        cy += o[1];
        cz += o[2];
    }
    cx /= N as f32;
    cy /= N as f32;
    cz /= N as f32;

    // 교차공분산 + src 분산
    let (mut sxx, mut sxy, mut sxz) = (0.0f32, 0.0, 0.0);
    let (mut syx, mut syy, mut syz) = (0.0f32, 0.0, 0.0);
    let (mut szx, mut szy, mut szz) = (0.0f32, 0.0, 0.0);
    let mut src_var = 0.0f32;
    for (i, f) in FIT_PTS.iter().enumerate() {
        let ax = f.1 - cent_src[0];
        let ay = f.2 - cent_src[1];
        let az = f.3 - cent_src[2];
        let bx = obs[i][0] - cx;
        let by = obs[i][1] - cy;
        let bz = obs[i][2] - cz;
        sxx += ax * bx;
        sxy += ax * by;
        sxz += ax * bz;
        syx += ay * bx;
        syy += ay * by;
        syz += ay * bz;
        szx += az * bx;
        szy += az * by;
        szz += az * bz;
        src_var += ax * ax + ay * ay + az * az;
    }

    // Horn N 행렬 (대칭 4x4) — 최대 고유벡터 = 회전 쿼터니언 (w,x,y,z)
    let n00 = sxx + syy + szz;
    let n01 = syz - szy;
    let n02 = szx - sxz;
    let n03 = sxy - syx;
    let n11 = sxx - syy - szz;
    let n12 = sxy + syx;
    let n13 = szx + sxz;
    let n22 = -sxx + syy - szz;
    let n23 = syz + szy;
    let n33 = -sxx - syy + szz;

    // 파워 이터레이션 (shift 로 최대 고유값 지배 보장)
    let shift = n00.abs() + n11.abs() + n22.abs() + n33.abs() + 1.0;
    let mut q = [1.0f32, 0.01, 0.01, 0.01]; // (w,x,y,z)
    for _ in 0..24 {
        let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
        let t = [
            (n00 + shift) * qw + n01 * qx + n02 * qy + n03 * qz,
            n01 * qw + (n11 + shift) * qx + n12 * qy + n13 * qz,
            n02 * qw + n12 * qx + (n22 + shift) * qy + n23 * qz,
            n03 * qw + n13 * qx + n23 * qy + (n33 + shift) * qz,
        ];
        let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + t[3] * t[3])
            .sqrt()
            .max(1e-20);
        q = [t[0] / len, t[1] / len, t[2] / len, t[3] / len];
    }
    let quat: Quat = [q[1], q[2], q[3], q[0]];

    // 스케일 = Σ dst_c·(R src_c) / Σ|src_c|²
    let mut dot = 0.0f32;
    for (i, f) in FIT_PTS.iter().enumerate() {
        let r = quat_rotate(
            quat,
            [f.1 - cent_src[0], f.2 - cent_src[1], f.3 - cent_src[2]],
        );
        dot += r[0] * (obs[i][0] - cx) + r[1] * (obs[i][1] - cy) + r[2] * (obs[i][2] - cz);
    }
    let s = dot / src_var.max(1e-6);

    // t = dst_centroid - s·R·src_centroid
    let rc = quat_rotate(quat, cent_src);
    FitResult {
        quat,
        pos: [cx - s * rc[0], cy - s * rc[1], cz - s * rc[2]],
        scale: s,
    }
}

// ═══════════════════ GLB 파서 (외부 크레이트 無) ═══════════════════

struct GlbMaterial {
    base_color: [f32; 4],
    tex: Option<image::RgbaImage>,
    /// glTF metallicRoughnessTexture — G=roughness, B=metallic (factor 와 곱해진다).
    /// 이게 없으면 factor 만 쓰는데, factor 키가 생략된 GLB 는 스펙상 1.0/1.0 이라
    /// 통째로 거친 금속이 된다(글래스1 렌즈가 밋밋했던 원인).
    mr_tex: Option<image::RgbaImage>,
    normal_tex: Option<image::RgbaImage>,
    normal_scale: f32,
    /// KHR_materials_transmission.transmissionFactor (기본 0)
    transmission: f32,
    /// glTF doubleSided (기본 false) — three 는 false 면 side=FrontSide 로 뒷면을
    /// 컬링하고, true 면 gl_FrontFacing 으로 노멀을 뒤집는다.
    double_sided: bool,
    blend: bool,
    metallic: f32,
    roughness: f32,
}

struct GlbPrimitive {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    material: usize,
    /// 노드 월드 변환 (씬 루트 기준으로 구움)
    world: Mat4,
}

struct GlbModel {
    materials: Vec<GlbMaterial>,
    primitives: Vec<GlbPrimitive>,
}

fn glb_node_local(node: &serde_json::Value) -> Mat4 {
    if let Some(m) = node.get("matrix").and_then(|v| v.as_array()) {
        let mut out = mat_identity();
        for (i, v) in m.iter().take(16).enumerate() {
            out[i] = v.as_f64().unwrap_or(0.0) as f32;
        }
        return out;
    }
    let get3 = |key: &str, def: [f32; 3]| -> [f32; 3] {
        node.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                let mut o = def;
                for (i, v) in a.iter().take(3).enumerate() {
                    o[i] = v.as_f64().unwrap_or(def[i] as f64) as f32;
                }
                o
            })
            .unwrap_or(def)
    };
    let t = get3("translation", [0.0; 3]);
    let s = get3("scale", [1.0; 3]);
    let r = node
        .get("rotation")
        .and_then(|v| v.as_array())
        .map(|a| {
            let mut q = [0.0f32, 0.0, 0.0, 1.0];
            for (i, v) in a.iter().take(4).enumerate() {
                q[i] = v.as_f64().unwrap_or(0.0) as f32;
            }
            q
        })
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    mat_compose(t, r, s)
}

struct GlbAccess<'a> {
    json: &'a serde_json::Value,
    bin: &'a [u8],
}

impl<'a> GlbAccess<'a> {
    /// accessor → (버퍼 슬라이스 시작, stride, count, componentType, 성분수)
    fn accessor(&self, idx: usize) -> Option<(usize, usize, usize, u64, usize)> {
        let acc = self.json.get("accessors")?.get(idx)?;
        let count = acc.get("count")?.as_u64()? as usize;
        let ctype = acc.get("componentType")?.as_u64()?;
        let ncomp = match acc.get("type")?.as_str()? {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => return None,
        };
        let csize = match ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return None,
        };
        let bv_idx = acc.get("bufferView")?.as_u64()? as usize;
        let bv = self.json.get("bufferViews")?.get(bv_idx)?;
        let bv_off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let acc_off = acc.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let stride = bv
            .get("byteStride")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(csize * ncomp);
        Some((bv_off + acc_off, stride, count, ctype, ncomp))
    }

    fn read_vec3(&self, idx: usize) -> Vec<[f32; 3]> {
        let Some((off, stride, count, ctype, ncomp)) = self.accessor(idx) else {
            return Vec::new();
        };
        if ctype != 5126 || ncomp < 3 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let p = off + i * stride;
            if p + 12 > self.bin.len() {
                break;
            }
            let f = |o: usize| f32::from_le_bytes(self.bin[p + o..p + o + 4].try_into().unwrap());
            out.push([f(0), f(4), f(8)]);
        }
        out
    }

    fn read_vec2(&self, idx: usize) -> Vec<[f32; 2]> {
        let Some((off, stride, count, ctype, ncomp)) = self.accessor(idx) else {
            return Vec::new();
        };
        if ctype != 5126 || ncomp < 2 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let p = off + i * stride;
            if p + 8 > self.bin.len() {
                break;
            }
            let f = |o: usize| f32::from_le_bytes(self.bin[p + o..p + o + 4].try_into().unwrap());
            out.push([f(0), f(4)]);
        }
        out
    }

    fn read_indices(&self, idx: usize) -> Vec<u32> {
        let Some((off, stride, count, ctype, _)) = self.accessor(idx) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let p = off + i * stride;
            let v = match ctype {
                5121 if p < self.bin.len() => self.bin[p] as u32,
                5123 if p + 2 <= self.bin.len() => {
                    u16::from_le_bytes(self.bin[p..p + 2].try_into().unwrap()) as u32
                }
                5125 if p + 4 <= self.bin.len() => {
                    u32::from_le_bytes(self.bin[p..p + 4].try_into().unwrap())
                }
                _ => break,
            };
            out.push(v);
        }
        out
    }

    fn image_bytes(&self, img_idx: usize) -> Option<&'a [u8]> {
        let img = self.json.get("images")?.get(img_idx)?;
        let bv_idx = img.get("bufferView")?.as_u64()? as usize;
        let bv = self.json.get("bufferViews")?.get(bv_idx)?;
        let off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let len = bv.get("byteLength")?.as_u64()? as usize;
        self.bin.get(off..off + len)
    }
}

fn parse_glb(bytes: &[u8]) -> Result<GlbModel> {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err("glb: bad magic".to_string());
    }
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if bytes.len() < 20 + json_len {
        return Err("glb: truncated json chunk".to_string());
    }
    let json: serde_json::Value = serde_json::from_slice(&bytes[20..20 + json_len])
        .map_err(|e| format!("glb: json parse: {e}"))?;
    // BIN 청크 (JSON 청크는 4바이트 정렬)
    let bin_hdr = 20 + json_len + (4 - json_len % 4) % 4;
    let bin = if bin_hdr + 8 <= bytes.len() {
        let bin_len = u32::from_le_bytes(bytes[bin_hdr..bin_hdr + 4].try_into().unwrap()) as usize;
        bytes.get(bin_hdr + 8..bin_hdr + 8 + bin_len).unwrap_or(&[])
    } else {
        &[]
    };
    let acc = GlbAccess { json: &json, bin };

    // 재질
    let mut materials = Vec::new();
    if let Some(mats) = json.get("materials").and_then(|v| v.as_array()) {
        for m in mats {
            let pbr = m.get("pbrMetallicRoughness").cloned().unwrap_or_default();
            let base_color = pbr
                .get("baseColorFactor")
                .and_then(|v| v.as_array())
                .map(|a| {
                    let mut o = [1.0f32; 4];
                    for (i, v) in a.iter().take(4).enumerate() {
                        o[i] = v.as_f64().unwrap_or(1.0) as f32;
                    }
                    o
                })
                .unwrap_or([1.0; 4]);
            let load_tex = |slot: Option<&serde_json::Value>| -> Option<image::RgbaImage> {
                let tex_idx = slot?.get("index")?.as_u64()?;
                let src = json
                    .get("textures")?
                    .get(tex_idx as usize)?
                    .get("source")?
                    .as_u64()? as usize;
                let bytes = acc.image_bytes(src)?;
                image::load_from_memory(bytes).ok().map(|i| i.to_rgba8())
            };
            let tex = load_tex(pbr.get("baseColorTexture"));
            let mr_tex = load_tex(pbr.get("metallicRoughnessTexture"));
            let normal_tex = load_tex(m.get("normalTexture"));
            let normal_scale = m
                .get("normalTexture")
                .and_then(|t| t.get("scale"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            // KHR_materials_transmission — three 는 씬을 별도 타깃에 렌더해 굴절
            // 샘플링하지만, face-3d 씬에는 아이템과 오클루더뿐이라 렌즈 뒤 배경이
            // 빈 투명이다. 그 경우 three transmission_fragment 는
            //   transmissionAlpha = mix(a, transmitted.a(=0), transmission)
            //   diffuseColor.a *= transmissionAlpha
            // 로 최종 알파를 0 으로 만든다 — 즉 웹에서 렌즈는 아예 안 보인다.
            // 확장을 무시하면 alphaMode=OPAQUE 인 렌즈가 불투명하게 눈을 덮는다.
            let transmission = m
                .get("extensions")
                .and_then(|e| e.get("KHR_materials_transmission"))
                .map(|t| {
                    t.get("transmissionFactor")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32
                })
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let double_sided = m
                .get("doubleSided")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            materials.push(GlbMaterial {
                base_color,
                tex,
                mr_tex,
                normal_tex,
                normal_scale,
                transmission,
                double_sided,
                blend: transmission > 0.0
                    || m.get("alphaMode").and_then(|v| v.as_str()) == Some("BLEND"),
                metallic: pbr
                    .get("metallicFactor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                roughness: pbr
                    .get("roughnessFactor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
            });
        }
    }
    if materials.is_empty() {
        materials.push(GlbMaterial {
            base_color: [0.8, 0.8, 0.8, 1.0],
            tex: None,
            mr_tex: None,
            normal_tex: None,
            normal_scale: 1.0,
            transmission: 0.0,
            double_sided: false,
            blend: false,
            metallic: 0.0,
            roughness: 0.8,
        });
    }

    // 노드 트리 순회 — 월드 변환 구워서 프리미티브 평탄화
    let mut primitives = Vec::new();
    let empty = Vec::new();
    let nodes = json
        .get("nodes")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let scene_idx = json.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let roots: Vec<usize> = json
        .get("scenes")
        .and_then(|v| v.get(scene_idx))
        .and_then(|s| s.get("nodes"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|u| u as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..nodes.len()).collect());

    fn walk(
        nodes: &[serde_json::Value],
        json: &serde_json::Value,
        acc: &GlbAccess,
        node_idx: usize,
        parent: Mat4,
        out: &mut Vec<GlbPrimitive>,
    ) {
        let Some(node) = nodes.get(node_idx) else {
            return;
        };
        let world = mat_mul(&parent, &glb_node_local(node));
        if let Some(mesh_idx) = node.get("mesh").and_then(|v| v.as_u64()) {
            if let Some(prims) = json
                .get("meshes")
                .and_then(|v| v.get(mesh_idx as usize))
                .and_then(|m| m.get("primitives"))
                .and_then(|v| v.as_array())
            {
                for prim in prims {
                    let attrs = prim.get("attributes").cloned().unwrap_or_default();
                    let pos_idx = attrs.get("POSITION").and_then(|v| v.as_u64());
                    let Some(pos_idx) = pos_idx else { continue };
                    let positions = acc.read_vec3(pos_idx as usize);
                    if positions.is_empty() {
                        continue;
                    }
                    let normals = attrs
                        .get("NORMAL")
                        .and_then(|v| v.as_u64())
                        .map(|i| acc.read_vec3(i as usize))
                        .unwrap_or_default();
                    let uvs = attrs
                        .get("TEXCOORD_0")
                        .and_then(|v| v.as_u64())
                        .map(|i| acc.read_vec2(i as usize))
                        .unwrap_or_default();
                    let indices = prim
                        .get("indices")
                        .and_then(|v| v.as_u64())
                        .map(|i| acc.read_indices(i as usize))
                        .unwrap_or_else(|| (0..positions.len() as u32).collect());
                    out.push(GlbPrimitive {
                        positions,
                        normals,
                        uvs,
                        indices,
                        material: prim.get("material").and_then(|v| v.as_u64()).unwrap_or(0)
                            as usize,
                        world,
                    });
                }
            }
        }
        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            for c in children {
                if let Some(ci) = c.as_u64() {
                    walk(nodes, json, acc, ci as usize, world, out);
                }
            }
        }
    }
    for r in roots {
        walk(nodes, &json, &acc, r, mat_identity(), &mut primitives);
    }
    if primitives.is_empty() {
        return Err("glb: no primitives".to_string());
    }
    // NORMAL 어트리뷰트가 없는 프리미티브는 면 노멀을 굽는다.
    // three GLTFLoader 는 이 경우 material.flatShading 을 켜서 GPU 미분으로 면
    // 노멀을 만든다 — 정점 분리 후 면 노멀을 넣으면 같은 결과다. 폴백 상수 노멀을
    // 쓰면 메시 전체가 평면 하나로 셰이딩돼 고개 각도에 따라 색이 통째로 바뀐다.
    for prim in &mut primitives {
        if !prim.normals.is_empty() {
            continue;
        }
        bake_flat_normals(prim);
    }
    Ok(GlbModel {
        materials,
        primitives,
    })
}

/// sRGB u8 → linear f32 룩업 (밉 생성용).
static SRGB_TO_LINEAR: OnceLock<[f32; 256]> = OnceLock::new();

fn srgb_lut() -> &'static [f32; 256] {
    SRGB_TO_LINEAR.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// 밉 체인 생성 — 2x2 박스 다운샘플. 웹은 three 가 자동 생성한다 — 없으면
/// 축소 시 지글거린다.
/// `srgb=true`(baseColor) 는 linear 공간에서 평균내고 다시 sRGB 인코딩한다
/// (u8 sRGB 값을 그대로 평균내면 축소본이 어두워진다).
/// `srgb=false`(metallicRoughness / normal) 는 데이터가 이미 선형이라 그대로 평균.
fn build_mip_chain(img: &image::RgbaImage, srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let lut = srgb_lut();
    let mut levels: Vec<(u32, u32, Vec<u8>)> =
        vec![(img.width(), img.height(), img.as_raw().clone())];
    loop {
        let (w, h, ref src) = *levels.last().unwrap();
        if w <= 1 && h <= 1 {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0, 0.0, 0.0);
                let mut n = 0.0f32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let sx = (x * 2 + dx).min(w - 1);
                        let sy = (y * 2 + dy).min(h - 1);
                        let o = ((sy * w + sx) * 4) as usize;
                        if srgb {
                            r += lut[src[o] as usize];
                            g += lut[src[o + 1] as usize];
                            b += lut[src[o + 2] as usize];
                        } else {
                            r += src[o] as f32 / 255.0;
                            g += src[o + 1] as f32 / 255.0;
                            b += src[o + 2] as f32 / 255.0;
                        }
                        a += src[o + 3] as f32 / 255.0;
                        n += 1.0;
                    }
                }
                let o = ((y * nw + x) * 4) as usize;
                let enc = |v: f32| -> u8 {
                    if srgb {
                        linear_to_srgb_u8(v)
                    } else {
                        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                    }
                };
                dst[o] = enc(r / n);
                dst[o + 1] = enc(g / n);
                dst[o + 2] = enc(b / n);
                dst[o + 3] = ((a / n).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        levels.push((nw, nh, dst));
    }
    levels
}

/// 인덱스를 풀어 삼각형마다 정점을 분리하고 면 노멀을 채운다(three flatShading 대응).
fn bake_flat_normals(prim: &mut GlbPrimitive) {
    let src_pos = std::mem::take(&mut prim.positions);
    let src_uv = std::mem::take(&mut prim.uvs);
    let src_idx = std::mem::take(&mut prim.indices);
    let tri_count = src_idx.len() / 3;
    let mut pos = Vec::with_capacity(tri_count * 3);
    let mut nrm = Vec::with_capacity(tri_count * 3);
    let mut uv = Vec::with_capacity(tri_count * 3);
    for t in 0..tri_count {
        let i = [
            src_idx[t * 3] as usize,
            src_idx[t * 3 + 1] as usize,
            src_idx[t * 3 + 2] as usize,
        ];
        let Some(&p0) = src_pos.get(i[0]) else {
            continue;
        };
        let Some(&p1) = src_pos.get(i[1]) else {
            continue;
        };
        let Some(&p2) = src_pos.get(i[2]) else {
            continue;
        };
        // glTF 는 CCW 가 앞면 — cross(p1-p0, p2-p0) 가 바깥쪽 노멀.
        let a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let mut n = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-12 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        } else {
            // 퇴화 삼각형 — 셰이딩에 기여하지 않게 위쪽으로 둔다.
            n = [0.0, 1.0, 0.0];
        }
        for k in 0..3 {
            pos.push(src_pos[i[k]]);
            nrm.push(n);
            uv.push(src_uv.get(i[k]).copied().unwrap_or([0.0, 0.0]));
        }
    }
    prim.indices = (0..pos.len() as u32).collect();
    prim.positions = pos;
    prim.normals = nrm;
    prim.uvs = uv;
}

// ═══════════════════ GPU 메시/파이프라인 ═══════════════════

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    key_dir: [f32; 4],
    key_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshUniform {
    model: [f32; 16],
    base_color: [f32; 4],
    factors: [f32; 4],
    /// x=has_mr_tex, y=has_normal_tex, z=normal_scale, w=double_sided(0/1)
    factors2: [f32; 4],
    /// x=transmission, yzw=미사용
    factors3: [f32; 4],
}

struct GpuMesh {
    vbuf: Buffer,
    ibuf: Buffer,
    index_count: u32,
    uniform: Buffer,
    bind: BindGroup,
    blend: bool,
    /// 아이템 로컬(그룹 앵커·GLB 보정·노드 월드 다 구움) — 프레임마다 face 행렬만 곱함.
    local: Mat4,
    base_color: [f32; 4],
    factors_xyz: [f32; 3], // metallic, roughness, has_tex
    mode: f32,             // 0=메시, 1=그림자 스프라이트
    has_mr: bool,
    has_normal: bool,
    normal_scale: f32,
    double_sided: bool,
    transmission: f32,
}

struct ItemModel {
    meshes: Vec<GpuMesh>,
}

/// RoomEnvironment 프리필터 큐브맵 — scripts/bake_room_env.py 산출물.
/// RGBA16F, mip-major → face-major → row-major. 64^2 베이스 7밉.
/// include_bytes! 라 에셋 배포 배선이 필요 없다(바이너리에 박힘, 256KB).
const ENV_CUBE_BYTES: &[u8] = include_bytes!("env_room.bin");
const ENV_CUBE_SIZE: u32 = 64;
const ENV_CUBE_MIPS: u32 = 7;

/// items3d.wgsl 의 group(0)/group(1) 바인딩 개수. 레이아웃·바인드그룹·셰이더
/// 셋이 어긋나면 호스트 테스트는 통과하고 기기에서만 create_bind_group 이
/// BindingsNumMismatch 로 죽는다(액세서리 전멸). WGSL 반사로 대조한다.
const CAM_BINDING_COUNT: usize = 3;
const MESH_BINDING_COUNT: usize = 5;

const MSAA_SAMPLES: u32 = 4;
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24Plus;

pub struct Items3dRenderer {
    p_opaque: RenderPipeline,
    p_blend: RenderPipeline,
    p_occluder: RenderPipeline,
    p_shadow: RenderPipeline,
    mesh_layout: BindGroupLayout,
    cam_buf: Buffer,
    cam_bind: BindGroup,
    sampler: Sampler,
    white_tex_view: TextureView,

    msaa_color: Option<Texture>,
    depth: Option<Texture>,
    size: (u32, u32),

    occluder: GpuMesh,
    shadow: GpuMesh,
    hats: HashMap<String, Option<ItemModel>>,
    eyewears: HashMap<String, Option<ItemModel>>,
    beards: HashMap<String, Option<ItemModel>>,

    // 스무딩 상태
    smooth_quat: Quat,
    smooth_pos: [f32; 3],
    smooth_scale: f32,
    has_smooth: bool,

    // 씬 광원 매칭 (EMA)
    exposure: f32,
    tint: [f32; 3],
    probe_counter: u32,
}

impl Items3dRenderer {
    pub fn new(device: &Device, queue: &Queue) -> Result<Self> {
        // 에러 스코프 없음 — wasm은 pollster 불가·error scope 비활성(context.rs).
        // 셰이더 정합성은 naga 정적 테스트(items3d_wgsl_parses)가 게이트.
        let shader_src: &str = include_str!("items3d.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("items3d"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let cam_entries: [BindGroupLayoutEntry; CAM_BINDING_COUNT] = [
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
                    view_dimension: TextureViewDimension::Cube,
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
        ];
        let cam_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("items3d_cam"),
            entries: &cam_entries,
        });
        let mesh_entries: [BindGroupLayoutEntry; MESH_BINDING_COUNT] = [
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
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];
        let mesh_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("items3d_mesh"),
            entries: &mesh_entries,
        });
        let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("items3d_layout"),
            bind_group_layouts: &[Some(&cam_layout), Some(&mesh_layout)],
            ..Default::default()
        });

        let vbuf_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
        };
        // 프리멀티 알파-오버
        let premul_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        };
        let make_pipeline = |label: &str,
                             fs_entry: &str,
                             depth_write: bool,
                             color_writes: ColorWrites|
         -> Result<RenderPipeline> {
            let p = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipe_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(vbuf_layout.clone())],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(premul_blend),
                        write_mask: color_writes,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None, // three DoubleSide 관례(GLB 소품 얇은 면 다수)
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(CompareFunction::LessEqual),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });
            Ok(p)
        };

        let p_opaque = make_pipeline("items3d_opaque", "fs_main", true, ColorWrites::ALL)?;
        // 블렌드 패스는 깊이를 쓰지 않는다 — three GLTFLoader 가 alphaMode=BLEND 에
        // depthWrite=false 를 거는 것과 동일(mrdoob/three.js#17706). 얇은 양면 껍데기
        // (안경 렌즈)는 앞뒤 면이 거의 같은 깊이라, 깊이를 쓰면 삼각형 단위로 앞뒤가
        // 경합해 결정형 패턴이 생긴다.
        let p_blend = make_pipeline("items3d_blend", "fs_main", false, ColorWrites::ALL)?;
        let p_occluder = make_pipeline(
            "items3d_occluder",
            "fs_occluder",
            true,
            ColorWrites::empty(),
        )?;
        let p_shadow = make_pipeline("items3d_shadow", "fs_main", false, ColorWrites::ALL)?;

        let cam_buf = device.create_buffer(&BufferDescriptor {
            label: Some("items3d_cam_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // ── 환경 큐브맵 (RoomEnvironment 프리필터) ──
        let env_cube = device.create_texture(&TextureDescriptor {
            label: Some("items3d_env_cube"),
            size: Extent3d {
                width: ENV_CUBE_SIZE,
                height: ENV_CUBE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: ENV_CUBE_MIPS,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        {
            let mut off = 0usize;
            for mip in 0..ENV_CUBE_MIPS {
                let s = (ENV_CUBE_SIZE >> mip).max(1);
                let face_bytes = (s * s * 4 * 2) as usize; // RGBA16F
                for face in 0..6u32 {
                    let end = off + face_bytes;
                    let Some(slice) = ENV_CUBE_BYTES.get(off..end) else {
                        return Err(format!(
                            "items3d: env_room.bin 크기 부족 ({} bytes, mip{mip} face{face} 에서 소진)",
                            ENV_CUBE_BYTES.len()
                        ));
                    };
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: &env_cube,
                            mip_level: mip,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: face,
                            },
                            aspect: TextureAspect::All,
                        },
                        slice,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(s * 4 * 2),
                            rows_per_image: Some(s),
                        },
                        Extent3d {
                            width: s,
                            height: s,
                            depth_or_array_layers: 1,
                        },
                    );
                    off = end;
                }
            }
        }
        let env_cube_view = env_cube.create_view(&TextureViewDescriptor {
            label: Some("items3d_env_cube_view"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("items3d_env_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..Default::default()
        });

        let cam_bind_entries: [BindGroupEntry; CAM_BINDING_COUNT] = [
            BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&env_cube_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&env_sampler),
            },
        ];
        let cam_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("items3d_cam_bind"),
            layout: &cam_layout,
            entries: &cam_bind_entries,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("items3d_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            ..Default::default()
        });

        let white_tex = device.create_texture(&TextureDescriptor {
            label: Some("items3d_white"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white_tex_view = white_tex.create_view(&TextureViewDescriptor::default());

        // 오클루더(두상 타원체) — 깊이 전용. canonical cm: 위치 (0,1,-3.8), 반경 (8.6,11.5,8.3).
        let (sph_v, sph_i) = build_sphere(32, 24);
        let occ_local = mat_mul(&mat_translate(0.0, 1.0, -3.8), &mat_scale(8.6, 11.5, 8.3));
        let occluder = Self::make_mesh(
            device,
            &mesh_layout,
            &sampler,
            &white_tex_view,
            &sph_v,
            &sph_i,
            None,
            None,
            None,
            1.0,
            false,
            0.0,
            occ_local,
            [0.0; 4],
            [0.0, 1.0, 0.0],
            0.0,
            false,
        );

        // 챙 그림자 — 이마 앞 평면 스프라이트 (0,6,5.3), rot.x -0.35, scale (13,4.6,1).
        let (pl_v, pl_i) = build_plane();
        let shadow_local = mat_mul(
            &mat_mul(&mat_translate(0.0, 6.0, 5.3), &mat_rot_x(-0.35)),
            &mat_scale(13.0, 4.6, 1.0),
        );
        let shadow = Self::make_mesh(
            device,
            &mesh_layout,
            &sampler,
            &white_tex_view,
            &pl_v,
            &pl_i,
            None,
            None,
            None,
            1.0,
            false,
            0.0,
            shadow_local,
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            1.0,
            false,
        );

        Ok(Self {
            p_opaque,
            p_blend,
            p_occluder,
            p_shadow,
            mesh_layout,
            cam_buf,
            cam_bind,
            sampler,
            white_tex_view,
            msaa_color: None,
            depth: None,
            size: (0, 0),
            occluder,
            shadow,
            hats: HashMap::new(),
            eyewears: HashMap::new(),
            beards: HashMap::new(),
            smooth_quat: [0.0, 0.0, 0.0, 1.0],
            smooth_pos: [0.0; 3],
            smooth_scale: 0.0,
            has_smooth: false,
            exposure: 1.0,
            tint: [1.0; 3],
            probe_counter: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn make_mesh(
        device: &Device,
        mesh_layout: &BindGroupLayout,
        sampler: &Sampler,
        white_view: &TextureView,
        verts: &[Vertex],
        indices: &[u32],
        tex_view: Option<&TextureView>,
        mr_view: Option<&TextureView>,
        normal_view: Option<&TextureView>,
        normal_scale: f32,
        double_sided: bool,
        transmission: f32,
        local: Mat4,
        base_color: [f32; 4],
        factors_xyz: [f32; 3],
        mode: f32,
        blend: bool,
    ) -> GpuMesh {
        let vbuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("items3d_vbuf"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("items3d_ibuf"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("items3d_mesh_uniform"),
            size: std::mem::size_of::<MeshUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_bind_entries: [BindGroupEntry; MESH_BINDING_COUNT] = [
            BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(tex_view.unwrap_or(white_view)),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(mr_view.unwrap_or(white_view)),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(normal_view.unwrap_or(white_view)),
            },
        ];
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("items3d_mesh_bind"),
            layout: mesh_layout,
            entries: &mesh_bind_entries,
        });
        GpuMesh {
            vbuf,
            ibuf,
            index_count: indices.len() as u32,
            uniform,
            bind,
            blend,
            local,
            base_color,
            factors_xyz,
            mode,
            has_mr: mr_view.is_some(),
            has_normal: normal_view.is_some(),
            normal_scale,
            double_sided,
            transmission,
        }
    }

    /// GLB bytes → GPU 메시 묶음을 종류별 맵에 등록 (호스트가 bytes 조달 —
    /// wasm은 fetch, ffi는 fs. 모델 조달이 호스트 몫이라는 폴백 사다리 규약 그대로).
    pub fn preload_glb(
        &mut self,
        device: &Device,
        queue: &Queue,
        kind: &str,
        bytes: &[u8],
    ) -> Result<()> {
        if !HAT_KINDS.contains(&kind)
            && !EYEWEAR_KINDS.contains(&kind)
            && !BEARD_KINDS.contains(&kind)
        {
            return Err(format!("미지의 아이템 종류: {kind}"));
        }
        let model = self.load_item(device, queue, kind, bytes);
        let ok = model.is_some();
        let map = if HAT_KINDS.contains(&kind) {
            &mut self.hats
        } else if EYEWEAR_KINDS.contains(&kind) {
            &mut self.eyewears
        } else {
            &mut self.beards
        };
        map.insert(kind.to_string(), model);
        if ok { Ok(()) } else { Err(format!("{kind}: GLB 파싱/업로드 실패")) }
    }

    /// GLB bytes → GPU 메시 묶음. 아이템 로컬 변환(그룹 앵커 + GLB 보정)을 구워 넣는다.
    fn load_item(
        &self,
        device: &Device,
        queue: &Queue,
        kind: &str,
        bytes: &[u8],
    ) -> Option<ItemModel> {
        let model = match parse_glb(bytes) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("[items3d] {kind} GLB parse failed: {e}");
                return None;
            }
        };

        // 그룹 앵커(웹 hatGroup/eyeGroup) ∘ GLB 보정(scale/pitch/dy/dz)
        // 수염류는 canonical cm 좌표 직접 배치 규약 — 그룹 앵커 없이 항등.
        let (scale, pitch, dy, dz, sx, sy) = item_adjust(kind);
        let group = if HAT_KINDS.contains(&kind) {
            mat_mul(
                &mat_translate(0.0, CANON_HAT_ANCHOR_Y, CANON_HAT_ANCHOR_Z),
                &mat_scale(CANON_FACE_W, CANON_FACE_W, CANON_FACE_W),
            )
        } else if EYEWEAR_KINDS.contains(&kind) {
            let s = CANON_PUPIL_DIST / 0.52;
            mat_mul(
                &mat_translate(
                    CANON_EYE_ANCHOR[0],
                    CANON_EYE_ANCHOR[1],
                    CANON_EYE_ANCHOR[2],
                ),
                &mat_scale(s, s, s),
            )
        } else {
            mat_identity()
        };
        let adjust = mat_mul(
            &mat_mul(&mat_translate(0.0, dy, dz), &mat_rot_x(pitch)),
            &mat_scale(scale * sx, scale * sy, scale),
        );
        let base = mat_mul(&group, &adjust);

        let mut meshes = Vec::new();
        for prim in &model.primitives {
            let n = prim.positions.len();
            let mut verts = Vec::with_capacity(n);
            for i in 0..n {
                verts.push(Vertex {
                    pos: prim.positions[i],
                    normal: *prim.normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
                    uv: *prim.uvs.get(i).unwrap_or(&[0.0, 0.0]),
                });
            }
            let mat = model
                .materials
                .get(prim.material)
                .unwrap_or(&model.materials[0]);
            let upload = |img: &image::RgbaImage, srgb: bool, label: &str| -> TextureView {
                let mips = build_mip_chain(img, srgb);
                let tex = device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: img.width(),
                        height: img.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: mips.len() as u32,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    // baseColor 만 sRGB. metallicRoughness/normal 은 선형 데이터라
                    // sRGB 포맷으로 올리면 값이 통째로 틀어진다.
                    format: if srgb {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    },
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                for (level, (mw, mh, data)) in mips.iter().enumerate() {
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: &tex,
                            mip_level: level as u32,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        data,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(mw * 4),
                            rows_per_image: Some(*mh),
                        },
                        Extent3d {
                            width: *mw,
                            height: *mh,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                tex.create_view(&TextureViewDescriptor::default())
            };
            let tex_view = mat
                .tex
                .as_ref()
                .map(|i| upload(i, true, "items3d_base_tex"));
            let mr_view = mat
                .mr_tex
                .as_ref()
                .map(|i| upload(i, false, "items3d_mr_tex"));
            let normal_view = mat
                .normal_tex
                .as_ref()
                .map(|i| upload(i, false, "items3d_normal_tex"));
            let local = mat_mul(&base, &prim.world);
            meshes.push(Self::make_mesh(
                device,
                &self.mesh_layout,
                &self.sampler,
                &self.white_tex_view,
                &verts,
                &prim.indices,
                tex_view.as_ref(),
                mr_view.as_ref(),
                normal_view.as_ref(),
                mat.normal_scale,
                mat.double_sided,
                mat.transmission,
                local,
                mat.base_color,
                [
                    mat.metallic,
                    mat.roughness,
                    if mat.tex.is_some() { 1.0 } else { 0.0 },
                ],
                0.0,
                mat.blend,
            ));
        }
        log::info!("[items3d] {kind} loaded ({} meshes)", meshes.len());
        Some(ItemModel { meshes })
    }

    fn ensure_targets(&mut self, device: &Device, w: u32, h: u32) {
        if self.size == (w, h) && self.msaa_color.is_some() {
            return;
        }
        let mk = |fmt: TextureFormat, label: &str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLES,
                dimension: TextureDimension::D2,
                format: fmt,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        self.msaa_color = Some(mk(COLOR_FORMAT, "items3d_msaa_color"));
        self.depth = Some(mk(DEPTH_FORMAT, "items3d_depth"));
        self.size = (w, h);
    }

    /// 씬 광원 매칭 — 프레임 평균 밝기/색조 (웹 probeSceneLight 의 YUV 버전).
    /// 8프레임마다 호출자가 부담 없이 불러도 되게 내부 스로틀.
    pub fn probe_scene_light(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        stride_y: usize,
        stride_u: usize,
        stride_v: usize,
        width: usize,
        height: usize,
    ) {
        self.probe_counter = self.probe_counter.wrapping_add(1);
        if self.probe_counter % 8 != 1 {
            return;
        }
        let step = (width.max(height) / 16).max(1);
        let (mut sr, mut sg, mut sb, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
        let mut yy = step / 2;
        while yy < height {
            let mut xx = step / 2;
            while xx < width {
                let yv = y.get(yy * stride_y + xx).copied().unwrap_or(128) as f32;
                let uv = u.get((yy / 2) * stride_u + xx / 2).copied().unwrap_or(128) as f32 - 128.0;
                let vv = v.get((yy / 2) * stride_v + xx / 2).copied().unwrap_or(128) as f32 - 128.0;
                sr += (yv + 1.402 * vv).clamp(0.0, 255.0);
                sg += (yv - 0.344136 * uv - 0.714136 * vv).clamp(0.0, 255.0);
                sb += (yv + 1.772 * uv).clamp(0.0, 255.0);
                n += 1;
                xx += step;
            }
            yy += step;
        }
        if n == 0 {
            return;
        }
        self.apply_probe(sr / (n as f32 * 255.0), sg / (n as f32 * 255.0), sb / (n as f32 * 255.0));
    }

    /// 씬 광원 매칭 — u8 RGB 인터리브 프레임 버전 (웹 probeSceneLight 등가).
    /// YUV판과 같은 8프레임 스로틀·EMA 를 공유한다.
    pub fn probe_scene_light_rgb(&mut self, rgb: &[u8], width: usize, height: usize) {
        self.probe_counter = self.probe_counter.wrapping_add(1);
        if self.probe_counter % 8 != 1 {
            return;
        }
        self.probe_scene_light_rgb_now(rgb, width, height);
    }

    /// 스로틀 없는 판 — **호출자가 페이싱을 소유할 때** (studio: getImageData
    /// 리드백을 아끼려고 JS가 8틱 게이트를 하므로, 여기서 또 스로틀하면
    /// 8×8=64틱으로 뭉개진다).
    pub fn probe_scene_light_rgb_now(&mut self, rgb: &[u8], width: usize, height: usize) {
        let step = (width.max(height) / 16).max(1);
        let (mut sr, mut sg, mut sb, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
        let mut yy = step / 2;
        while yy < height {
            let mut xx = step / 2;
            while xx < width {
                let o = (yy * width + xx) * 3;
                if o + 2 < rgb.len() {
                    sr += rgb[o] as f32;
                    sg += rgb[o + 1] as f32;
                    sb += rgb[o + 2] as f32;
                    n += 1;
                }
                xx += step;
            }
            yy += step;
        }
        if n == 0 {
            return;
        }
        self.apply_probe(sr / (n as f32 * 255.0), sg / (n as f32 * 255.0), sb / (n as f32 * 255.0));
    }

    fn apply_probe(&mut self, r: f32, g: f32, b: f32) {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        // 기준 밝기 0.42 — 웹과 동일. EMA 0.15.
        let target_exp = (luma / 0.42).clamp(0.5, 1.4);
        self.exposure += (target_exp - self.exposure) * 0.15;
        let maxc = r.max(g).max(b).max(1e-4);
        for (t, c) in self.tint.iter_mut().zip([r / maxc, g / maxc, b / maxc]) {
            *t += (c - *t) * 0.15;
        }
    }

    /// 프레임 렌더 — 랜드마크로 포즈 피팅 후 아이템을 target(리졸브)에 그린다.
    /// 그릴 게 없으면 false (pack 은 enabled=0 으로 합성 스킵).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        target: &Texture,
        landmarks: &[[f32; 3]],
        vw: u32,
        vh: u32,
        hat: &str,
        eyewear: &str,
        beard: &str,
    ) -> bool {
        let show_hat = hat != "none" && HAT_KINDS.contains(&hat);
        let show_eye = eyewear != "none" && EYEWEAR_KINDS.contains(&eyewear);
        let show_beard = beard != "none" && BEARD_KINDS.contains(&beard);
        if (!show_hat && !show_eye && !show_beard) || landmarks.len() < 468 {
            return false;
        }
        // GLB는 preload_glb로 사전 주입 — 미주입 종류는 그냥 안 그린다.
        // 로드 성공 여부만 먼저 — 실제 참조는 self 가변 사용(스무딩/타깃 재생성) 뒤에 잡는다.
        let hat_ok = show_hat && self.hats.get(hat).map_or(false, |m| m.is_some());
        let eye_ok = show_eye && self.eyewears.get(eyewear).map_or(false, |m| m.is_some());
        let beard_ok = show_beard && self.beards.get(beard).map_or(false, |m| m.is_some());
        if !hat_ok && !eye_ok && !beard_ok {
            return false;
        }

        // ── 얼굴 depth (cm): 동공간격 추정 (159/386 = 눈꺼풀 중앙 근사) ──
        let f = vh as f32 / 2.0 / (METRIC_VFOV_DEG.to_radians() / 2.0).tan();
        let er = landmarks[159];
        let el = landmarks[386];
        let iod_px = ((el[0] - er[0]).powi(2) + (el[1] - er[1]).powi(2))
            .sqrt()
            .max(1.0);
        let depth = ((f * CANON_PUPIL_DIST) / iod_px).max(10.0);

        // ── 관측 랜드마크 역투영 (z 포함: z_cm ≈ depth + lm.z·depth/f) ──
        let (wf, hf) = (vw as f32, vh as f32);
        let mut obs = [[0.0f32; 3]; 15];
        for (i, fp) in FIT_PTS.iter().enumerate() {
            let p = landmarks[fp.0];
            let zi = depth + p[2] * (depth / f);
            obs[i] = [
                ((p[0] - wf / 2.0) * zi) / f,
                ((hf / 2.0 - p[1]) * zi) / f,
                -zi,
            ];
        }

        let fit = fit_pose(&obs);

        // ── 스무딩: 이동량 적응형 저역통과 ──
        if !self.has_smooth {
            self.smooth_quat = fit.quat;
            self.smooth_pos = fit.pos;
            self.smooth_scale = fit.scale;
            self.has_smooth = true;
        } else {
            let dx = self.smooth_pos[0] - fit.pos[0];
            let dy = self.smooth_pos[1] - fit.pos[1];
            let dz = self.smooth_pos[2] - fit.pos[2];
            let move_cm = (dx * dx + dy * dy + dz * dz).sqrt();
            // 0.05cm 미동 → α 0.35, 1cm+ 이동 → α 1 (웹과 동일)
            let alpha = (0.35 + move_cm * 0.65).min(1.0);
            for k in 0..3 {
                self.smooth_pos[k] += (fit.pos[k] - self.smooth_pos[k]) * alpha;
            }
            self.smooth_quat = quat_slerp(self.smooth_quat, fit.quat, alpha);
            self.smooth_scale += (fit.scale - self.smooth_scale) * alpha;
        }

        let face_mat = mat_compose(self.smooth_pos, self.smooth_quat, [self.smooth_scale; 3]);

        // ── 카메라/조명 uniform ──
        let proj = mat_perspective(
            METRIC_VFOV_DEG.to_radians(),
            wf / hf,
            METRIC_NEAR,
            METRIC_FAR,
        );
        // 키라이트 (-0.4, 1, 1) 정규화 — three key.position 방향. 틴트는 절반만.
        let kd = {
            let l = (0.4f32 * 0.4 + 1.0 + 1.0).sqrt();
            [-0.4 / l, 1.0 / l, 1.0 / l]
        };
        let key_color = [
            1.0 - (1.0 - self.tint[0]) * 0.5,
            1.0 - (1.0 - self.tint[1]) * 0.5,
            1.0 - (1.0 - self.tint[2]) * 0.5,
        ];
        let cam = CameraUniform {
            view_proj: proj,
            key_dir: [kd[0], kd[1], kd[2], self.exposure],
            key_color: [key_color[0], key_color[1], key_color[2], 0.0],
        };
        queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&cam));

        self.ensure_targets(device, vw, vh);

        // 여기서부터 self 불변 참조만 — 모델 참조를 안전하게 잡는다.
        let hat_model = if hat_ok {
            self.hats.get(hat).and_then(|m| m.as_ref())
        } else {
            None
        };
        let eye_model = if eye_ok {
            self.eyewears.get(eyewear).and_then(|m| m.as_ref())
        } else {
            None
        };
        let beard_model = if beard_ok {
            self.beards.get(beard).and_then(|m| m.as_ref())
        } else {
            None
        };

        // 그릴 메시 수집: 오클루더 → 불투명 → 블렌드 → (모자 시) 챙 그림자
        let mut draw_opaque: Vec<&GpuMesh> = vec![&self.occluder];
        let mut draw_blend: Vec<&GpuMesh> = Vec::new();
        for model in [hat_model, eye_model, beard_model].into_iter().flatten() {
            for m in &model.meshes {
                if m.blend {
                    draw_blend.push(m);
                } else {
                    draw_opaque.push(m);
                }
            }
        }
        let show_shadow = hat_model.is_some();

        // 메시 uniform 갱신 (face 행렬 ∘ 아이템 로컬)
        let update = |m: &GpuMesh| {
            let model_mat = mat_mul(&face_mat, &m.local);
            let u = MeshUniform {
                model: model_mat,
                base_color: m.base_color,
                factors: [m.factors_xyz[0], m.factors_xyz[1], m.factors_xyz[2], m.mode],
                factors2: [
                    if m.has_mr { 1.0 } else { 0.0 },
                    if m.has_normal { 1.0 } else { 0.0 },
                    m.normal_scale,
                    if m.double_sided { 1.0 } else { 0.0 },
                ],
                factors3: [m.transmission, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&m.uniform, 0, bytemuck::bytes_of(&u));
        };
        for m in draw_opaque.iter().chain(draw_blend.iter()) {
            update(m);
        }
        if show_shadow {
            update(&self.shadow);
        }

        let msaa_view = self
            .msaa_color
            .as_ref()
            .unwrap()
            .create_view(&TextureViewDescriptor::default());
        let depth_view = self
            .depth
            .as_ref()
            .unwrap()
            .create_view(&TextureViewDescriptor::default());
        let resolve_view = target.create_view(&TextureViewDescriptor::default());

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("items3d_render"),
        });
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("items3d_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&resolve_view),
                    ops: Operations {
                        load: LoadOp::Clear(Color::TRANSPARENT),
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.cam_bind, &[]);

            let draw = |pass: &mut RenderPass, m: &GpuMesh| {
                pass.set_bind_group(1, &m.bind, &[]);
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..1);
            };

            pass.set_pipeline(&self.p_occluder);
            draw(&mut pass, &self.occluder);

            pass.set_pipeline(&self.p_opaque);
            for m in draw_opaque.iter().skip(1) {
                draw(&mut pass, m);
            }
            if !draw_blend.is_empty() {
                pass.set_pipeline(&self.p_blend);
                for m in &draw_blend {
                    draw(&mut pass, m);
                }
            }
            if show_shadow {
                pass.set_pipeline(&self.p_shadow);
                draw(&mut pass, &self.shadow);
            }
        }
        queue.submit(Some(enc.finish()));
        true
    }

    /// 얼굴 상실/이펙트 off 시 스무딩 리셋 — 재등장 때 이전 포즈에서 미끄러져 오지 않게.
    pub fn reset_smoothing(&mut self) {
        self.has_smooth = false;
    }
}

// ═══════════════════ ItemsOverlay — 서피스 합성 래퍼 ═══════════════════

/// 렌더러 + 리졸브 텍스처 + 알파-오버 블릿. 호스트(studio/ffi)는 이것만 만진다:
/// preload_glb(bytes 주입) → set_items/set_pose(비전 틱) → draw(서피스에 오버레이).
/// ⚠ 프레이밍 크롭 중 좌표 보정은 미적용 (INTEGRATION.md §2 계약 — P3 이월분).
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> VOut {
    var out: VOut;
    let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4f(xy.x * 2.0 - 1.0, 1.0 - xy.y * 2.0, 0.0, 1.0);
    out.uv = xy;
    return out;
}

@fragment
fn fs_blit(in: VOut) -> @location(0) vec4f {
    // 렌더 패스가 프리멀티 알파를 만들었다 — 그대로 넘기고 블렌드가 오버한다
    return textureSample(t_src, s_src, in.uv);
}
"#;

pub struct ItemsOverlay {
    pub renderer: Items3dRenderer,
    resolve: Option<(Texture, TextureView)>,
    resolve_size: (u32, u32),
    blit: Option<(RenderPipeline, BindGroupLayout, TextureFormat)>,
    blit_bind: Option<BindGroup>,
    blit_sampler: Sampler,
    landmarks: Option<Vec<[f32; 3]>>,
    hat: String,
    eyewear: String,
    beard: String,
    /// 프레이밍 크롭 (scale, cx, cy — compose와 동일 규약, 1/0.5/0.5 = 크롭 없음).
    /// 화면이 crop 영역만 보여주므로 랜드마크를 같은 변환의 역으로 화면 좌표화.
    view_crop: (f32, f32, f32),
}

impl ItemsOverlay {
    pub fn new(ctx: &GpuContext) -> std::result::Result<Self, TaskError> {
        let renderer =
            Items3dRenderer::new(&ctx.device, &ctx.queue).map_err(TaskError::Other)?;
        let blit_sampler = ctx.device.create_sampler(&SamplerDescriptor {
            label: Some("items3d_blit_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        Ok(ItemsOverlay {
            renderer,
            resolve: None,
            resolve_size: (0, 0),
            blit: None,
            blit_bind: None,
            blit_sampler,
            landmarks: None,
            hat: "none".into(),
            eyewear: "none".into(),
            beard: "none".into(),
            view_crop: (1.0, 0.5, 0.5),
        })
    }

    /// 프레이밍 크롭 갱신 (VideoPipeline::framing_current()) — 매 프레임 호출 가능
    pub fn set_view_crop(&mut self, scale: f32, cx: f32, cy: f32) {
        self.view_crop = (scale, cx, cy);
    }

    pub fn preload_glb(
        &mut self,
        ctx: &GpuContext,
        kind: &str,
        bytes: &[u8],
    ) -> std::result::Result<(), TaskError> {
        self.renderer
            .preload_glb(&ctx.device, &ctx.queue, kind, bytes)
            .map_err(TaskError::Other)
    }

    pub fn set_items(&mut self, hat: &str, eyewear: &str, beard: &str) {
        self.hat = hat.to_string();
        self.eyewear = eyewear.to_string();
        self.beard = beard.to_string();
        if !self.active() {
            self.renderer.reset_smoothing();
        }
    }

    /// 최신 포즈 (정규화 [x,y,z]×478 — FaceTask points). None = 얼굴 소실
    /// (스무딩 리셋 — 재등장 때 이전 포즈에서 미끄러져 오지 않게, 웹 규약).
    pub fn set_pose(&mut self, pts: Option<Vec<[f32; 3]>>) {
        if pts.is_none() {
            self.renderer.reset_smoothing();
        }
        self.landmarks = pts;
    }

    pub fn active(&self) -> bool {
        self.hat != "none" || self.eyewear != "none" || self.beard != "none"
    }

    fn ensure_resolve(&mut self, ctx: &GpuContext, w: u32, h: u32) {
        if self.resolve_size == (w, h) && self.resolve.is_some() {
            return;
        }
        let tex = ctx.device.create_texture(&TextureDescriptor {
            label: Some("items3d_resolve"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        self.resolve = Some((tex, view));
        self.resolve_size = (w, h);
        self.blit_bind = None; // 리졸브 재생성 → 바인드그룹 무효
    }

    fn ensure_blit(&mut self, ctx: &GpuContext, format: TextureFormat) {
        if let Some((_, _, f)) = &self.blit {
            if *f == format {
                return;
            }
        }
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("items3d_blit"),
            source: ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let layout = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("items3d_blit_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("items3d_blit_pl"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = ctx.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("items3d_blit_pipe"),
            layout: Some(&pl),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_blit"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    // 프리멀티 알파-오버 — 렌더 패스 출력이 이미 프리멀티
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.blit = Some((pipeline, layout, format));
        self.blit_bind = None;
    }

    /// 아이템을 서피스 뷰에 오버레이. 그린 게 없으면 false (합성 생략).
    pub fn draw(
        &mut self,
        ctx: &GpuContext,
        target: &TextureView,
        format: TextureFormat,
        vw: u32,
        vh: u32,
    ) -> bool {
        if !self.active() {
            return false;
        }
        let Some(pts) = self.landmarks.clone() else {
            return false;
        };
        // 정규화 → 화면 px (z는 폭 스케일 — MediaPipe 규약).
        // 프레이밍 크롭 중이면 화면 = crop 영역이므로 역변환으로 보정
        // (compose: cuv = uv·s + (c − s/2) → uv = (p − (c − s/2))/s; z도 1/s —
        // 줌만큼 얼굴이 커지니 아이템 스케일이 따라간다).
        let (s, ccx, ccy) = self.view_crop;
        let inv = 1.0 / s.max(1e-4);
        let (ox, oy) = (ccx - s * 0.5, ccy - s * 0.5);
        let lm_px: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| {
                [
                    (p[0] - ox) * inv * vw as f32,
                    (p[1] - oy) * inv * vh as f32,
                    p[2] * inv * vw as f32,
                ]
            })
            .collect();
        self.ensure_resolve(ctx, vw, vh);
        let drew = {
            let (tex, _) = self.resolve.as_ref().unwrap();
            self.renderer.render(
                &ctx.device,
                &ctx.queue,
                tex,
                &lm_px,
                vw,
                vh,
                &self.hat,
                &self.eyewear,
                &self.beard,
            )
        };
        if !drew {
            return false;
        }
        self.ensure_blit(ctx, format);
        let (pipeline, layout, _) = self.blit.as_ref().unwrap();
        if self.blit_bind.is_none() {
            let (_, view) = self.resolve.as_ref().unwrap();
            self.blit_bind = Some(ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some("items3d_blit_bind"),
                layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(view) },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            }));
        }
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("items3d_blit_enc"),
        });
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("items3d_blit_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, self.blit_bind.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit(Some(enc.finish()));
        true
    }
}

/// UV 스피어 (반지름 1) — 오클루더용.
fn build_sphere(seg_w: u32, seg_h: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for iy in 0..=seg_h {
        let v = iy as f32 / seg_h as f32;
        let phi = v * std::f32::consts::PI;
        for ix in 0..=seg_w {
            let u = ix as f32 / seg_w as f32;
            let theta = u * std::f32::consts::TAU;
            let x = -phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            verts.push(Vertex {
                pos: [x, y, z],
                normal: [x, y, z],
                uv: [u, v],
            });
        }
    }
    let stride = seg_w + 1;
    for iy in 0..seg_h {
        for ix in 0..seg_w {
            let a = iy * stride + ix;
            let b = a + stride;
            idx.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    (verts, idx)
}

/// 1×1 평면 (xy, 중심 원점) — 챙 그림자 스프라이트용.
fn build_plane() -> (Vec<Vertex>, Vec<u32>) {
    let verts = vec![
        Vertex {
            pos: [-0.5, 0.5, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [0.5, 0.5, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
        },
        Vertex {
            pos: [-0.5, -0.5, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            pos: [0.5, -0.5, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
        },
    ];
    (verts, vec![0, 2, 1, 2, 3, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_identity_pose() {
        // canonical 점군을 그대로 관측으로 주면 항등 변환이 나와야 한다.
        let mut obs = [[0.0f32; 3]; 15];
        for (i, f) in FIT_PTS.iter().enumerate() {
            obs[i] = [f.1, f.2, f.3];
        }
        let fit = fit_pose(&obs);
        assert!((fit.scale - 1.0).abs() < 1e-3, "scale={}", fit.scale);
        assert!(fit.pos.iter().all(|v| v.abs() < 1e-2));
        assert!((fit.quat[3].abs() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn fit_translation_and_scale() {
        // 2배 스케일 + 평행이동.
        let mut obs = [[0.0f32; 3]; 15];
        for (i, f) in FIT_PTS.iter().enumerate() {
            obs[i] = [f.1 * 2.0 + 3.0, f.2 * 2.0 - 1.0, f.3 * 2.0 + 5.0];
        }
        let fit = fit_pose(&obs);
        assert!((fit.scale - 2.0).abs() < 1e-3);
        assert!((fit.pos[0] - 3.0).abs() < 1e-2);
        assert!((fit.pos[1] + 1.0).abs() < 1e-2);
        assert!((fit.pos[2] - 5.0).abs() < 1e-2);
    }

    #[test]
    fn fit_rotation_z90() {
        // z축 90° 회전: (x,y,z) → (-y,x,z).
        let mut obs = [[0.0f32; 3]; 15];
        for (i, f) in FIT_PTS.iter().enumerate() {
            obs[i] = [-f.2, f.1, f.3];
        }
        let fit = fit_pose(&obs);
        // 기대 쿼터니언: z축 90° = (0,0,sin45,cos45)
        let expect = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (fit.quat[2].abs() - expect).abs() < 1e-3,
            "quat={:?}",
            fit.quat
        );
        assert!((fit.quat[3].abs() - expect).abs() < 1e-3);
    }

    #[test]
    fn glb_parse_minimal() {
        // 삼각형 1개짜리 최소 GLB 를 손으로 구성.
        let positions: Vec<u8> = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            .iter()
            .flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes()))
            .collect();
        let indices: Vec<u8> = [0u16, 1, 2].iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut bin = positions.clone();
        bin.extend_from_slice(&indices);
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6}
            ],
            "buffers": [{"byteLength": bin.len()}]
        });
        let mut json_bytes = serde_json::to_vec(&json).unwrap();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);

        let model = parse_glb(&glb).expect("parse");
        assert_eq!(model.primitives.len(), 1);
        assert_eq!(model.primitives[0].positions.len(), 3);
        assert_eq!(model.primitives[0].indices, vec![0, 1, 2]);
    }

    #[test]
    fn glasses1_has_mr_and_normal_maps() {
        // glasses1 은 metallicFactor/roughnessFactor 키가 없고 값이 전부
        // metallicRoughnessTexture 에 들어있다. 이걸 놓치면 스펙 기본값 1.0/1.0 이
        // 적용돼 렌즈가 통째로 거친 금속이 된다(반사 소멸).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/demo/assets/glb/glasses1.glb"
        );
        let Ok(bytes) = std::fs::read(path) else {
            return; // 에셋 없는 환경은 스킵
        };
        let model = parse_glb(&bytes).expect("glasses1 parse");
        let mat = &model.materials[0];
        assert!(mat.mr_tex.is_some(), "metallicRoughnessTexture 미파싱");
        assert!(mat.normal_tex.is_some(), "normalTexture 미파싱");
        assert!((mat.normal_scale - 0.5).abs() < 1e-6, "normalTexture.scale");
        // factor 키가 없으므로 glTF 기본값 1.0 이어야 한다(텍스처와 곱해짐).
        assert_eq!(mat.metallic, 1.0);
        assert_eq!(mat.roughness, 1.0);
        assert!(mat.double_sided, "glasses1 은 doubleSided=true");
    }

    #[test]
    fn mustache_is_single_sided() {
        // 수염은 doubleSided 키가 없다 → glTF 기본 false → three 는 뒷면 컬링.
        // true 로 잘못 읽으면 뒷면이 노멀 뒤집힌 채로 같이 그려진다.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/demo/assets/glb/mustache1.glb"
        );
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let model = parse_glb(&bytes).expect("mustache1 parse");
        assert!(!model.materials[0].double_sided);
    }

    #[test]
    fn glasses3_transmissive_lens_uses_transmission_factor() {
        // KHR_materials_transmission 은 알파를 0 으로 만드는 게 아니다.
        // three 는 투과 패스 배경을 흰색·알파 0.5 로 클리어하고 그걸 샘플하므로
        // 렌즈가 반투명 유리로 남는다. 알파를 깎으면 렌즈에 구멍이 뚫린다.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../web/demo/assets/glb/glasses3.glb"
        );
        let Ok(bytes) = std::fs::read(path) else {
            return; // 에셋 없는 환경은 스킵
        };
        let model = parse_glb(&bytes).expect("glasses3 parse");
        let lens = model
            .materials
            .iter()
            .find(|m| m.transmission > 0.0)
            .expect("투과 재질을 못 찾음 — transmission 파싱 확인");
        assert_eq!(lens.transmission, 1.0);
        assert!(lens.blend, "투과 재질은 블렌드 패스로 가야 한다");
        assert!(
            lens.base_color[3] > 0.99,
            "베이스 알파를 깎으면 안 된다(셰이더가 transmission 으로 처리)"
        );
        // 나머지 재질은 투과 0.
        assert_eq!(model.materials.iter().filter(|m| m.transmission > 0.0).count(), 1);
    }

    #[test]
    fn env_cube_bytes_match_mip_chain() {
        // scripts/bake_room_env.py 파라미터가 바뀌면 여기서 잡는다 — 크기가 어긋나면
        // 업로드가 조용히 잘리거나 밉이 밀린다.
        let mut expect = 0usize;
        for mip in 0..ENV_CUBE_MIPS {
            let s = (ENV_CUBE_SIZE >> mip).max(1) as usize;
            expect += s * s * 4 * 2 * 6; // RGBA16F x 6면
        }
        assert_eq!(
            ENV_CUBE_BYTES.len(),
            expect,
            "env_room.bin 크기 불일치 — bake_room_env.py 의 mip_sizes 와 ENV_CUBE_SIZE/MIPS 확인"
        );
    }

    #[test]
    fn real_glb_assets_parse() {
        // 리포에 실제 에셋이 있으면 11종 전부 파싱 검증 (없으면 스킵).
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/assets/glb");
        for kind in HAT_KINDS
            .iter()
            .chain(EYEWEAR_KINDS.iter())
            .chain(BEARD_KINDS.iter())
        {
            let path = format!("{dir}/{kind}.glb");
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let model = parse_glb(&bytes).unwrap_or_else(|e| panic!("{kind}: {e}"));
            assert!(!model.primitives.is_empty(), "{kind}: no primitives");
            // NORMAL 이 없는 GLB(수염 2종)도 파싱 후에는 면 노멀이 채워져 있어야 한다 —
            // 폴백 상수 노멀이 남으면 고개 각도에 따라 색이 통째로 바뀐다.
            for (pi, prim) in model.primitives.iter().enumerate() {
                assert_eq!(
                    prim.normals.len(),
                    prim.positions.len(),
                    "{kind}[{pi}]: normal count mismatch"
                );
                for n in &prim.normals {
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    assert!(
                        (len - 1.0).abs() < 1e-3,
                        "{kind}[{pi}]: normal not unit ({len})"
                    );
                }
            }
        }
    }

    #[test]
    fn items3d_wgsl_parses() {
        // 렌더 셰이더 정적 검증 — 기기 없이 파싱/검증 게이트 (video_kernels 테스트와 동일 패턴).
        let src = include_str!("items3d.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("items3d.wgsl parse error");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .expect("items3d.wgsl validation error");

        // 백엔드 코드젠까지 확인 — WGSL 검증만 통과하고 MSL/SPIR-V 생성에서
        // 깨지면 기기에서만 파이프라인 생성이 실패한다(아이템이 통째로 안 뜸).
        let mut msl_opts = naga::back::msl::Options::default();
        msl_opts.lang_version = (2, 0);
        naga::back::msl::write_string(
            &module,
            &info,
            &msl_opts,
            &naga::back::msl::PipelineOptions::default(),
        )
        .expect("items3d.wgsl → MSL (iOS)");
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .expect("items3d.wgsl → SPIR-V (Android)");

        // 바인딩 수를 Rust 레이아웃 상수와 대조 — 어긋나면 기기에서만
        // create_bind_group 이 BindingsNumMismatch 로 죽고 아이템이 전멸한다.
        let mut per_group = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        for (_, g) in module.global_variables.iter() {
            if let Some(b) = g.binding.as_ref() {
                per_group.entry(b.group).or_default().push(b.binding);
            }
        }
        for slots in per_group.values_mut() {
            slots.sort_unstable();
        }
        assert_eq!(
            per_group.get(&0).map(Vec::as_slice),
            Some([0u32, 1, 2].as_slice()),
            "group(0) 바인딩이 CAM_BINDING_COUNT 와 불일치"
        );
        assert_eq!(
            per_group.get(&1).map(Vec::as_slice),
            Some([0u32, 1, 2, 3, 4].as_slice()),
            "group(1) 바인딩이 MESH_BINDING_COUNT 와 불일치"
        );
        assert_eq!(per_group[&0].len(), CAM_BINDING_COUNT);
        assert_eq!(per_group[&1].len(), MESH_BINDING_COUNT);
    }

    #[test]
    fn blit_wgsl_parses() {
        let m = naga::front::wgsl::parse_str(BLIT_WGSL).expect("blit parse");
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(&m)
            .expect("blit validate");
    }
}
