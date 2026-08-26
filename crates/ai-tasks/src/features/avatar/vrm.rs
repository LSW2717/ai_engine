// vrm — VRM 0.x 아바타 컨테이너 파서 (GLB + `extensions.VRM`).
//
// items3d 의 GLB 파서와 별개인 이유: 소비 형태가 다르다 — items3d 는 노드 월드를
// 구워 리지드 메시로 평탄화하지만, 아바타는 스켈레톤(조인트 트리)·스킨 가중치·
// 모프 타깃·humanoid 본맵을 **살아있는 구조로** 유지해야 한다. 공유점은 바이트
// 레이아웃 지식뿐이라 코드 공유 이득이 없다.
//
// 지원 범위는 보유 에셋(Open Source Avatars CC0 6종)이 실제로 쓰는 부분집합으로
// 한정하고, 벗어나면 **조용한 오렌더 대신 명시 에러**를 낸다:
// - accessor: 비-sparse, 밀집(tightly packed) — POSITION/NORMAL f32vec3,
//   TEXCOORD_0 f32vec2, JOINTS_0 u16vec4(+u8 허용), WEIGHTS_0 f32vec4,
//   indices u16/u32, IBM f32mat4
// - 노드 변환: TRS (matrix 노드는 미지원 에러)
// - 머티리얼: VRM/MToon + VRM_USE_GLTFSHADER(표준 PBR — MToon 기본값으로 수용)
//
// 좌표 규약: VRM 0.x 은 +Z 를 바라보게 저작된다(glTF 관례 -Z 와 반대) —
// 렌더러가 Y축 180° 회전으로 흡수한다. 파서는 원본 좌표를 보존한다.

use std::collections::HashMap;

use serde_json::Value;

type Result<T> = std::result::Result<T, String>;

pub type Mat4 = [f32; 16];

// ── 파싱 결과 ──

pub struct VrmNode {
    pub name: String,
    pub parent: Option<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4], // (x, y, z, w)
    pub scale: [f32; 3],
}

/// 모프 타깃 1개 — 정점 델타 (프리미티브 정점 수와 동일 길이).
pub struct MorphTarget {
    pub d_pos: Vec<[f32; 3]>,
    pub d_normal: Vec<[f32; 3]>, // 없으면 빈 벡터
}

pub struct VrmPrimitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub targets: Vec<MorphTarget>,
    pub material: usize,
}

/// blendShapeGroup 1개 — preset 이름과 (타깃 index, 가중 0..1) 바인드 목록.
/// mesh 단위 바인드지만 보유 에셋은 mesh 1개라 프리미티브 전역으로 본다.
pub struct MorphGroup {
    pub preset: String,
    pub binds: Vec<(usize, f32)>,
}

/// MToon 머티리얼 — v1 렌더에 필요한 부분집합.
pub struct MToonMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub shade_color: [f32; 4],
    pub outline_color: [f32; 4],
    /// _OutlineWidth (MToon 규약: cm 단위 → 렌더 시 0.01 배)
    pub outline_width: f32,
    pub cutoff: f32,
    /// _BlendMode: 0=opaque 1=cutout 2=transparent 3=transparent+zwrite
    pub blend_mode: u32,
    /// _CullMode: 0=off 1=front 2=back
    pub cull_mode: u32,
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub main_tex: Option<usize>,  // images 인덱스
    pub shade_tex: Option<usize>, // images 인덱스
}

pub struct VrmImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// 스프링본 체인 그룹 (secondaryAnimation.boneGroups 1개) — v1 은 파싱만.
pub struct SpringGroup {
    pub stiffness: f32,
    pub gravity_power: f32,
    pub gravity_dir: [f32; 3],
    pub drag_force: f32,
    pub hit_radius: f32,
    pub root_nodes: Vec<usize>,
}

pub struct VrmModel {
    pub nodes: Vec<VrmNode>,
    /// 스킨 조인트 → 노드 인덱스 (팔레트 순서)
    pub skin_joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
    /// humanoid 본 이름("head"/"neck"/"leftUpperArm"…) → 노드 인덱스
    pub humanoid: HashMap<String, usize>,
    pub primitives: Vec<VrmPrimitive>,
    pub morph_groups: Vec<MorphGroup>,
    pub materials: Vec<MToonMaterial>,
    pub images: Vec<VrmImage>,
    pub springs: Vec<SpringGroup>,
    pub meta_title: String,
}

impl VrmModel {
    /// blink 계열 바인드 — 통합 `blink` 우선, 없으면 `blink_l`+`blink_r` 합산
    /// (EggBOY 처럼 좌우 분리만 있는 에셋 흡수). (타깃 idx, 가중) 목록.
    pub fn blink_binds(&self) -> Vec<(usize, f32)> {
        if let Some(g) = self.morph_groups.iter().find(|g| g.preset == "blink") {
            return g.binds.clone();
        }
        let mut out: Vec<(usize, f32)> = Vec::new();
        for g in &self.morph_groups {
            if g.preset == "blink_l" || g.preset == "blink_r" {
                for &(idx, w) in &g.binds {
                    match out.iter_mut().find(|(i, _)| *i == idx) {
                        Some(e) => e.1 = e.1.max(w),
                        None => out.push((idx, w)),
                    }
                }
            }
        }
        out
    }

    /// 입 벌림 `a` 바인드 (마이크 구동 입뻐끔용 — v1 미사용, 배선만).
    pub fn aa_binds(&self) -> Vec<(usize, f32)> {
        self.morph_groups
            .iter()
            .find(|g| g.preset == "a" || g.preset == "aa")
            .map(|g| g.binds.clone())
            .unwrap_or_default()
    }
}

// ── GLB 청크/accessor 리더 ──

struct Access<'a> {
    json: &'a Value,
    bin: &'a [u8],
}

const CT_I8: u64 = 5120;
const CT_U8: u64 = 5121;
const CT_I16: u64 = 5122;
const CT_U16: u64 = 5123;
const CT_U32: u64 = 5125;
const CT_F32: u64 = 5126;

fn comp_size(ct: u64) -> usize {
    match ct {
        CT_I8 | CT_U8 => 1,
        CT_I16 | CT_U16 => 2,
        _ => 4,
    }
}

fn type_comps(t: &str) -> usize {
    match t {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => 0,
    }
}

impl<'a> Access<'a> {
    /// accessor idx → (원소 수, componentType, 원소당 컴포넌트 수, 바이트 슬라이스).
    /// sparse/인터리브(byteStride≠밀집)는 명시 에러.
    fn raw(&self, idx: usize) -> Result<(usize, u64, usize, &'a [u8])> {
        let acc = self.json["accessors"]
            .get(idx)
            .ok_or_else(|| format!("accessor {idx} 없음"))?;
        if acc.get("sparse").is_some() {
            return Err(format!("accessor {idx}: sparse 미지원"));
        }
        let count = acc["count"].as_u64().unwrap_or(0) as usize;
        let ct = acc["componentType"].as_u64().unwrap_or(0);
        let comps = type_comps(acc["type"].as_str().unwrap_or(""));
        if comps == 0 {
            return Err(format!("accessor {idx}: type 미지원"));
        }
        let elem = comp_size(ct) * comps;
        let bv_idx = acc["bufferView"]
            .as_u64()
            .ok_or_else(|| format!("accessor {idx}: bufferView 없음"))? as usize;
        let bv = &self.json["bufferViews"][bv_idx];
        if let Some(stride) = bv.get("byteStride").and_then(|s| s.as_u64()) {
            if stride as usize != elem {
                return Err(format!("accessor {idx}: 인터리브 stride 미지원"));
            }
        }
        let off = bv["byteOffset"].as_u64().unwrap_or(0) as usize
            + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = count * elem;
        let end = off + len;
        if end > self.bin.len() {
            return Err(format!("accessor {idx}: BIN 범위 초과 ({end} > {})", self.bin.len()));
        }
        Ok((count, ct, comps, &self.bin[off..end]))
    }

    fn f32_n<const N: usize>(&self, idx: usize) -> Result<Vec<[f32; N]>> {
        let (count, ct, comps, bytes) = self.raw(idx)?;
        if ct != CT_F32 || comps != N {
            return Err(format!("accessor {idx}: f32×{N} 아님 (ct {ct}, comps {comps})"));
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut v = [0f32; N];
            for (c, vc) in v.iter_mut().enumerate() {
                let o = (i * N + c) * 4;
                *vc = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            }
            out.push(v);
        }
        Ok(out)
    }

    fn u16x4(&self, idx: usize) -> Result<Vec<[u16; 4]>> {
        let (count, ct, comps, bytes) = self.raw(idx)?;
        if comps != 4 {
            return Err(format!("accessor {idx}: vec4 아님"));
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut v = [0u16; 4];
            for (c, vc) in v.iter_mut().enumerate() {
                *vc = match ct {
                    CT_U8 => bytes[i * 4 + c] as u16,
                    CT_U16 => {
                        let o = (i * 4 + c) * 2;
                        u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap())
                    }
                    _ => return Err(format!("accessor {idx}: JOINTS 타입 미지원 (ct {ct})")),
                };
            }
            out.push(v);
        }
        Ok(out)
    }

    fn indices(&self, idx: usize) -> Result<Vec<u32>> {
        let (count, ct, comps, bytes) = self.raw(idx)?;
        if comps != 1 {
            return Err(format!("accessor {idx}: SCALAR 아님"));
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(match ct {
                CT_U16 => {
                    let o = i * 2;
                    u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap()) as u32
                }
                CT_U32 => {
                    let o = i * 4;
                    u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
                }
                _ => return Err(format!("accessor {idx}: index 타입 미지원 (ct {ct})")),
            });
        }
        Ok(out)
    }

    fn mat4s(&self, idx: usize) -> Result<Vec<Mat4>> {
        let (count, ct, comps, bytes) = self.raw(idx)?;
        if ct != CT_F32 || comps != 16 {
            return Err(format!("accessor {idx}: MAT4 f32 아님"));
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut m = [0f32; 16];
            for (c, mc) in m.iter_mut().enumerate() {
                let o = (i * 16 + c) * 4;
                *mc = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            }
            out.push(m);
        }
        Ok(out)
    }

    fn image_bytes(&self, img_idx: usize) -> Result<&'a [u8]> {
        let img = self.json["images"]
            .get(img_idx)
            .ok_or_else(|| format!("image {img_idx} 없음"))?;
        let bv_idx = img["bufferView"]
            .as_u64()
            .ok_or_else(|| format!("image {img_idx}: bufferView 없음 (외부 URI 미지원)"))?
            as usize;
        let bv = &self.json["bufferViews"][bv_idx];
        let off = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = bv["byteLength"].as_u64().unwrap_or(0) as usize;
        if off + len > self.bin.len() {
            return Err(format!("image {img_idx}: BIN 범위 초과"));
        }
        Ok(&self.bin[off..off + len])
    }
}

fn json_f32(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default as f64) as f32
}

fn json_vec3(v: &Value, key: &str, default: [f32; 3]) -> [f32; 3] {
    match v.get(key).and_then(|x| x.as_array()) {
        Some(a) if a.len() >= 3 => [
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
        ],
        _ => default,
    }
}

fn json_vec4(v: &Value, key: &str, default: [f32; 4]) -> [f32; 4] {
    match v.get(key).and_then(|x| x.as_array()) {
        Some(a) if a.len() >= 4 => [
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
            a[3].as_f64().unwrap_or(1.0) as f32,
        ],
        _ => default,
    }
}

// ── 본체 ──

pub fn parse_vrm(bytes: &[u8]) -> Result<VrmModel> {
    // GLB 헤더/청크
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err("GLB 매직 아님".into());
    }
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_end = 20 + json_len;
    if bytes.len() < json_end || &bytes[16..20] != b"JSON" {
        return Err("GLB JSON 청크 손상".into());
    }
    let json: Value = serde_json::from_slice(&bytes[20..json_end])
        .map_err(|e| format!("GLB JSON 파싱 실패: {e}"))?;
    let bin = if bytes.len() >= json_end + 8 {
        let bin_len =
            u32::from_le_bytes(bytes[json_end..json_end + 4].try_into().unwrap()) as usize;
        let start = json_end + 8;
        bytes
            .get(start..start + bin_len)
            .ok_or("GLB BIN 청크 손상")?
    } else {
        &[]
    };
    let acc = Access { json: &json, bin };

    let vrm = json
        .get("extensions")
        .and_then(|e| e.get("VRM"))
        .ok_or("extensions.VRM 없음 — VRM 0.x 아님 (1.0 VRMC_vrm 은 미지원)")?;

    // ── 노드 트리 (TRS) ──
    let jnodes = json["nodes"].as_array().ok_or("nodes 없음")?;
    let mut parents: Vec<Option<usize>> = vec![None; jnodes.len()];
    for (i, n) in jnodes.iter().enumerate() {
        if let Some(ch) = n.get("children").and_then(|c| c.as_array()) {
            for c in ch {
                if let Some(ci) = c.as_u64() {
                    parents[ci as usize] = Some(i);
                }
            }
        }
    }
    let mut nodes = Vec::with_capacity(jnodes.len());
    for (i, n) in jnodes.iter().enumerate() {
        if n.get("matrix").is_some() {
            return Err(format!("node {i}: matrix 변환 미지원 (TRS만)"));
        }
        nodes.push(VrmNode {
            name: n.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            parent: parents[i],
            translation: json_vec3(n, "translation", [0.0, 0.0, 0.0]),
            rotation: json_vec4(n, "rotation", [0.0, 0.0, 0.0, 1.0]),
            scale: json_vec3(n, "scale", [1.0, 1.0, 1.0]),
        });
    }

    // ── 스킨 (0번만 — 보유 에셋 전부 1개) ──
    let skin = json["skins"].get(0).ok_or("skins 없음")?;
    let skin_joints: Vec<usize> = skin["joints"]
        .as_array()
        .ok_or("skin.joints 없음")?
        .iter()
        .filter_map(|j| j.as_u64().map(|x| x as usize))
        .collect();
    let ibm_idx = skin["inverseBindMatrices"]
        .as_u64()
        .ok_or("skin.inverseBindMatrices 없음")? as usize;
    let inverse_bind = acc.mat4s(ibm_idx)?;
    if inverse_bind.len() != skin_joints.len() {
        return Err("IBM 수 ≠ joints 수".into());
    }

    // ── humanoid 본맵 ──
    let mut humanoid = HashMap::new();
    if let Some(bones) = vrm
        .get("humanoid")
        .and_then(|h| h.get("humanBones"))
        .and_then(|b| b.as_array())
    {
        for b in bones {
            if let (Some(name), Some(node)) = (
                b.get("bone").and_then(|s| s.as_str()),
                b.get("node").and_then(|n| n.as_u64()),
            ) {
                humanoid.insert(name.to_string(), node as usize);
            }
        }
    }
    if !humanoid.contains_key("head") {
        return Err("humanoid.head 없음".into());
    }

    // ── 메시/프리미티브 ──
    let jmeshes = json["meshes"].as_array().ok_or("meshes 없음")?;
    let mut primitives = Vec::new();
    for (mi, m) in jmeshes.iter().enumerate() {
        for (pi, p) in m["primitives"].as_array().unwrap_or(&Vec::new()).iter().enumerate() {
            let at = &p["attributes"];
            let need = |k: &str| -> Result<usize> {
                at.get(k)
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .ok_or_else(|| format!("mesh {mi} prim {pi}: {k} 없음"))
            };
            let positions = acc.f32_n::<3>(need("POSITION")?)?;
            let normals = acc.f32_n::<3>(need("NORMAL")?)?;
            let uvs = acc.f32_n::<2>(need("TEXCOORD_0")?)?;
            let joints = acc.u16x4(need("JOINTS_0")?)?;
            let weights = acc.f32_n::<4>(need("WEIGHTS_0")?)?;
            let n = positions.len();
            if normals.len() != n || uvs.len() != n || joints.len() != n || weights.len() != n {
                return Err(format!("mesh {mi} prim {pi}: 속성 길이 불일치"));
            }
            let indices = acc.indices(
                p["indices"]
                    .as_u64()
                    .ok_or_else(|| format!("mesh {mi} prim {pi}: indices 없음"))?
                    as usize,
            )?;
            let mut targets = Vec::new();
            for (ti, t) in p["targets"].as_array().unwrap_or(&Vec::new()).iter().enumerate() {
                let d_pos = acc.f32_n::<3>(
                    t.get("POSITION")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| format!("mesh {mi} prim {pi} target {ti}: POSITION 없음"))?
                        as usize,
                )?;
                if d_pos.len() != n {
                    return Err(format!("mesh {mi} prim {pi} target {ti}: 델타 길이 불일치"));
                }
                let d_normal = match t.get("NORMAL").and_then(|v| v.as_u64()) {
                    Some(a) => acc.f32_n::<3>(a as usize)?,
                    None => Vec::new(),
                };
                targets.push(MorphTarget { d_pos, d_normal });
            }
            primitives.push(VrmPrimitive {
                positions,
                normals,
                uvs,
                joints,
                weights,
                indices,
                targets,
                material: p["material"].as_u64().unwrap_or(0) as usize,
            });
        }
    }
    if primitives.is_empty() {
        return Err("프리미티브 없음".into());
    }

    // ── blendShapeGroups ──
    let mut morph_groups = Vec::new();
    if let Some(groups) = vrm
        .get("blendShapeMaster")
        .and_then(|b| b.get("blendShapeGroups"))
        .and_then(|g| g.as_array())
    {
        for g in groups {
            let preset = g
                .get("presetName")
                .and_then(|s| s.as_str())
                .filter(|s| !s.is_empty() && *s != "unknown")
                .or_else(|| g.get("name").and_then(|s| s.as_str()))
                .unwrap_or("")
                .to_lowercase();
            let mut binds = Vec::new();
            for b in g.get("binds").and_then(|b| b.as_array()).unwrap_or(&Vec::new()) {
                if let Some(idx) = b.get("index").and_then(|v| v.as_u64()) {
                    let w = json_f32(b, "weight", 100.0) / 100.0;
                    binds.push((idx as usize, w));
                }
            }
            if !preset.is_empty() && !binds.is_empty() {
                morph_groups.push(MorphGroup { preset, binds });
            }
        }
    }

    // ── MToon 머티리얼 (VRM materialProperties 기준 — glTF materials 는 폴백) ──
    let jtextures = json["textures"].as_array().cloned().unwrap_or_default();
    let tex_to_image = |tex_idx: f64| -> Option<usize> {
        jtextures
            .get(tex_idx as usize)
            .and_then(|t| t.get("source"))
            .and_then(|s| s.as_u64())
            .map(|s| s as usize)
    };
    let mut materials = Vec::new();
    if let Some(props) = vrm.get("materialProperties").and_then(|m| m.as_array()) {
        for mp in props {
            let f = mp.get("floatProperties").cloned().unwrap_or(Value::Null);
            let v = mp.get("vectorProperties").cloned().unwrap_or(Value::Null);
            let t = mp.get("textureProperties").cloned().unwrap_or(Value::Null);
            let main_tex = t.get("_MainTex").and_then(|x| x.as_f64()).and_then(tex_to_image);
            let shade_tex = t
                .get("_ShadeTexture")
                .and_then(|x| x.as_f64())
                .and_then(tex_to_image);
            materials.push(MToonMaterial {
                name: mp.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                base_color: json_vec4(&v, "_Color", [1.0, 1.0, 1.0, 1.0]),
                shade_color: json_vec4(&v, "_ShadeColor", [0.8, 0.75, 0.8, 1.0]),
                outline_color: json_vec4(&v, "_OutlineColor", [0.0, 0.0, 0.0, 1.0]),
                outline_width: json_f32(&f, "_OutlineWidth", 0.0),
                cutoff: json_f32(&f, "_Cutoff", 0.5),
                blend_mode: json_f32(&f, "_BlendMode", 0.0) as u32,
                cull_mode: json_f32(&f, "_CullMode", 2.0) as u32,
                shade_shift: json_f32(&f, "_ShadeShift", 0.0),
                shade_toony: json_f32(&f, "_ShadeToony", 0.9),
                main_tex,
                shade_tex,
            });
        }
    }
    if materials.is_empty() {
        // VRM_USE_GLTFSHADER 등 materialProperties 없는 에셋 — 기본 MToon 1개
        materials.push(MToonMaterial {
            name: "default".into(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            shade_color: [0.7, 0.65, 0.7, 1.0],
            outline_color: [0.0, 0.0, 0.0, 1.0],
            outline_width: 0.0,
            cutoff: 0.5,
            blend_mode: 0,
            cull_mode: 2,
            shade_shift: 0.0,
            shade_toony: 0.9,
            main_tex: Some(0),
            shade_tex: None,
        });
    }

    // ── 이미지 디코드 (PNG/JPEG — image 크레이트) ──
    let n_images = json["images"].as_array().map(|a| a.len()).unwrap_or(0);
    let mut images = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let raw = acc.image_bytes(i)?;
        let img = image::load_from_memory(raw)
            .map_err(|e| format!("image {i} 디코드 실패: {e}"))?
            .to_rgba8();
        images.push(VrmImage { width: img.width(), height: img.height(), rgba: img.into_raw() });
    }

    // ── 스프링본 (파싱만 — 시뮬은 후속) ──
    let mut springs = Vec::new();
    if let Some(groups) = vrm
        .get("secondaryAnimation")
        .and_then(|s| s.get("boneGroups"))
        .and_then(|g| g.as_array())
    {
        for g in groups {
            let root_nodes: Vec<usize> = g
                .get("bones")
                .and_then(|b| b.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect())
                .unwrap_or_default();
            if root_nodes.is_empty() {
                continue;
            }
            springs.push(SpringGroup {
                stiffness: json_f32(g, "stiffiness", 1.0).max(json_f32(g, "stiffness", 0.0)),
                gravity_power: json_f32(g, "gravityPower", 0.0),
                gravity_dir: json_vec3(g, "gravityDir", [0.0, -1.0, 0.0]),
                drag_force: json_f32(g, "dragForce", 0.4),
                hit_radius: json_f32(g, "hitRadius", 0.02),
                root_nodes,
            });
        }
    }

    let meta_title = vrm
        .get("meta")
        .and_then(|m| m.get("title"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    Ok(VrmModel {
        nodes,
        skin_joints,
        inverse_bind,
        humanoid,
        primitives,
        morph_groups,
        materials,
        images,
        springs,
        meta_title,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_dir() -> String {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/assets/vrm").to_string()
    }

    const ASSETS: [&str; 6] = ["Butter", "CoolBanana", "EggBOY", "Hotdog", "Milk", "Mushy"];

    #[test]
    fn real_vrm_assets_parse() {
        // 리포에 실제 에셋이 있으면 6종 전부 검증 (없으면 스킵 — items3d 와 동일 규약)
        for name in ASSETS {
            let path = format!("{}/{name}.vrm", asset_dir());
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let m = parse_vrm(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!m.primitives.is_empty(), "{name}: no primitives");
            assert!(!m.blink_binds().is_empty(), "{name}: blink 바인드 없음");
            assert!(!m.aa_binds().is_empty(), "{name}: aa 바인드 없음");
            assert!(m.humanoid.contains_key("head"), "{name}: head 본 없음");
            assert!(m.humanoid.contains_key("neck") || m.humanoid.contains_key("spine"));
            assert_eq!(m.inverse_bind.len(), m.skin_joints.len());
            assert!(!m.images.is_empty(), "{name}: 텍스처 없음");
            for (pi, p) in m.primitives.iter().enumerate() {
                let n = p.positions.len();
                assert!(n > 0 && p.indices.len() % 3 == 0, "{name}[{pi}]");
                // 스킨 가중치 합 ≈ 1 (정규화 확인 — 렌더에서 재정규화 안 함)
                for (vi, w) in p.weights.iter().enumerate().step_by(97) {
                    let s: f32 = w.iter().sum();
                    assert!(
                        (s - 1.0).abs() < 1e-2,
                        "{name}[{pi}] v{vi}: weight sum {s}"
                    );
                }
                // blink 바인드가 가리키는 타깃이 실제로 존재
                for &(ti, _) in &m.blink_binds() {
                    assert!(ti < p.targets.len(), "{name}[{pi}]: blink 타깃 {ti} 범위 밖");
                    assert_eq!(p.targets[ti].d_pos.len(), n);
                }
            }
            // 조인트 인덱스가 팔레트 범위 안
            let nj = m.skin_joints.len() as u16;
            for p in &m.primitives {
                for j in p.joints.iter().step_by(53) {
                    assert!(j.iter().all(|&x| x < nj), "{name}: joint idx 범위 밖");
                }
            }
        }
    }

    #[test]
    fn rejects_non_glb() {
        assert!(parse_vrm(b"not a glb").is_err());
    }
}
