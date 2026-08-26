// rig — 아바타 스켈레톤 포즈 솔버 + blink 컨트롤러.
//
// 입력은 items3d 와 같은 얼굴 피팅 결과(카메라계 머리 쿼터니언)와 478pt
// 랜드마크의 눈 종횡비(EAR)뿐 — 표정 전체 추적은 스코프 밖(설계 결정).
//
// 머리 회전 분배: 피팅 쿼터니언을 head 70% / neck 30% 로 나눠 꽂는다
// (전부 head 에 주면 목이 굳은 인형처럼 보인다). 회전은 rest 월드 기준
// 상대 회전으로 계산해 로컬에 재투영한다 — 본 로컬축이 제각각이어도 안전.
//
// blink: 관측(EAR→0..1)이 있으면 빠른 attack/느린 release 로 추종, 얼굴을
// 잃으면 자동 깜빡임(가변 주기 2.8~5.2s, 140ms 봉투)으로 살아있게 유지한다.
// 관측 원값을 프레임마다 그대로 꽂지 않는 이유: 낮은 detect fps 에선 깜빡임
// (~120ms)이 1~2 샘플이라 raw 추종은 뚝뚝 끊긴다 — 관측은 "감김 정도"의
// 목표값이고 봉투는 로컬에서 그린다.

use std::collections::HashMap;

use super::vrm::{Mat4, VrmModel};

pub type Quat = [f32; 4]; // (x, y, z, w)

// ── 미니 수학 (열우선 mat4 — wgpu/WGSL 규약, items3d 와 동일) ──

pub fn mat_identity() -> Mat4 {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
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

pub fn mat_compose(pos: [f32; 3], q: Quat, scale: [f32; 3]) -> Mat4 {
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

pub fn mat_perspective(vfov_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (vfov_rad / 2.0).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = (far * near) / (near - far);
    m
}

pub fn quat_normalize(q: Quat) -> Quat {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l < 1e-9 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

pub fn quat_conj(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn quat_slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let mut cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b2 = b;
    if cos < 0.0 {
        cos = -cos;
        b2 = [-b[0], -b[1], -b[2], -b[3]];
    }
    if cos > 0.9995 {
        return quat_normalize([
            a[0] + (b2[0] - a[0]) * t,
            a[1] + (b2[1] - a[1]) * t,
            a[2] + (b2[2] - a[2]) * t,
            a[3] + (b2[3] - a[3]) * t,
        ]);
    }
    let theta = cos.clamp(-1.0, 1.0).acos();
    let s = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / s;
    let wb = (t * theta).sin() / s;
    [
        a[0] * wa + b2[0] * wb,
        a[1] * wa + b2[1] * wb,
        a[2] * wa + b2[2] * wb,
        a[3] * wa + b2[3] * wb,
    ]
}

fn quat_from_yaw(rad: f32) -> Quat {
    [0.0, (rad / 2.0).sin(), 0.0, (rad / 2.0).cos()]
}

pub fn quat_axis_angle(axis: [f32; 3], rad: f32) -> Quat {
    let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if l < 1e-9 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let s = (rad / 2.0).sin() / l;
    [axis[0] * s, axis[1] * s, axis[2] * s, (rad / 2.0).cos()]
}

/// a→b 최단 회전 (둘 다 정규화 전제 아님 — 내부 정규화)
fn quat_from_to(a: [f32; 3], b: [f32; 3]) -> Quat {
    let (a, b) = (v_norm(a), v_norm(b));
    let d = v_dot(a, b);
    if d > 0.99999 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -0.99999 {
        // 정반대 — 수직축 아무거나
        let ax = if a[0].abs() < 0.9 { v_cross([1.0, 0.0, 0.0], a) } else { v_cross([0.0, 1.0, 0.0], a) };
        return quat_axis_angle(ax, std::f32::consts::PI);
    }
    let c = v_cross(a, b);
    quat_normalize([c[0], c[1], c[2], 1.0 + d])
}

fn quat_rotate_vec(q: Quat, v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let uv = v_cross(u, v);
    let uuv = v_cross(u, uv);
    [
        v[0] + 2.0 * (s * uv[0] + uuv[0]),
        v[1] + 2.0 * (s * uv[1] + uuv[1]),
        v[2] + 2.0 * (s * uv[2] + uuv[2]),
    ]
}

fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn v_len(a: [f32; 3]) -> f32 {
    v_dot(a, a).sqrt()
}
fn v_norm(a: [f32; 3]) -> [f32; 3] {
    let l = v_len(a).max(1e-9);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn v_lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

// ── 스켈레톤 ──

struct Trs {
    t: [f32; 3],
    r: Quat,
    s: [f32; 3],
}

/// 팔 IK 체인 (humanoid upperArm→lowerArm→hand)
struct ArmChain {
    upper: usize,
    lower: usize,
    len_upper: f32,
    len_lower: f32,
    /// rest 월드 위치 (어깨/팔꿈치/손목)
    p_upper: [f32; 3],
    p_lower: [f32; 3],
    p_hand: [f32; 3],
    /// idle 손목 타깃 — 차렷(팔 내림). VRM rest 는 T-포즈라 그대로 두면
    /// 팔을 좌우로 펼친 채 서 있다 — 관측 없으면 항상 이쪽으로 IK.
    down_target: [f32; 3],
}

/// 손가락 관절 1개 — 로컬 굴곡축 (rest 본 로컬 프레임 고정이라 상위 본이
/// 움직여도 유효) + 최대 굴곡각
struct FingerJoint {
    node: usize,
    /// 손가락 번호 0=엄지..4=새끼 (curls 인덱스)
    finger: usize,
    axis_local: [f32; 3],
    max_rad: f32,
}

/// 팔 1개의 관측/스무딩 상태
#[derive(Default)]
struct ArmState {
    /// 모델 공간 손목 타깃 (이번 틱 관측 — None=미관측)
    obs: Option<[f32; 3]>,
    smoothed: Option<[f32; 3]>,
    /// 0..1 — 관측 있으면 상승, 없으면 감쇠. IK 는 rest↔target 을 이 값으로 블렌드
    strength: f32,
    curls_obs: [f32; 5],
    curls: [f32; 5],
}

pub struct Rig {
    rest: Vec<Trs>,
    /// 현재 프레임의 로컬 회전 오버라이드 (없으면 rest)
    rot_override: HashMap<usize, Quat>,
    parent: Vec<Option<usize>>,
    globals: Vec<Mat4>,
    /// rest 월드 회전 (스켈레톤은 scale 1 전제 — VRoid/OSA 에셋 공통)
    rest_world_q: Vec<Quat>,
    head: usize,
    neck: Option<usize>,
    /// 스킨 팔레트 입력 (VrmModel 에서 복사 — 렌더러가 모델 없이 팔레트를 뽑게)
    skin_joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
    /// [0]=아바타 왼팔, [1]=오른팔 (humanoid 기준)
    arms: [Option<ArmChain>; 2],
    /// 손가락 관절 (side, joint) — 엄지 3 + 4지×3 = 최대 15/손
    fingers: [Vec<FingerJoint>; 2],
    arm_state: [ArmState; 2],
}

impl Rig {
    pub fn new(model: &VrmModel) -> Self {
        let n = model.nodes.len();
        let rest: Vec<Trs> = model
            .nodes
            .iter()
            .map(|nd| Trs { t: nd.translation, r: nd.rotation, s: nd.scale })
            .collect();
        let parent: Vec<Option<usize>> = model.nodes.iter().map(|nd| nd.parent).collect();
        let mut rig = Rig {
            rest,
            rot_override: HashMap::new(),
            parent,
            globals: vec![mat_identity(); n],
            rest_world_q: vec![[0.0, 0.0, 0.0, 1.0]; n],
            head: model.humanoid["head"],
            neck: model.humanoid.get("neck").copied(),
            skin_joints: model.skin_joints.clone(),
            inverse_bind: model.inverse_bind.clone(),
            arms: [None, None],
            fingers: [Vec::new(), Vec::new()],
            arm_state: [ArmState::default(), ArmState::default()],
        };
        // rest 월드 회전 캐시 (부모가 항상 앞 인덱스라는 보장이 없어 반복 해소)
        let mut done = vec![false; n];
        for i in 0..n {
            rig.resolve_rest_q(i, &mut done);
        }
        // rest 월드 위치 — 팔 IK/프레이밍의 기준. 오버라이드 없는 상태로 1회 전개.
        rig.update_globals();
        let rest_p: Vec<[f32; 3]> = (0..n).map(|i| rig.node_world_pos(i)).collect();
        rig.build_arms(model, &rest_p);
        rig
    }

    fn build_arms(&mut self, model: &VrmModel, rest_p: &[[f32; 3]]) {
        for (side, prefix) in ["left", "right"].iter().enumerate() {
            let get = |bone: &str| model.humanoid.get(&format!("{prefix}{bone}")).copied();
            let (Some(upper), Some(lower), Some(hand)) =
                (get("UpperArm"), get("LowerArm"), get("Hand"))
            else {
                continue;
            };
            let (pu, pl, ph) = (rest_p[upper], rest_p[lower], rest_p[hand]);
            let reach = v_len(v_sub(pl, pu)) + v_len(v_sub(ph, pl));
            // 차렷: 어깨에서 거의 수직 아래 + 몸 바깥쪽으로 살짝 (몸 관통 방지)
            let outward = if pu[0] >= 0.0 { 1.0 } else { -1.0 };
            let down_target =
                v_add(pu, [outward * reach * 0.18, -reach * 0.92, reach * 0.06]);
            self.arms[side] = Some(ArmChain {
                upper,
                lower,
                len_upper: v_len(v_sub(pl, pu)).max(1e-4),
                len_lower: v_len(v_sub(ph, pl)).max(1e-4),
                p_upper: pu,
                p_lower: pl,
                p_hand: ph,
                down_target,
            });
            // 손가락 굴곡축: VRM T-포즈(손바닥 아래) 전제 — 굴곡 = rest 본 방향을
            // 월드 -Y 쪽으로 감는 회전. 축을 rest 로컬로 옮겨 두면 팔이 움직여도 유효.
            // 엄지는 -Y 대신 손바닥 안쪽(새끼 방향)으로 감는다.
            let inward = {
                let idx_p = get("IndexProximal");
                let lit_p = get("LittleProximal");
                match (idx_p, lit_p) {
                    (Some(i), Some(l)) => v_norm(v_sub(rest_p[l], rest_p[i])),
                    _ => [0.0, 0.0, -1.0],
                }
            };
            let mut joints = Vec::new();
            for (finger_idx, (finger, thumb)) in [
                ("Thumb", true),
                ("Index", false),
                ("Middle", false),
                ("Ring", false),
                ("Little", false),
            ]
            .into_iter()
            .enumerate()
            {
                for (seg, max_deg) in [("Proximal", 65.0f32), ("Intermediate", 80.0), ("Distal", 40.0)] {
                    let Some(node) = get(&format!("{finger}{seg}")) else { continue };
                    // 본 방향: 자식(다음 세그) 또는 부모 기준 근사
                    let child_p = joints_child_pos(model, node, rest_p);
                    let dir = v_norm(v_sub(child_p, rest_p[node]));
                    let bend_toward = if thumb { inward } else { [0.0, -1.0, 0.0] };
                    let axis_world = v_cross(dir, bend_toward);
                    if v_len(axis_world) < 1e-4 {
                        continue;
                    }
                    let axis_local =
                        quat_rotate_vec(quat_conj(self.rest_world_q[node]), v_norm(axis_world));
                    let scale = if thumb { 0.55 } else { 1.0 };
                    joints.push(FingerJoint {
                        node,
                        finger: finger_idx,
                        axis_local,
                        max_rad: max_deg.to_radians() * scale,
                    });
                }
            }
            self.fingers[side] = joints;
        }
    }

    fn resolve_rest_q(&mut self, i: usize, done: &mut [bool]) {
        if done[i] {
            return;
        }
        match self.parent[i] {
            Some(p) => {
                self.resolve_rest_q(p, done);
                self.rest_world_q[i] =
                    quat_normalize(quat_mul(self.rest_world_q[p], self.rest[i].r));
            }
            None => self.rest_world_q[i] = quat_normalize(self.rest[i].r),
        }
        done[i] = true;
    }

    /// 피팅 머리 회전(카메라계, 정면=항등)을 head/neck 에 분배한다.
    /// `root_yaw` = 모델 루트에 적용된 yaw (VRM0 전방 보정) — 카메라계 회전을
    /// 모델계로 옮길 때 같은 yaw 로 켤레변환해야 좌우가 맞는다.
    pub fn set_head_rotation(&mut self, fit_q: Quat, root_yaw: f32) {
        let ry = quat_from_yaw(root_yaw);
        let q_model = quat_normalize(quat_mul(quat_mul(quat_conj(ry), fit_q), ry));
        let id = [0.0, 0.0, 0.0, 1.0];
        let q_head = quat_slerp(id, q_model, 0.7);
        let q_neck = quat_slerp(id, q_model, 0.3);
        self.apply_world_delta(self.head, q_head);
        if let Some(neck) = self.neck {
            self.apply_world_delta(neck, q_neck);
        }
    }

    /// 본의 월드 회전에 delta 를 곱한 것과 등가인 로컬 회전을 대입:
    /// world' = delta · world_rest  ⇒  local' = conj(parent_world) · delta · world_rest
    /// (parent 월드는 rest 근사 — neck/head 지터 케이스에서 오차는 2차항)
    fn apply_world_delta(&mut self, node: usize, delta: Quat) {
        let pw = match self.parent[node] {
            Some(p) => self.rest_world_q[p],
            None => [0.0, 0.0, 0.0, 1.0],
        };
        let target_world = quat_mul(delta, self.rest_world_q[node]);
        let local = quat_normalize(quat_mul(quat_conj(pw), target_world));
        self.rot_override.insert(node, local);
    }

    pub fn clear_pose(&mut self) {
        self.rot_override.clear();
    }

    /// 로컬 → 월드 전파. 호출 후 `globals()`/`joint_palette()` 유효.
    pub fn update_globals(&mut self) {
        let n = self.rest.len();
        let mut done = vec![false; n];
        for i in 0..n {
            self.resolve_global(i, &mut done);
        }
    }

    fn resolve_global(&mut self, i: usize, done: &mut [bool]) {
        if done[i] {
            return;
        }
        let r = *self.rot_override.get(&i).unwrap_or(&self.rest[i].r);
        let local = mat_compose(self.rest[i].t, r, self.rest[i].s);
        self.globals[i] = match self.parent[i] {
            Some(p) => {
                self.resolve_global(p, done);
                mat_mul(&self.globals[p], &local)
            }
            None => local,
        };
        done[i] = true;
    }

    /// 스킨 팔레트: joint 순서대로 global · inverseBind
    pub fn joint_palette(&self) -> Vec<Mat4> {
        self.skin_joints
            .iter()
            .zip(self.inverse_bind.iter())
            .map(|(&node, ibm)| mat_mul(&self.globals[node], ibm))
            .collect()
    }

    pub fn node_world_pos(&self, node: usize) -> [f32; 3] {
        let m = &self.globals[node];
        [m[12], m[13], m[14]]
    }

    pub fn head_node(&self) -> usize {
        self.head
    }

    // ── 팔/손가락 ──

    /// 이번 틱의 손 관측 주입. side: 0=아바타 왼팔(화면 오른쪽), 1=오른팔.
    /// wrist = 모델 공간 손목 타깃, curls = 손가락 5개 굴곡 0..1.
    pub fn set_hand_obs(&mut self, side: usize, wrist: Option<[f32; 3]>, curls: Option<[f32; 5]>) {
        let st = &mut self.arm_state[side];
        st.obs = wrist;
        if let Some(c) = curls {
            st.curls_obs = c;
        }
    }

    /// 스무딩 + 2-bone IK + 손가락 컬 오버라이드 적용 — clear_pose 후·update_globals 전.
    pub fn tick_arms(&mut self) {
        for side in 0..2 {
            let Some(chain) = &self.arms[side] else { continue };
            let st = &mut self.arm_state[side];
            // strength: 관측 시 상승(0.25/틱), 소실 시 감쇠(0.08/틱 — 스르륵 복귀)
            match st.obs {
                Some(t) => {
                    st.strength = (st.strength + 0.25).min(1.0);
                    st.smoothed = Some(match st.smoothed {
                        Some(prev) => v_lerp(prev, t, 0.5),
                        None => t,
                    });
                }
                None => {
                    st.strength = (st.strength - 0.08).max(0.0);
                    if st.strength <= 0.0 {
                        st.smoothed = None;
                    }
                    // idle 손 — 살짝 말린 자연스러운 손 (곧게 편 손은 인형 느낌)
                    for c in st.curls_obs.iter_mut() {
                        *c += (0.12 - *c) * 0.2;
                    }
                }
            }
            for i in 0..5 {
                st.curls[i] += (st.curls_obs[i] - st.curls[i]) * 0.5;
            }
            let strength = st.strength;
            let curls = st.curls;

            // 관측 없으면 차렷 타깃으로 — rest(T-포즈)는 화면에 절대 노출하지 않는다.
            // 관측 등장/퇴장은 차렷↔타깃 블렌드가 IK 경로를 따라 흐른다.
            let t_obs = st.smoothed.unwrap_or(chain.down_target);
            let t_eff = v_lerp(chain.down_target, t_obs, strength);
            let (s, a, b) = (chain.p_upper, chain.len_upper, chain.len_lower);
            let d = v_len(v_sub(t_eff, s)).clamp((a - b).abs() + 1e-3, a + b - 1e-3);
            let dir = v_norm(v_sub(t_eff, s));
            // 어깨 내각 (law of cosines) — 팔꿈치는 아래·뒤쪽 폴 방향으로 굽힘
            let cos_sh = ((a * a + d * d - b * b) / (2.0 * a * d)).clamp(-1.0, 1.0);
            let sh = cos_sh.acos();
            let pole = [0.0, -1.0, -0.35];
            let mut axis = v_cross(dir, pole);
            if v_len(axis) < 1e-4 {
                axis = [0.0, 0.0, 1.0];
            }
            let upper_dir = quat_rotate_vec(quat_axis_angle(v_norm(axis), sh), dir);
            let elbow = v_add(s, v_scale(upper_dir, a));
            let fore_dir = v_norm(v_sub(t_eff, elbow));

            // 상완: 부모(어깨)는 안 움직이므로 rest 부모 월드가 정확하다
            let u0 = v_norm(v_sub(chain.p_lower, chain.p_upper));
            let r_u_world =
                quat_normalize(quat_mul(quat_from_to(u0, upper_dir), self.rest_world_q[chain.upper]));
            let pw_u = match self.parent[chain.upper] {
                Some(p) => self.rest_world_q[p],
                None => [0.0, 0.0, 0.0, 1.0],
            };
            self.rot_override
                .insert(chain.upper, quat_normalize(quat_mul(quat_conj(pw_u), r_u_world)));
            // 전완: 부모(상완)의 실제 월드 회전 기준으로 로컬 재투영 — 체인 정확
            let f0 = v_norm(v_sub(chain.p_hand, chain.p_lower));
            let r_l_world =
                quat_normalize(quat_mul(quat_from_to(f0, fore_dir), self.rest_world_q[chain.lower]));
            self.rot_override
                .insert(chain.lower, quat_normalize(quat_mul(quat_conj(r_u_world), r_l_world)));

            // 손가락: 로컬축 고정 컬 — rest 로컬에 굴곡 회전을 합성
            let fingers = &self.fingers[side];
            for j in fingers.iter() {
                let curl = curls[j.finger] * strength;
                if curl < 0.01 {
                    continue;
                }
                let local = quat_normalize(quat_mul(
                    self.rest[j.node].r,
                    quat_axis_angle(j.axis_local, j.max_rad * curl),
                ));
                self.rot_override.insert(j.node, local);
            }
        }
    }
}

/// node 의 첫 자식 rest 위치 — 없으면(말단) 부모 반대 방향으로 연장 근사.
fn joints_child_pos(model: &VrmModel, node: usize, rest_p: &[[f32; 3]]) -> [f32; 3] {
    for (i, nd) in model.nodes.iter().enumerate() {
        if nd.parent == Some(node) {
            return rest_p[i];
        }
    }
    match model.nodes[node].parent {
        Some(p) => v_add(rest_p[node], v_sub(rest_p[node], rest_p[p])),
        None => v_add(rest_p[node], [0.0, 0.01, 0.0]),
    }
}

/// MediaPipe 21pt(픽셀) → 손가락 5개 굴곡 0..1 [엄지, 검지, 중지, 약지, 새끼].
/// 비율 기반: |MCP→TIP| / 체인 길이 — 원근·손 크기 불변.
pub fn finger_curls_from_px(lm: &[[f32; 2]; 21]) -> [f32; 5] {
    let d = |a: usize, b: usize| -> f32 {
        let dx = lm[a][0] - lm[b][0];
        let dy = lm[a][1] - lm[b][1];
        (dx * dx + dy * dy).sqrt()
    };
    let ratio = |mcp: usize, pip: usize, dip: usize, tip: usize| -> f32 {
        let chain = (d(mcp, pip) + d(pip, dip) + d(dip, tip)).max(1e-3);
        d(mcp, tip) / chain
    };
    let to_curl = |r: f32| -> f32 { ((0.92 - r) / (0.92 - 0.45)).clamp(0.0, 1.0) };
    [
        to_curl(ratio(1, 2, 3, 4)), // 엄지 (CMC 기준)
        to_curl(ratio(5, 6, 7, 8)),
        to_curl(ratio(9, 10, 11, 12)),
        to_curl(ratio(13, 14, 15, 16)),
        to_curl(ratio(17, 18, 19, 20)),
    ]
}

// ── blink ──

/// 478pt 랜드마크 → 감김 정도 0(뜸)..1(감김). 양눈 평균.
/// EAR = (윗꺼풀-아랫꺼풀 거리) / (눈 가로폭), 왼눈 159/145·33/133,
/// 오른눈 386/374·362/263 (MediaPipe 메시 인덱스).
pub fn blink_from_landmarks(lm: &[[f32; 3]]) -> Option<f32> {
    if lm.len() < 468 {
        return None;
    }
    let d = |a: usize, b: usize| -> f32 {
        let dx = lm[a][0] - lm[b][0];
        let dy = lm[a][1] - lm[b][1];
        (dx * dx + dy * dy).sqrt()
    };
    let ear_l = d(159, 145) / d(33, 133).max(1e-3);
    let ear_r = d(386, 374) / d(362, 263).max(1e-3);
    let ear = (ear_l + ear_r) * 0.5;
    // 개인차 흡수 전의 보수적 고정 매핑: 0.28(뜸) → 0, 0.15(감김) → 1
    Some(((0.28 - ear) / (0.28 - 0.15)).clamp(0.0, 1.0))
}

pub struct BlinkCtl {
    weight: f32,
    last_obs_ms: f64,
}

impl Default for BlinkCtl {
    fn default() -> Self {
        BlinkCtl { weight: 0.0, last_obs_ms: -1e12 }
    }
}

const AUTO_BLINK_AFTER_MS: f64 = 700.0;
const BLINK_CLOSE_MS: f64 = 60.0;
const BLINK_HOLD_MS: f64 = 30.0;
const BLINK_OPEN_MS: f64 = 90.0;

impl BlinkCtl {
    /// obs = 관측 감김 정도 (얼굴 추적 프레임에만 Some). 반환 = 모프 가중 0..1.
    pub fn update(&mut self, obs: Option<f32>, t_ms: f64) -> f32 {
        if let Some(o) = obs {
            self.last_obs_ms = t_ms;
            // attack 빠르게(감는 건 즉각), release 완만하게(뜨는 건 부드럽게)
            let alpha = if o > self.weight { 0.85 } else { 0.45 };
            self.weight += (o - self.weight) * alpha;
            return self.weight;
        }
        if t_ms - self.last_obs_ms < AUTO_BLINK_AFTER_MS {
            // 관측 잠깐 끊김 — 마지막 가중 유지 (release 방향으로만 감쇠)
            self.weight *= 0.9;
            return self.weight;
        }
        // 자동 깜빡임: 주기 2.8~5.2s (사이클 인덱스 해시로 가변 — 메트로놈 방지)
        let mut t = t_ms;
        let mut cycle: u64 = 0;
        loop {
            let period = 2800.0 + ((cycle.wrapping_mul(2654435761) >> 8) % 2400) as f64;
            if t < period {
                let env_len = BLINK_CLOSE_MS + BLINK_HOLD_MS + BLINK_OPEN_MS;
                self.weight = if t < env_len {
                    if t < BLINK_CLOSE_MS {
                        (t / BLINK_CLOSE_MS) as f32
                    } else if t < BLINK_CLOSE_MS + BLINK_HOLD_MS {
                        1.0
                    } else {
                        (1.0 - (t - BLINK_CLOSE_MS - BLINK_HOLD_MS) / BLINK_OPEN_MS) as f32
                    }
                } else {
                    0.0
                };
                return self.weight;
            }
            t -= period;
            cycle += 1;
            if cycle > 1_000_000 {
                self.weight = 0.0;
                return 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quat_roundtrip_world_delta() {
        // 항등 delta 는 rest 로컬을 그대로 돌려줘야 한다
        let q = quat_normalize([0.1, 0.2, -0.05, 0.97]);
        let pw = quat_normalize([0.0, 0.3, 0.0, 0.95]);
        let world = quat_mul(pw, q);
        let local = quat_normalize(quat_mul(quat_conj(pw), world));
        for k in 0..4 {
            assert!((local[k] - q[k]).abs() < 1e-5);
        }
    }

    #[test]
    fn blink_envelope_fires() {
        let mut ctl = BlinkCtl::default();
        // 관측 없이 자동 깜빡임 — 첫 사이클 감김 구간에서 1.0 근처 도달
        let w = ctl.update(None, 70.0);
        assert!(w > 0.8, "auto blink close phase: {w}");
        let w2 = ctl.update(None, 1000.0);
        assert!(w2 < 0.05, "open phase: {w2}");
    }

    #[test]
    fn blink_obs_attack_release() {
        let mut ctl = BlinkCtl::default();
        let w = ctl.update(Some(1.0), 0.0);
        assert!(w > 0.7);
        let w2 = ctl.update(Some(0.0), 33.0);
        assert!(w2 < w * 0.7);
    }

    #[test]
    fn ear_open_closed() {
        // 합성 랜드마크: 뜬 눈 EAR 0.3 → 0, 감은 눈 EAR 0.1 → 1
        let mut lm = vec![[0.0f32; 3]; 478];
        let set = |lm: &mut Vec<[f32; 3]>, open: bool| {
            let h = if open { 30.0 } else { 10.0 };
            lm[33] = [0.0, 0.0, 0.0];
            lm[133] = [100.0, 0.0, 0.0];
            lm[159] = [50.0, 0.0, 0.0];
            lm[145] = [50.0, h, 0.0];
            lm[362] = [200.0, 0.0, 0.0];
            lm[263] = [300.0, 0.0, 0.0];
            lm[386] = [250.0, 0.0, 0.0];
            lm[374] = [250.0, h, 0.0];
        };
        set(&mut lm, true);
        assert!(blink_from_landmarks(&lm).unwrap() < 0.05);
        set(&mut lm, false);
        assert!(blink_from_landmarks(&lm).unwrap() > 0.95);
    }
}
