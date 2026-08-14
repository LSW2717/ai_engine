//! 오디오 그래프 미니 실행기 — tools/prep_fastenhancer.py --export 산출물
//! (graph.json + weights.bin) 로드·실행.
//!
//! **왜 .sw가 아닌가**: fastenhancer 서브그래프는 rank-4 attention·동적
//! MatMul(양변 활성)·ConvTranspose1d 등 이미지 IR(NHWC)과 이질적인 계열이고
//! 텐서가 미소(최대 48×128)해서 NHWC 커널 기계가 과잉이다. vcx-noise가 ncnn
//! 전용 재작성으로 간 것과 같은 판단 — ai-convert 오디오 계열 일반화는 다음
//! 오디오 모델이 생기면 그때 (NEXT.md #6). 오디오는 CPU 고정(AudioWorklet에
//! WebGPU 없음)이라 이 실행기는 순수 CPU다.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::error::TaskError;
use super::ops::{self, Tens};

#[derive(Deserialize)]
struct IoDecl {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct NodeDecl {
    op: String,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    #[serde(default)]
    attrs: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct ConstF32 {
    name: String,
    shape: Vec<usize>,
    offset: usize,
    len: usize,
}

/// 전/후처리 상수 — prep 스크립트가 원본 그래프에서 발굴 (Rust 하드코딩 금지)
#[derive(Deserialize, Clone, Copy, Debug)]
pub struct PrePost {
    pub n_fft: usize,
    pub hop: usize,
    pub alpha: f32,
    pub beta: f32,
    pub clip_min: f32,
}

#[derive(Deserialize)]
struct Manifest {
    inputs: Vec<IoDecl>,
    outputs: Vec<String>,
    nodes: Vec<NodeDecl>,
    consts_f32: Vec<ConstF32>,
    consts_int: HashMap<String, Vec<i64>>,
    pre_post: PrePost,
}

pub struct FeGraph {
    nodes: Vec<NodeDecl>,
    consts: HashMap<String, Tens>,
    ints: HashMap<String, Vec<i64>>,
    /// 텐서별 소비 횟수 (그래프 출력은 +1) — reshape 계열 zero-copy 판정용
    uses: HashMap<String, usize>,
    pub inputs: Vec<(String, Vec<usize>)>,
    pub outputs: Vec<String>,
    pub pre_post: PrePost,
}

fn err(msg: String) -> TaskError {
    TaskError::Other(format!("audio-graph: {msg}"))
}

fn attr_i64(n: &NodeDecl, key: &str, default: i64) -> i64 {
    n.attrs.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
}

fn attr_f32(n: &NodeDecl, key: &str, default: f32) -> f32 {
    n.attrs.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
}

fn attr_ints(n: &NodeDecl, key: &str) -> Option<Vec<i64>> {
    n.attrs.get(key)?.as_array().map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
}

impl FeGraph {
    pub fn load(json: &[u8], weights: &[u8]) -> Result<Self, TaskError> {
        let m: Manifest =
            serde_json::from_slice(json).map_err(|e| err(format!("graph.json: {e}")))?;
        let mut consts = HashMap::new();
        for c in &m.consts_f32 {
            let bytes = weights
                .get(c.offset * 4..(c.offset + c.len) * 4)
                .ok_or_else(|| err(format!("weights.bin 범위 밖: {}", c.name)))?;
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            // 스칼라([])는 shape [1]로 정규화 (브로드캐스트가 처리)
            let shape = if c.shape.is_empty() { vec![1] } else { c.shape.clone() };
            consts.insert(c.name.clone(), Tens::new(shape, data));
        }
        let mut uses: HashMap<String, usize> = HashMap::new();
        for n in &m.nodes {
            for i in &n.inputs {
                *uses.entry(i.clone()).or_default() += 1;
            }
        }
        for o in &m.outputs {
            *uses.entry(o.clone()).or_default() += 1;
        }
        Ok(FeGraph {
            nodes: m.nodes,
            consts,
            ints: m.consts_int,
            uses,
            inputs: m.inputs.into_iter().map(|i| (i.name, i.shape)).collect(),
            outputs: m.outputs,
            pre_post: m.pre_post,
        })
    }

    fn get<'a>(
        &'a self,
        env: &'a HashMap<String, Tens>,
        name: &str,
    ) -> Result<&'a Tens, TaskError> {
        env.get(name)
            .or_else(|| self.consts.get(name))
            .ok_or_else(|| err(format!("텐서 없음: {name}")))
    }

    fn get_ints(&self, name: &str) -> Result<&[i64], TaskError> {
        self.ints
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| err(format!("int 상수 없음: {name}")))
    }

    /// 그래프 1회 실행 — inputs는 self.inputs 순서
    pub fn run(&self, inputs: Vec<Tens>) -> Result<Vec<Tens>, TaskError> {
        let mut env: HashMap<String, Tens> = HashMap::with_capacity(self.nodes.len() * 2);
        for ((name, shape), t) in self.inputs.iter().zip(inputs) {
            if t.numel() != shape.iter().product::<usize>().max(1) {
                return Err(err(format!("입력 {name} 크기 불일치 {:?}≠{shape:?}", t.shape)));
            }
            env.insert(name.clone(), Tens::new(shape.clone(), t.data));
        }
        let mut remain = self.uses.clone();
        for n in &self.nodes {
            self.exec(n, &mut env, &mut remain)?;
        }
        self.outputs
            .iter()
            .map(|o| env.remove(o).ok_or_else(|| err(format!("출력 없음: {o}"))))
            .collect()
    }

    /// run + op별 누적 시간(ms) — 벤치 진단용
    pub fn run_profiled(
        &self,
        inputs: Vec<Tens>,
    ) -> Result<(Vec<Tens>, Vec<(String, f64)>), TaskError> {
        let mut env: HashMap<String, Tens> = HashMap::with_capacity(self.nodes.len() * 2);
        for ((name, shape), t) in self.inputs.iter().zip(inputs) {
            env.insert(name.clone(), Tens::new(shape.clone(), t.data));
        }
        let mut per_op: HashMap<String, f64> = HashMap::new();
        let mut remain = self.uses.clone();
        for n in &self.nodes {
            let t0 = Instant::now();
            self.exec(n, &mut env, &mut remain)?;
            *per_op.entry(n.op.clone()).or_default() += t0.elapsed().as_secs_f64() * 1e3;
        }
        let mut v: Vec<(String, f64)> = per_op.into_iter().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let outs = self
            .outputs
            .iter()
            .map(|o| env.remove(o).unwrap())
            .collect();
        Ok((outs, v))
    }

    /// 입력 텐서를 zero-copy로 가져온다 — env에 있고 이번이 마지막 소비면 move,
    /// 아니면 clone (상수는 항상 clone 아님 — 호출부가 참조 경로 사용)
    fn take_or_clone(
        &self,
        env: &mut HashMap<String, Tens>,
        remain: &HashMap<String, usize>,
        name: &str,
    ) -> Result<Tens, TaskError> {
        if remain.get(name) == Some(&1) {
            if let Some(t) = env.remove(name) {
                return Ok(t);
            }
        }
        Ok(self.get(env, name)?.clone())
    }

    fn exec(
        &self,
        n: &NodeDecl,
        env: &mut HashMap<String, Tens>,
        remain: &mut HashMap<String, usize>,
    ) -> Result<(), TaskError> {
        let out = match n.op.as_str() {
            "Reshape" => {
                let target = self.get_ints(&n.inputs[1])?.to_vec();
                let a = self.take_or_clone(env, remain, &n.inputs[0])?;
                let shape = ops::resolve_reshape(&a.shape, &target)
                    .map_err(|e| err(format!("{} (in={})", e, n.inputs[0])))?;
                vec![Tens::new(shape, a.data)]
            }
            "Squeeze" | "Unsqueeze" => {
                let axes: Vec<i64> = if n.inputs.len() > 1 {
                    self.get_ints(&n.inputs[1])?.to_vec()
                } else {
                    attr_ints(n, "axes").unwrap_or_default()
                };
                let a = self.take_or_clone(env, remain, &n.inputs[0])?;
                let mut shape = a.shape.clone();
                if n.op == "Squeeze" {
                    let r = shape.len() as i64;
                    let mut drop: Vec<usize> = axes
                        .iter()
                        .map(|&x| if x < 0 { (x + r) as usize } else { x as usize })
                        .collect();
                    drop.sort_unstable();
                    for &d in drop.iter().rev() {
                        if shape[d] != 1 {
                            return Err(err(format!("{}: squeeze 축 {d}≠1", n.name)));
                        }
                        shape.remove(d);
                    }
                } else {
                    let r = (shape.len() + axes.len()) as i64;
                    let mut ins: Vec<usize> = axes
                        .iter()
                        .map(|&x| if x < 0 { (x + r) as usize } else { x as usize })
                        .collect();
                    ins.sort_unstable();
                    for &d in &ins {
                        shape.insert(d, 1);
                    }
                }
                vec![Tens::new(shape, a.data)]
            }
            "Transpose" => {
                let a = self.get(env, &n.inputs[0])?;
                let perm: Vec<usize> = attr_ints(n, "perm")
                    .ok_or_else(|| err(format!("{}: perm 없음", n.name)))?
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                vec![ops::transpose(a, &perm)]
            }
            "Mul" | "Add" | "Sub" => {
                let a = self.get(env, &n.inputs[0])?;
                let b = self.get(env, &n.inputs[1])?;
                vec![ops::ew_binary(&n.op, a, b)?]
            }
            "Sigmoid" | "Tanh" => vec![ops::unary(&n.op, self.get(env, &n.inputs[0])?)?],
            "Gemm" => {
                let a = self.get(env, &n.inputs[0])?;
                let b = self.get(env, &n.inputs[1])?;
                let c = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    Some(self.get(env, &n.inputs[2])?)
                } else {
                    None
                };
                vec![ops::gemm(
                    a,
                    b,
                    c,
                    attr_f32(n, "alpha", 1.0),
                    attr_f32(n, "beta", 1.0),
                    attr_i64(n, "transA", 0) != 0,
                    attr_i64(n, "transB", 0) != 0,
                )?]
            }
            "MatMul" => {
                let a = self.get(env, &n.inputs[0])?;
                let b = self.get(env, &n.inputs[1])?;
                vec![ops::matmul(a, b)?]
            }
            "Conv" => {
                let x = self.get(env, &n.inputs[0])?;
                let w = self.get(env, &n.inputs[1])?;
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    Some(self.get(env, &n.inputs[2])?)
                } else {
                    None
                };
                if attr_i64(n, "group", 1) != 1 {
                    return Err(err(format!("{}: group>1 미지원", n.name)));
                }
                let pads = attr_ints(n, "pads").unwrap_or_else(|| vec![0, 0]);
                let stride = attr_ints(n, "strides").map(|v| v[0]).unwrap_or(1) as usize;
                let dil = attr_ints(n, "dilations").map(|v| v[0]).unwrap_or(1) as usize;
                vec![ops::conv1d(x, w, b, (pads[0] as usize, pads[1] as usize), stride, dil)?]
            }
            "ConvTranspose" => {
                let x = self.get(env, &n.inputs[0])?;
                let w = self.get(env, &n.inputs[1])?;
                let b = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    Some(self.get(env, &n.inputs[2])?)
                } else {
                    None
                };
                let pads = attr_ints(n, "pads").unwrap_or_else(|| vec![0, 0]);
                let stride = attr_ints(n, "strides").map(|v| v[0]).unwrap_or(1) as usize;
                let op = attr_ints(n, "output_padding").map(|v| v[0]).unwrap_or(0) as usize;
                vec![ops::conv_transpose1d(
                    x,
                    w,
                    b,
                    (pads[0] as usize, pads[1] as usize),
                    stride,
                    op,
                )?]
            }
            "Softmax" => {
                vec![ops::softmax(self.get(env, &n.inputs[0])?, attr_i64(n, "axis", -1) as isize)?]
            }
            "Slice" => {
                let a = self.get(env, &n.inputs[0])?;
                let starts = self.get_ints(&n.inputs[1])?.to_vec();
                let ends = self.get_ints(&n.inputs[2])?.to_vec();
                let axes = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                    Some(self.get_ints(&n.inputs[3])?.to_vec())
                } else {
                    None
                };
                let steps = if n.inputs.len() > 4 && !n.inputs[4].is_empty() {
                    Some(self.get_ints(&n.inputs[4])?.to_vec())
                } else {
                    None
                };
                vec![ops::slice(a, &starts, &ends, axes.as_deref(), steps.as_deref())?]
            }
            "Split" => {
                let a = self.get(env, &n.inputs[0])?;
                let r = a.shape.len() as i64;
                let axis = attr_i64(n, "axis", 0);
                let axis = if axis < 0 { (axis + r) as usize } else { axis as usize };
                let sizes: Vec<usize> = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                    self.get_ints(&n.inputs[1])?.iter().map(|&v| v as usize).collect()
                } else if let Some(v) = attr_ints(n, "split") {
                    v.iter().map(|&x| x as usize).collect()
                } else {
                    let parts = n.outputs.len();
                    vec![a.shape[axis] / parts; parts]
                };
                let mut outs = Vec::with_capacity(sizes.len());
                let mut start = 0i64;
                for &sz in &sizes {
                    outs.push(ops::slice(
                        a,
                        &[start],
                        &[start + sz as i64],
                        Some(&[axis as i64]),
                        None,
                    )?);
                    start += sz as i64;
                }
                outs
            }
            "Concat" => {
                let ts: Vec<&Tens> =
                    n.inputs.iter().map(|i| self.get(env, i)).collect::<Result<_, _>>()?;
                vec![ops::concat(&ts, attr_i64(n, "axis", 0) as isize)?]
            }
            "Pad" => {
                let a = self.get(env, &n.inputs[0])?;
                let pads = self.get_ints(&n.inputs[1])?.to_vec();
                let value = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    self.get(env, &n.inputs[2])?.data[0]
                } else {
                    0.0
                };
                vec![ops::pad_constant(a, &pads, value)?]
            }
            other => return Err(err(format!("{}: 미지원 op {other}", n.name))),
        };
        for (name, t) in n.outputs.iter().zip(out) {
            env.insert(name.clone(), t);
        }
        for i in &n.inputs {
            if let Some(u) = remain.get_mut(i) {
                *u = u.saturating_sub(1);
            }
        }
        Ok(())
    }
}
