//! 변환기 내부 그래프 IR — 패스들이 변형하는 유일한 자료구조.
//!
//! prost 타입에는 패스를 돌리지 않는다(import에서 이 IR로 옮긴 뒤 시작).
//! 노드 ~350개 규모라 producer/consumers는 필요 시 선형 탐색으로 충분하다.

use std::collections::{BTreeMap, HashMap};

use crate::ir::tensor_info::TensorInfo;

#[derive(Clone, Debug)]
pub enum Attr {
    I(i64),
    Is(Vec<i64>),
    F(f32),
    Fs(Vec<f32>),
    S(String),
    T(TensorInfo),
}

#[derive(Clone, Debug)]
pub struct Node {
    /// ONNX op_type (canonicalize 후엔 내부 op명도 사용: gpool, hswish 등)
    pub op: String,
    pub name: String,
    pub attrs: HashMap<String, Attr>,
    /// 생략된 옵션 입력("")은 import에서 제거됨
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub dead: bool,
}

impl Node {
    pub fn attr_i(&self, k: &str) -> Option<i64> {
        match self.attrs.get(k) {
            Some(Attr::I(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_is(&self, k: &str) -> Option<&[i64]> {
        match self.attrs.get(k) {
            Some(Attr::Is(v)) => Some(v),
            _ => None,
        }
    }

    pub fn attr_f(&self, k: &str) -> Option<f32> {
        match self.attrs.get(k) {
            Some(Attr::F(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_s(&self, k: &str) -> Option<&str> {
        match self.attrs.get(k) {
            Some(Attr::S(v)) => Some(v),
            _ => None,
        }
    }
}

#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors: HashMap<String, TensorInfo>,
    /// 그래프 입력 이름 (이니셜라이저 제외)
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// (입력명, 출력명) 순환 상태 쌍 — CLI --state로 지정
    pub states: Vec<(String, String)>,
    /// alias 마킹: 소비자 재배선 후 "이 이름은 저 이름의 별칭" 기록 (출력 이름 보존용)
    pub alias_of: HashMap<String, String>,
    /// 호출자 데이터가 NHWC 논리 순서인 입력/출력 (경계 Transpose에서 마킹 — 물리
    /// 레이아웃은 어차피 NHWC-C4라 로더가 논리 순서만 알면 된다)
    pub nhwc_inputs: Vec<String>,
    pub nhwc_outputs: Vec<String>,
    /// 융합으로 값의 의미가 원본 ONNX와 달라진 텐서 이름 (act/res가 흡수된 conv 출력 등)
    /// — 오라클 대조에서 제외해야 한다
    pub semantic_changed: Vec<String>,
}

impl Graph {
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn info_mut(&mut self, name: &str) -> &mut TensorInfo {
        self.tensors.entry(name.to_string()).or_default()
    }

    pub fn is_const(&self, name: &str) -> bool {
        self.tensors.get(name).is_some_and(|t| t.is_const())
    }

    /// name을 출력하는 살아있는 노드 인덱스
    pub fn producer(&self, name: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| !n.dead && n.outputs.iter().any(|o| o == name))
    }

    /// name을 입력으로 읽는 살아있는 노드 인덱스들
    pub fn consumers(&self, name: &str) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.dead && n.inputs.iter().any(|i| i == name))
            .map(|(i, _)| i)
            .collect()
    }

    /// 그래프 출력 여부
    pub fn is_output(&self, name: &str) -> bool {
        self.outputs.iter().any(|o| o == name)
    }

    /// from을 읽는 모든 노드 입력을 to로 재배선 (+ 그래프 출력 목록도)
    pub fn replace_uses(&mut self, from: &str, to: &str) {
        for n in &mut self.nodes {
            if n.dead {
                continue;
            }
            for i in &mut n.inputs {
                if i == from {
                    *i = to.to_string();
                }
            }
        }
        for o in &mut self.outputs {
            if o == from {
                *o = to.to_string();
            }
        }
        for (_, out) in &mut self.states {
            if out == from {
                *out = to.to_string();
            }
        }
    }

    /// 노드를 alias로 대체: out의 사용처를 src로 재배선하고 노드 kill.
    /// 그래프 출력 이름은 보존해야 하므로 alias_of에 기록만 한다.
    pub fn make_alias(&mut self, node_idx: usize, src: &str, out: &str) {
        let (src, out) = (src.to_string(), out.to_string());
        if self.is_output(&out) || self.states.iter().any(|(_, o)| *o == out) {
            // 출력 이름 유지 — 재배선 대신 별칭 테이블에 기록
            self.alias_of.insert(out.clone(), src.clone());
        } else {
            self.replace_uses(&out, &src);
        }
        self.nodes[node_idx].dead = true;
    }

    pub fn add_const(&mut self, name: impl Into<String>, info: TensorInfo) {
        debug_assert!(info.is_const());
        self.tensors.insert(name.into(), info);
    }

    /// 살아있는 노드 op 히스토그램
    pub fn op_histogram(&self) -> BTreeMap<String, usize> {
        let mut h = BTreeMap::new();
        for n in self.nodes.iter().filter(|n| !n.dead) {
            *h.entry(n.op.clone()).or_insert(0) += 1;
        }
        h
    }

    /// alias 체인 해석 (alias_of 테이블 경유)
    pub fn resolve_alias<'a>(&'a self, mut name: &'a str) -> &'a str {
        while let Some(next) = self.alias_of.get(name) {
            name = next;
        }
        name
    }

    pub fn live_nodes(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes.iter().enumerate().filter(|(_, n)| !n.dead)
    }
}
