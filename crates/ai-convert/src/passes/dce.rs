//! 죽은 노드 제거 — 그래프 출력(+상태 출력)에서 역방향 마킹.
//! alias_of 테이블을 통과해 백킹 텐서의 생산자까지 추적한다.

use std::collections::HashSet;

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut live_tensors: Vec<String> = g
        .outputs
        .iter()
        .map(|o| g.resolve_alias(o).to_string())
        .collect();
    let mut seen: HashSet<String> = live_tensors.iter().cloned().collect();
    let mut live_nodes: HashSet<usize> = HashSet::new();

    while let Some(t) = live_tensors.pop() {
        if let Some(p) = g.producer(&t) {
            if live_nodes.insert(p) {
                let mut reads: Vec<String> = g.nodes[p].inputs.clone();
                // res 에필로그(fuse_residual)는 inputs가 아니라 attr로 읽는다 —
                // 추적 안 하면 res 유일 소비자인 생산자가 여기서 죽는다
                // (facelm의 maxpool 전멸로 발견).
                if let Some(r) = g.nodes[p].attr_s("res") {
                    reads.push(r.to_string());
                }
                for i in reads {
                    let r = g.resolve_alias(&i).to_string();
                    if seen.insert(r.clone()) {
                        live_tensors.push(r);
                    }
                }
            }
        }
    }

    let mut report = PassReport::default();
    for (idx, n) in g.nodes.iter_mut().enumerate() {
        if !n.dead && !live_nodes.contains(&idx) {
            n.dead = true;
            report.rewrites += 1;
        }
    }
    Ok(report)
}
