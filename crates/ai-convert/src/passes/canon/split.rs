//! Split 전개 — 채널 축 Split을 출력별 chview(4배수 오프셋)/chcopy(비정렬)로.
//! (split_concat 상쇄 이후에 실행)

use crate::error::ConvertError;
use crate::ir::{Attr, Graph, Node};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    let mut idx = 0;
    while idx < g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Split" {
            idx += 1;
            continue;
        }
        let node = g.nodes[idx].clone();
        let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
        let axis = node.attr_i("axis").unwrap_or(0);
        let ax = if axis < 0 { (x.len() as i64 + axis) as usize } else { axis as usize };
        if ax != 1 {
            return Err(ConvertError::Unsupported(vec![format!(
                "채널 축이 아닌 Split (axis={ax}, {})",
                node.name
            )]));
        }

        // 출력 shape에서 파트 크기 추출
        let mut new_nodes = Vec::new();
        let mut off = 0i64;
        for out in &node.outputs {
            let c = g.info(out).unwrap().static_shape().unwrap()[1];
            let mut attrs = std::collections::HashMap::new();
            let op = if off % 4 == 0 {
                attrs.insert("cg_off".into(), Attr::I(off / 4));
                attrs.insert("c".into(), Attr::I(c));
                "chview"
            } else {
                attrs.insert("src_c".into(), Attr::I(off));
                attrs.insert("n".into(), Attr::I(c));
                "chcopy"
            };
            new_nodes.push(Node {
                op: op.into(),
                name: format!("{}#{}", node.name, out),
                attrs,
                inputs: vec![node.inputs[0].clone()],
                outputs: vec![out.clone()],
                dead: false,
            });
            off += c;
        }

        // 원 노드를 첫 파트로 교체, 나머지는 바로 뒤에 삽입 (topo 순서 유지)
        g.nodes[idx] = new_nodes.remove(0);
        for (k, n) in new_nodes.into_iter().enumerate() {
            g.nodes.insert(idx + 1 + k, n);
        }
        report.rewrites += 1;
        idx += 1;
    }
    Ok(report)
}
