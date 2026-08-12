//! 데이터 Slice 정규화 — 풀범위→alias, 채널 슬라이스→chview(4배수)/chcopy(비정렬),
//! 공간 크롭→에러(--size가 16의 배수면 발생하지 않는다).

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Slice" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
        let o = g.info(&node.outputs[0]).unwrap().static_shape().unwrap().to_vec();

        let get_const = |i: usize| -> Option<Vec<i64>> {
            node.inputs
                .get(i)
                .and_then(|n| g.info(n))
                .and_then(|t| t.as_i64s().map(|v| v.to_vec()))
        };
        let starts = get_const(1).unwrap_or_default();
        let axes = get_const(3).unwrap_or_else(|| (0..starts.len() as i64).collect());

        // 실제로 잘리는 축
        let mut cut_axes = Vec::new();
        for (i, &ax) in axes.iter().enumerate() {
            let ax = if ax < 0 { (x.len() as i64 + ax) as usize } else { ax as usize };
            let dim = x[ax];
            let start = {
                let s = starts.get(i).copied().unwrap_or(0);
                if s < 0 { (dim + s).max(0) } else { s.min(dim) }
            };
            if o[ax] != dim || start != 0 {
                cut_axes.push((ax, start, o[ax]));
            }
        }

        if cut_axes.is_empty() {
            let (src, out) = (node.inputs[0].clone(), node.outputs[0].clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
            continue;
        }
        if cut_axes.len() == 1 && cut_axes[0].0 == 1 {
            let (_, start, n_ch) = cut_axes[0];
            let n = &mut g.nodes[idx];
            n.inputs.truncate(1);
            n.attrs.clear();
            if start % 4 == 0 {
                n.op = "chview".into();
                n.attrs.insert("cg_off".into(), Attr::I(start / 4));
                n.attrs.insert("c".into(), Attr::I(n_ch));
            } else {
                n.op = "chcopy".into();
                n.attrs.insert("src_c".into(), Attr::I(start));
                n.attrs.insert("n".into(), Attr::I(n_ch));
            }
            report.rewrites += 1;
            continue;
        }
        return Err(ConvertError::Unsupported(vec![format!(
            "공간 Slice {:?} ({}) — --size의 H,W가 16의 배수인지 확인",
            cut_axes, node.name
        )]));
    }
    Ok(report)
}
