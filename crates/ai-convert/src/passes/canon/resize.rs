//! Resize 정규화 — scales/sizes 입력 양식(opset 12↔18)을 정적 (oh, ow) attr로.
//! 항등 크기는 alias. mode=linear만, pytorch_half_pixel은 out>1 조건에서 half_pixel과 동치.

use crate::error::ConvertError;
use crate::ir::{Attr, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Resize" {
            continue;
        }
        let node = g.nodes[idx].clone();
        let mode = node.attr_s("mode").unwrap_or("nearest");
        if mode != "linear" {
            return Err(ConvertError::Unsupported(vec![format!(
                "Resize mode={mode} ({})",
                node.name
            )]));
        }
        if node.attr_i("antialias").unwrap_or(0) == 1 {
            return Err(ConvertError::Unsupported(vec![format!(
                "Resize antialias ({})",
                node.name
            )]));
        }
        let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
        let o = g.info(&node.outputs[0]).unwrap().static_shape().unwrap().to_vec();
        let (ih, iw, oh, ow) = (x[2], x[3], o[2], o[3]);

        if (ih, iw) == (oh, ow) {
            let (src, out) = (node.inputs[0].clone(), node.outputs[0].clone());
            g.make_alias(idx, &src, &out);
            report.rewrites += 1;
            continue;
        }

        let ctm = node
            .attr_s("coordinate_transformation_mode")
            .unwrap_or("half_pixel");
        let mode_s = match ctm {
            "half_pixel" => "half_pixel",
            "pytorch_half_pixel" => {
                if oh <= 1 || ow <= 1 {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "pytorch_half_pixel + 출력 1픽셀 축 ({})",
                        node.name
                    )]));
                }
                "half_pixel"
            }
            "asymmetric" => "asymmetric",
            other => {
                return Err(ConvertError::Unsupported(vec![format!(
                    "Resize ctm={other} ({})",
                    node.name
                )]))
            }
        };

        let n = &mut g.nodes[idx];
        n.op = "resize".into();
        n.inputs.truncate(1);
        n.attrs.clear();
        n.attrs.insert("oh".into(), Attr::I(oh));
        n.attrs.insert("ow".into(), Attr::I(ow));
        n.attrs.insert("mode".into(), Attr::S(mode_s.into()));
        report.rewrites += 1;
    }
    Ok(report)
}
