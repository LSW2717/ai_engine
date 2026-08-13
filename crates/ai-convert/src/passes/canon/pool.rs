//! 풀링 정규화 — GlobalAveragePool→gpool, AveragePool→avgpool, MaxPool→maxpool
//! (ceil_mode 무해성 검증 공통).

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead {
            continue;
        }
        match g.nodes[idx].op.as_str() {
            "GlobalAveragePool" => {
                let n = &mut g.nodes[idx];
                n.op = "gpool".into();
                n.attrs.clear();
                report.rewrites += 1;
            }
            "AveragePool" => {
                let node = g.nodes[idx].clone();
                let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
                let k = node.attr_is("kernel_shape").unwrap().to_vec();
                let s = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
                let p = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0; 4]);
                if node.attr_i("ceil_mode").unwrap_or(0) == 1 {
                    let exact = (x[2] + p[0] + p[2] - k[0]) % s[0] == 0
                        && (x[3] + p[1] + p[3] - k[1]) % s[1] == 0;
                    if !exact {
                        return Err(ConvertError::Unsupported(vec![format!(
                            "AveragePool ceil_mode 유효 ({}) — 크기가 나눠떨어지지 않음",
                            node.name
                        )]));
                    }
                }
                // count_include_pad=0 + pad>0 은 커널 분모(k*k 고정)와 의미가 다름
                if p.iter().any(|v| *v != 0) && node.attr_i("count_include_pad").unwrap_or(0) == 0 {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "AveragePool pad>0 + count_include_pad=0 ({})",
                        node.name
                    )]));
                }
                let n = &mut g.nodes[idx];
                n.op = "avgpool".into();
                n.attrs.remove("ceil_mode");
                report.rewrites += 1;
            }
            "MaxPool" => {
                let node = g.nodes[idx].clone();
                let x = g.info(&node.inputs[0]).unwrap().static_shape().unwrap().to_vec();
                let k = node.attr_is("kernel_shape").unwrap().to_vec();
                let s = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
                let p = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0; 4]);
                if node.attr_i("ceil_mode").unwrap_or(0) == 1 {
                    let exact = (x[2] + p[0] + p[2] - k[0]) % s[0] == 0
                        && (x[3] + p[1] + p[3] - k[1]) % s[1] == 0;
                    if !exact {
                        return Err(ConvertError::Unsupported(vec![format!(
                            "MaxPool ceil_mode 유효 ({}) — 크기가 나눠떨어지지 않음",
                            node.name
                        )]));
                    }
                }
                let n = &mut g.nodes[idx];
                n.op = "maxpool".into();
                n.attrs.remove("ceil_mode");
                report.rewrites += 1;
            }
            _ => {}
        }
    }
    Ok(report)
}
