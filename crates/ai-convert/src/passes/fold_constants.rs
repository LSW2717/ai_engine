//! 상수 접기 잔여분 — 양쪽 상수는 resolve_static의 eval이 이미 접었다.
//! 여기서는 `Div(x, const)` → `Mul(x, 1/const)` 재작성만 남는다
//! (골든 plan의 `__const_1` = 1/std 규약과 동일).

use std::sync::Arc;

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, _ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    for idx in 0..g.nodes.len() {
        if g.nodes[idx].dead || g.nodes[idx].op != "Div" {
            continue;
        }
        let divisor_name = g.nodes[idx].inputs[1].clone();
        let Some(divisor) = g.info(&divisor_name).filter(|t| t.is_const()) else {
            return Err(ConvertError::Unsupported(vec![format!(
                "비상수 나눗셈: {}",
                g.nodes[idx].name
            )]));
        };
        let vals = divisor.as_f32s().ok_or_else(|| {
            ConvertError::Malformed(format!("Div 상수가 f32 아님: {divisor_name}"))
        })?;
        let recip: Vec<f32> = vals.iter().map(|v| 1.0 / v).collect();
        let shape = divisor.shape.clone();
        let recip_name = format!("{divisor_name}__recip");
        g.add_const(
            recip_name.clone(),
            TensorInfo {
                shape,
                dtype: OnnxDtype::F32,
                data: Some(Arc::new(bytemuck::cast_slice(&recip).to_vec())),
            },
        );
        let n = &mut g.nodes[idx];
        n.op = "Mul".into();
        n.inputs[1] = recip_name;
        report.rewrites += 1;
    }
    Ok(report)
}
