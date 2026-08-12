//! 입력 바인딩 + 정적 해석 fixpoint.
//!
//! {상수 평가(eval) + shape 추론(shape)}을 변화가 없을 때까지 반복한다.
//! `Expand(상태입력, const_shape)`가 상태 shape을 확정하는 유일한 지점이며
//! (RVM의 심볼릭 rNi가 여기서 해소), 해소 후 Expand는 alias가 된다.
//! 종료 시 모든 살아있는 텐서는 완전 정적이어야 한다.

use crate::error::ConvertError;
use crate::ir::tensor_info::{OnnxDtype, TensorInfo};
use crate::ir::{eval, shape, Graph};
use crate::passes::{Ctx, PassReport};

pub fn run(g: &mut Graph, ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut report = PassReport::default();
    bind_inputs(g, ctx, &mut report)?;

    // fixpoint
    loop {
        let mut changed = false;

        for idx in 0..g.nodes.len() {
            if g.nodes[idx].dead {
                continue;
            }
            let node = g.nodes[idx].clone();

            // 1) 상수 평가 (성공 시 노드는 상수로 대체되어 죽는다)
            if let Some(outs) = eval::try_eval(g, &node) {
                for (name, info) in node.outputs.iter().zip(outs) {
                    g.add_const(name.clone(), info);
                }
                g.nodes[idx].dead = true;
                changed = true;
                report.rewrites += 1;
                continue;
            }

            // 2) Expand(상태입력, const_shape) → 상태 shape 확정 + 항등화
            if node.op == "Expand" && g.inputs.iter().any(|i| *i == node.inputs[0]) {
                let target = node
                    .inputs
                    .get(1)
                    .and_then(|n| g.info(n))
                    .and_then(|t| t.as_i64s().map(|v| v.to_vec()));
                if let Some(target) = target {
                    g.info_mut(&node.inputs[0]).shape = Some(target.clone());
                    g.info_mut(&node.outputs[0]).shape = Some(target);
                    let (src, out) = (node.inputs[0].clone(), node.outputs[0].clone());
                    g.make_alias(idx, &src, &out);
                    changed = true;
                    report.rewrites += 1;
                    continue;
                }
            }

            // 3) shape 추론
            if let Some(shapes) = shape::infer(g, &node)? {
                for (name, s) in node.outputs.iter().zip(shapes) {
                    let info = g.info_mut(name);
                    if info.static_shape().map(|e| e.to_vec()) != Some(s.clone()) {
                        info.shape = Some(s);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // 검산: 살아있는 노드의 모든 입출력이 정적인가
    for (_, n) in g.live_nodes() {
        for name in n.inputs.iter().chain(&n.outputs) {
            let ok = g
                .info(name)
                .map(|t| t.static_shape().is_some() || t.is_const())
                .unwrap_or(false);
            if !ok {
                return Err(ConvertError::ShapeUnresolved(format!(
                    "{name} (op {} {})",
                    n.op, n.name
                )));
            }
        }
    }
    Ok(report)
}

fn bind_inputs(g: &mut Graph, ctx: &Ctx, report: &mut PassReport) -> Result<(), ConvertError> {
    // --set-input: 스칼라 상수로 고정
    for (name, v) in &ctx.set_inputs {
        if !g.inputs.iter().any(|i| i == name) {
            return Err(ConvertError::Other(format!("--set-input 대상 없음: {name}")));
        }
        let declared = g.info(name).and_then(|t| t.shape.clone()).unwrap_or(vec![1]);
        let elems: i64 = declared.iter().map(|d| d.max(&1)).product();
        g.add_const(
            name.clone(),
            TensorInfo {
                shape: Some(declared),
                dtype: OnnxDtype::F32,
                data: Some(std::sync::Arc::new(
                    std::iter::repeat_n(*v, elems as usize)
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                )),
            },
        );
        g.inputs.retain(|i| i != name);
        report.rewrites += 1;
    }

    // --state: 그래프에 상태 쌍 기록 (shape은 Expand가 확정)
    for (i, o) in &ctx.states {
        if !g.inputs.iter().any(|x| x == i) {
            return Err(ConvertError::Other(format!("--state 입력 없음: {i}")));
        }
        if !g.outputs.iter().any(|x| x == o) {
            return Err(ConvertError::Other(format!("--state 출력 없음: {o}")));
        }
        g.states.push((i.clone(), o.clone()));
    }

    // 이미지 입력: 4-D 비상태 입력. --size 적용 (NCHW C=3 규약; 이미 정적이면 검증만)
    let state_names: Vec<String> = ctx.states.iter().map(|(i, _)| i.clone()).collect();
    let image_inputs: Vec<String> = g
        .inputs
        .iter()
        .filter(|i| !state_names.contains(i))
        .cloned()
        .collect();
    for name in &image_inputs {
        let info = g.info_mut(name);
        let declared = info.shape.clone().unwrap_or_default();
        if let Some((w, h)) = ctx.size {
            if declared.len() == 4 {
                let already = info.static_shape().map(|s| s.to_vec());
                let nchw = vec![1, 3, h as i64, w as i64];
                let nhwc = vec![1, h as i64, w as i64, 3];
                match already {
                    Some(s) if s == nchw || s == nhwc => {}
                    Some(s) => {
                        return Err(ConvertError::Other(format!(
                            "{name}의 정적 shape {s:?}가 --size {w}x{h}와 불일치"
                        )))
                    }
                    None => {
                        // 심볼릭 → NCHW로 고정 (NHWC 선언 모델은 이미 정적이므로 여기 안 옴)
                        info.shape = Some(nchw);
                        report.rewrites += 1;
                    }
                }
            }
        } else if info.static_shape().is_none() {
            return Err(ConvertError::ShapeUnresolved(format!(
                "{name} — --size로 입력 크기를 지정하세요"
            )));
        }
    }

    // 남은 심볼릭 입력 검사 (상태는 Expand가 해소 예정이므로 제외)
    for name in g.inputs.clone() {
        if state_names.contains(&name) {
            continue;
        }
        if g.info(&name).and_then(|t| t.static_shape()).is_none() {
            return Err(ConvertError::ShapeUnresolved(format!(
                "입력 {name} — --size 또는 --set-input으로 고정 필요"
            )));
        }
    }
    Ok(())
}
