//! op별 shape 추론 — --size로 입력이 고정되면 전 그래프가 정적이 된다는 전제.
//! NCHW 의미론 (lowering 전까지 유지).

use crate::error::ConvertError;
use crate::ir::graph::{Graph, Node};

fn shape_of(g: &Graph, name: &str) -> Option<Vec<i64>> {
    g.info(name)?.static_shape().map(|s| s.to_vec())
}

/// numpy 브로드캐스트 (rank ≤ 4)
fn broadcast(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for i in 0..n {
        let av = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let bv = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        out[i] = if av == bv || bv == 1 {
            av
        } else if av == 1 {
            bv
        } else {
            return None;
        };
    }
    Some(out)
}

fn conv_like_out(i: i64, k: i64, s: i64, p0: i64, p1: i64, d: i64, ceil: bool) -> i64 {
    let eff_k = d * (k - 1) + 1;
    let num = i + p0 + p1 - eff_k;
    if ceil {
        (num + s - 1) / s + 1
    } else {
        num / s + 1
    }
}

/// 노드 출력 shape 추론. 아직 판단 불가면 Ok(None) — fixpoint가 재시도한다.
/// 구조적으로 지원 불가면 Err.
pub fn infer(g: &Graph, node: &Node) -> Result<Option<Vec<Vec<i64>>>, ConvertError> {
    let one = |s: Vec<i64>| Ok(Some(vec![s]));
    match node.op.as_str() {
        // 단항 (elementwise)
        "Relu" | "Sigmoid" | "Tanh" | "HardSigmoid" | "Clip" | "Identity" | "Erf"
        | "hswish" => match shape_of(g, &node.inputs[0]) {
            Some(s) => one(s),
            None => Ok(None),
        },
        "Mul" | "Add" | "Sub" | "Div" => {
            let (Some(a), Some(b)) =
                (shape_of(g, &node.inputs[0]), shape_of(g, &node.inputs[1]))
            else {
                return Ok(None);
            };
            match broadcast(&a, &b) {
                Some(s) => one(s),
                None => Err(ConvertError::Malformed(format!(
                    "브로드캐스트 불가 {:?} vs {:?} ({})",
                    a, b, node.name
                ))),
            }
        }
        "Conv" => {
            let (Some(x), Some(w)) =
                (shape_of(g, &node.inputs[0]), shape_of(g, &node.inputs[1]))
            else {
                return Ok(None);
            };
            if let Some(ap) = node.attr_s("auto_pad") {
                if ap != "NOTSET" {
                    return Err(ConvertError::Unsupported(vec![format!(
                        "auto_pad={ap} ({})",
                        node.name
                    )]));
                }
            }
            let k = node
                .attr_is("kernel_shape")
                .map(|v| v.to_vec())
                .unwrap_or_else(|| w[2..].to_vec());
            let s = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
            let d = node.attr_is("dilations").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
            let p = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0, 0, 0, 0]);
            let oh = conv_like_out(x[2], k[0], s[0], p[0], p[2], d[0], false);
            let ow = conv_like_out(x[3], k[1], s[1], p[1], p[3], d[1], false);
            one(vec![x[0], w[0], oh, ow])
        }
        "AveragePool" | "MaxPool" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            let k = node.attr_is("kernel_shape").ok_or_else(|| {
                ConvertError::Malformed(format!("kernel_shape 없음 ({})", node.name))
            })?;
            let s = node.attr_is("strides").map(|v| v.to_vec()).unwrap_or(vec![1, 1]);
            let p = node.attr_is("pads").map(|v| v.to_vec()).unwrap_or(vec![0, 0, 0, 0]);
            let ceil = node.attr_i("ceil_mode").unwrap_or(0) == 1;
            let oh = conv_like_out(x[2], k[0], s[0], p[0], p[2], 1, ceil);
            let ow = conv_like_out(x[3], k[1], s[1], p[1], p[3], 1, ceil);
            one(vec![x[0], x[1], oh, ow])
        }
        "GlobalAveragePool" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            one(vec![x[0], x[1], 1, 1])
        }
        "Resize" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            // sizes(입력 4) 우선, 없으면 scales(입력 3 — import에서 빈 문자열 제거됨에 유의)
            // 입력 배열: [X, roi?, scales?, sizes?] — 빈 입력 제거 후 개수로 판별 불가한
            // 경우가 있어 뒤에서부터 상수를 찾는다.
            for cand in node.inputs.iter().skip(1).rev() {
                let Some(info) = g.info(cand) else { continue };
                if !info.is_const() {
                    continue;
                }
                if let Some(sizes) = info.as_i64s() {
                    if sizes.len() == 4 {
                        return one(sizes.to_vec());
                    }
                }
                if let Some(scales) = info.as_f32s() {
                    if scales.len() == 4 {
                        let oh = (x[2] as f64 * scales[2] as f64).floor() as i64;
                        let ow = (x[3] as f64 * scales[3] as f64).floor() as i64;
                        return one(vec![x[0], x[1], oh, ow]);
                    }
                }
            }
            Ok(None) // scales/sizes 아직 미상수 — fixpoint 재시도
        }
        "Concat" => {
            let axis = node.attr_i("axis").unwrap_or(0);
            let mut shapes = Vec::new();
            for i in &node.inputs {
                match shape_of(g, i) {
                    Some(s) => shapes.push(s),
                    None => return Ok(None),
                }
            }
            let rank = shapes[0].len() as i64;
            let ax = if axis < 0 { (rank + axis) as usize } else { axis as usize };
            let mut out = shapes[0].clone();
            out[ax] = shapes.iter().map(|s| s[ax]).sum();
            one(out)
        }
        "Split" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            let axis = node.attr_i("axis").unwrap_or(0);
            let rank = x.len() as i64;
            let ax = if axis < 0 { (rank + axis) as usize } else { axis as usize };
            let parts: Vec<i64> = if let Some(sp) = node.attr_is("split") {
                sp.to_vec()
            } else if let Some(second) = node.inputs.get(1) {
                match g.info(second).and_then(|t| t.as_i64s().map(|v| v.to_vec())) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            } else {
                let n = node.outputs.len() as i64;
                if x[ax] % n != 0 {
                    return Err(ConvertError::Malformed(format!(
                        "균등 분할 불가 ({})",
                        node.name
                    )));
                }
                vec![x[ax] / n; node.outputs.len()]
            };
            let mut outs = Vec::new();
            for p in parts {
                let mut s = x.clone();
                s[ax] = p;
                outs.push(s);
            }
            Ok(Some(outs))
        }
        "Slice" => {
            // 데이터 슬라이스 (배관용 1-D는 eval이 먼저 처리)
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            let get_const = |idx: usize| -> Option<Vec<i64>> {
                node.inputs
                    .get(idx)
                    .and_then(|n| g.info(n))
                    .and_then(|t| t.as_i64s().map(|v| v.to_vec()))
            };
            let (Some(starts), Some(ends)) = (get_const(1), get_const(2)) else {
                return Ok(None);
            };
            let axes = get_const(3)
                .unwrap_or_else(|| (0..starts.len() as i64).collect());
            let steps = get_const(4).unwrap_or_else(|| vec![1; starts.len()]);
            if steps.iter().any(|s| *s != 1) {
                return Err(ConvertError::Unsupported(vec![format!(
                    "Slice step≠1 ({})",
                    node.name
                )]));
            }
            let mut out = x.clone();
            for (i, &ax) in axes.iter().enumerate() {
                let ax = if ax < 0 { (x.len() as i64 + ax) as usize } else { ax as usize };
                let dim = x[ax];
                let norm = |v: i64| if v < 0 { (dim + v).max(0) } else { v.min(dim) };
                out[ax] = (norm(ends[i]) - norm(starts[i])).max(0);
            }
            one(out)
        }
        "Transpose" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            let perm = node
                .attr_is("perm")
                .map(|v| v.to_vec())
                .unwrap_or_else(|| (0..x.len() as i64).rev().collect());
            one(perm.iter().map(|p| x[*p as usize]).collect())
        }
        "ReduceMean" => {
            let Some(x) = shape_of(g, &node.inputs[0]) else { return Ok(None) };
            let axes: Vec<i64> = if let Some(a) = node.attr_is("axes") {
                a.to_vec()
            } else if let Some(second) = node.inputs.get(1) {
                match g.info(second).and_then(|t| t.as_i64s().map(|v| v.to_vec())) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            } else {
                (0..x.len() as i64).collect()
            };
            let keep = node.attr_i("keepdims").unwrap_or(1) == 1;
            let rank = x.len() as i64;
            let axset: Vec<usize> = axes
                .iter()
                .map(|a| if *a < 0 { (rank + a) as usize } else { *a as usize })
                .collect();
            let mut out = Vec::new();
            for (i, d) in x.iter().enumerate() {
                if axset.contains(&i) {
                    if keep {
                        out.push(1);
                    }
                } else {
                    out.push(*d);
                }
            }
            one(out)
        }
        "Expand" => {
            let (Some(x), Some(target)) = (
                shape_of(g, &node.inputs[0]).or_else(|| {
                    // 상태 입력은 심볼릭일 수 있음 — target만으로 결정
                    Some(vec![])
                }),
                node.inputs
                    .get(1)
                    .and_then(|n| g.info(n))
                    .and_then(|t| t.as_i64s().map(|v| v.to_vec())),
            ) else {
                return Ok(None);
            };
            if x.is_empty() {
                return one(target);
            }
            match broadcast(&x, &target) {
                Some(s) => one(s),
                None => Err(ConvertError::Malformed(format!("Expand 불가 ({})", node.name))),
            }
        }
        // eval 전용 op들 (여기 오면 아직 상수 미해소)
        "Shape" | "Cast" | "Gather" | "Squeeze" | "Unsqueeze" => Ok(None),
        other => Err(ConvertError::Unsupported(vec![format!(
            "op {other} ({})",
            node.name
        )])),
    }
}
