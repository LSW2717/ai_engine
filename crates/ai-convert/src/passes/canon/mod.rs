//! 정규화(canonicalize) 패스군 — ONNX op 형태를 엔진 op으로 수렴시킨다.
//! 실행 순서 고정: identity → split_concat(상쇄) → resize → slice → reduce →
//! pool → hswish → transpose → clip → expand → split(전개).

pub mod clip;
pub mod expand;
pub mod hswish;
pub mod identity;
pub mod pool;
pub mod reduce;
pub mod resize;
pub mod slice;
pub mod split;
pub mod split_concat;
pub mod transpose;

use crate::error::ConvertError;
use crate::ir::Graph;
use crate::passes::{Ctx, PassReport};

pub fn run_all(g: &mut Graph, ctx: &Ctx) -> Result<PassReport, ConvertError> {
    let mut total = PassReport::default();
    for pass in [
        identity::run,
        split_concat::run,
        resize::run,
        slice::run,
        reduce::run,
        pool::run,
        hswish::run,
        transpose::run,
        clip::run,
        expand::run,
        split::run,
    ] {
        total.rewrites += pass(g, ctx)?.rewrites;
    }
    Ok(total)
}
