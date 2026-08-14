//! 정규화(canonicalize) 패스군 — ONNX op 형태를 엔진 op으로 수렴시킨다.
//! 실행 순서 고정: identity → split_concat(상쇄) → resize → slice → reduce →
//! pool → hswish → transpose → reshape(flatten) → gemm → pad(채널패드 접기) →
//! clip → expand → split(전개).
//! (reshape은 transpose의 flat_ok 마킹 뒤, pad는 pool의 maxpool 재작성 뒤여야 한다)

pub mod clip;
pub mod concat_w;
pub mod expand;
pub mod gemm;
pub mod hswish;
pub mod identity;
pub mod outer;
pub mod pad;
pub mod pool;
pub mod reduce;
pub mod rowvec;
pub mod reshape;
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
        concat_w::run,
        outer::run,
        rowvec::run,
        reshape::run,
        gemm::run,
        pad::run,
        clip::run,
        expand::run,
        split::run,
    ] {
        total.rewrites += pass(g, ctx)?.rewrites;
    }
    Ok(total)
}
