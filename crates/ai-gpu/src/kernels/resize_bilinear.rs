//! bilinear 리사이즈 커널 — half_pixel/asymmetric 좌표 변환은 codegen 플래그.

use ai_core::ops::CoordMode;
use ai_core::DType;

use crate::context::DeviceCaps;
use crate::kernel::{KernelSpec, StorageDir};
use crate::kernels::common::source::{self, SrcView};
use crate::kernels::common::writer::fill;
use crate::kernels::common::sv4_alias;

const TEMPLATE: &str = include_str!("shaders/resize_bilinear.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct ResizeBilinearSpec {
    pub ih: u32,
    pub iw: u32,
    pub c: u32,
    pub oh: u32,
    pub ow: u32,
    pub mode: CoordMode,
    pub dt: DType,
    /// 입력 소스들 — 2개 이상이면 concat-into-resize 융합, 1개면 채널 뷰,
    /// 전부 비면 평범한 단일 입력. cg 범위로 소스 선택.
    pub srcs: [SrcView; 3],
}

/// (버퍼명, cg 폭, 출력 cg축 오프셋, 백킹 행 stride cg, 백킹 시작 cg)
struct InPart {
    buf: String,
    cg: u32,
    off: u32,
    stride_cg: u32,
    off_cg: u32,
}

impl ResizeBilinearSpec {
    fn cg(&self) -> u32 {
        self.c.div_ceil(4)
    }

    fn nsrcs(&self) -> u32 {
        self.srcs.iter().filter(|s| s.is_used()).count() as u32
    }

    fn parts(&self) -> Vec<InPart> {
        if self.nsrcs() <= 1 {
            let s = self.srcs[0];
            let (stride_cg, off_cg) =
                if s.is_used() { (s.stride_cg(), s.off_cg()) } else { (self.cg(), 0) };
            return vec![InPart { buf: "IN".into(), cg: self.cg(), off: 0, stride_cg, off_cg }];
        }
        let mut v = Vec::new();
        let mut off = 0;
        for (i, s) in self.srcs.iter().enumerate() {
            if !s.is_used() {
                break;
            }
            let cg = s.cg();
            v.push(InPart {
                buf: format!("IN{i}"),
                cg,
                off,
                stride_cg: s.stride_cg(),
                off_cg: s.off_cg(),
            });
            off += cg;
        }
        v
    }
}

impl KernelSpec for ResizeBilinearSpec {
    fn cache_key(&self, _caps: &DeviceCaps) -> String {
        format!(
            "resize {}x{}->{}x{} c{} {}{} dt={}",
            self.ih,
            self.iw,
            self.oh,
            self.ow,
            self.c,
            self.mode.tag(),
            source::key_of(&self.srcs),
            self.dt.tag()
        )
    }

    fn wgsl(&self, _caps: &DeviceCaps) -> String {
        let consts = format!(
            "const IH: i32 = {};\nconst IW: i32 = {};\nconst OH: u32 = {}u;\nconst OW: u32 = {}u;\nconst CG: u32 = {}u;\nconst SY: f32 = {}.0 / {}.0;\nconst SX: f32 = {}.0 / {}.0;",
            self.ih, self.iw, self.oh, self.ow, self.cg(), self.ih, self.oh, self.iw, self.ow
        );
        let coord = match self.mode {
            CoordMode::HalfPixel => {
                "let fy = (f32(oy) + 0.5) * SY - 0.5;\nlet fx = (f32(ox) + 0.5) * SX - 0.5;"
            }
            CoordMode::Asymmetric => "let fy = f32(oy) * SY;\nlet fx = f32(ox) * SX;",
        };
        let parts = self.parts();
        // 바인딩 블록 + cg 범위로 파트를 고르는 로드 함수
        let mut binds = String::new();
        let mut b = 1u32;
        for p in &parts {
            let name = &p.buf;
            binds.push_str(&format!(
                "@group(0) @binding({b}) var<storage, read> {name}: array<sv4>;\n"
            ));
            b += 1;
        }
        binds.push_str(&format!(
            "@group(0) @binding({b}) var<storage, read_write> OUT: array<sv4>;"
        ));
        let mut ld = String::from("fn ld(pix: u32, cg: u32) -> vec4f {\n");
        for (i, p) in parts.iter().enumerate() {
            let (name, off) = (&p.buf, p.off);
            // 뷰면 백킹 stride로 행을 잡고 시작 채널그룹만큼 민다
            let idx = format!("pix * {}u + {}u + cg - {off}u", p.stride_cg, p.off_cg);
            if i + 1 < parts.len() {
                ld.push_str(&format!(
                    "  if (cg < {}u) {{ return vec4f({name}[{idx}]); }}\n",
                    off + p.cg
                ));
            } else {
                ld.push_str(&format!("  return vec4f({name}[{idx}]);\n"));
            }
        }
        ld.push_str("}");
        fill(
            TEMPLATE,
            &[
                ("TYPES", sv4_alias(self.dt)),
                ("BINDINGS", binds),
                ("CONSTS", consts),
                ("LOAD_FN", ld),
                ("COORD", coord.to_string()),
            ],
        )
    }

    fn bindings(&self) -> Vec<StorageDir> {
        let n_in = self.nsrcs().max(1) as usize;
        let mut b = vec![StorageDir::Read; n_in];
        b.push(StorageDir::ReadWrite);
        b
    }

    fn workgroups(&self) -> [u32; 3] {
        [(self.oh * self.ow * self.cg()).div_ceil(256), 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::validate_wgsl;

    #[test]
    fn naga_validates() {
        let caps = crate::test_util::fake_caps();
        for mode in [CoordMode::HalfPixel, CoordMode::Asymmetric] {
            let spec = ResizeBilinearSpec {
                ih: 18,
                iw: 32,
                c: 40,
                oh: 36,
                ow: 64,
                mode,
                dt: DType::F32,
                srcs: [SrcView::NONE; 3],
            };
            validate_wgsl(&spec.wgsl(&caps));
            // concat 융합 3파트 (홀수 꼬리)
            let multi = ResizeBilinearSpec { c: 107, srcs: [SrcView::plain(80), SrcView::plain(24), SrcView::plain(3)], ..spec };
            validate_wgsl(&multi.wgsl(&caps));
        }
    }
}
