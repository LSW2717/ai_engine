//! 프레임 루프 — Plan 디스패치. 분석·재패킹은 전부 plan에서 끝났고,
//! 여기는 슬롯 꺼내기(mem::take) → 커널 호출 → 되돌리기만 한다.
//!
//! 상태 ping-pong은 **프레임 시작**에 swap한다: 직전 프레임 출력이 이번
//! 입력이 되고, 이번 출력 슬롯은 프레임이 끝나도 신선하게 남아
//! read_output이 항상 최신 값을 읽는다.

use ai_core::format::SwModel;

use crate::kernels::{conv, dw, elementwise, pool, resize, segate, shape};
use crate::plan::{self, Operand, Plan, PlanKind, Step, ViewRef};
use crate::view::View;
use crate::CpuError;

pub struct CpuModel {
    sw: SwModel,
    plan: Plan,
    slots: Vec<Vec<f32>>,
    /// SeGate 재사용 버퍼 (프레임 중 할당 회피)
    se_scratch: (Vec<f32>, Vec<f32>),
    /// conv/dw 행 밴드 분할 폭 (1 = 단일 스레드)
    threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pool: Option<rayon::ThreadPool>,
}

fn view<'s>(slots: &'s [Vec<f32>], vr: ViewRef) -> View<'s> {
    View { data: &slots[vr.slot], c_off: vr.c_off, stride: vr.stride, c: vr.c }
}

impl CpuModel {
    pub fn load(bytes: &[u8]) -> Result<Self, CpuError> {
        let (sw, blob) = SwModel::parse_container(bytes)
            .map_err(|e| CpuError::Format(format!("{e:?}")))?;
        let plan = plan::build(&sw, blob)?;
        let slots = plan.slot_len.iter().map(|&l| vec![0f32; l]).collect();
        Ok(Self {
            sw,
            plan,
            slots,
            se_scratch: (vec![], vec![]),
            threads: 1,
            #[cfg(not(target_arch = "wasm32"))]
            pool: None,
        })
    }

    /// conv/dw 병렬 폭 설정. 기본 1 (결정적). 네이티브 전용 —
    /// wasm 스레드(COOP/COEP + atomics)는 별도 단계라 웹에서는 no-op.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_threads(&mut self, n: usize) -> Result<(), CpuError> {
        let n = n.max(1);
        self.threads = n;
        self.pool = if n > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| CpuError::Other(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn set_threads(&mut self, _n: usize) -> Result<(), CpuError> {
        Ok(())
    }

    pub fn sw(&self) -> &SwModel {
        &self.sw
    }

    /// 논리 NHWC f32 입력 주입
    pub fn set_input(&mut self, name: &str, data: &[f32]) -> Result<(), CpuError> {
        let &(_, slot, len) = self
            .plan
            .inputs
            .iter()
            .find(|(n, ..)| n == name)
            .ok_or_else(|| CpuError::Other(format!("입력 아님: {name}")))?;
        if data.len() != len {
            return Err(CpuError::Other(format!(
                "입력 크기 불일치 {name}: {} != {len}",
                data.len()
            )));
        }
        self.slots[slot][..len].copy_from_slice(data);
        Ok(())
    }

    /// 그래프 1회 실행 (상태 ping-pong 포함, 동기)
    pub fn infer(&mut self) -> Result<(), CpuError> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pool) = self.pool.take() {
            let r = pool.install(|| self.run_steps());
            self.pool = Some(pool);
            return r;
        }
        self.run_steps()
    }

    fn run_steps(&mut self) -> Result<(), CpuError> {
        for &(a, b) in &self.plan.states {
            self.slots.swap(a, b);
        }
        // steps를 인덱스로 돌면 plan 불변 차용과 slots 가변 차용이 분리된다
        for si in 0..self.plan.steps.len() {
            let step: &Step = &self.plan.steps[si];
            let mut out = std::mem::take(&mut self.slots[step.out_slot]);
            dispatch(
                &self.slots,
                &self.plan,
                &mut self.se_scratch,
                self.threads,
                &step.kind,
                &mut out[..step.out_len],
            );
            self.slots[step.out_slot] = out;
        }
        Ok(())
    }

    /// 출력을 논리 NHWC로 복사해 반환 (진단·게이트용)
    pub fn read_output(&self, name: &str) -> Result<Vec<f32>, CpuError> {
        let (_, vr, c, px) = self
            .plan
            .outputs
            .iter()
            .find(|(n, ..)| n == name)
            .ok_or_else(|| CpuError::Other(format!("출력 아님: {name}")))?;
        let mut out = vec![0f32; px * c];
        shape::copy_view_into(view(&self.slots, *vr), *px, &mut out, *c, 0);
        Ok(out)
    }

    pub fn output_names(&self) -> Vec<String> {
        self.plan.outputs.iter().map(|(n, ..)| n.clone()).collect()
    }
}

fn dispatch(
    slots: &[Vec<f32>],
    plan: &Plan,
    se_scratch: &mut (Vec<f32>, Vec<f32>),
    threads: usize,
    kind: &PlanKind,
    out: &mut [f32],
) {
    match kind {
        PlanKind::ConvStd { op, ih, iw, oh, parts, w, b, res } => {
            // 파트 소형 Vec — 프레임당 conv 수 × 수십 ns라 전체 예산에서 무시 가능
            let parts: Vec<conv::ConvPart> = parts
                .iter()
                .map(|p| conv::ConvPart { view: view(slots, p.view), ic0: p.ic0 })
                .collect();
            let res_v = res.map(|r| view(slots, r));
            #[cfg(not(target_arch = "wasm32"))]
            if threads > 1 && *oh as usize >= threads {
                use rayon::prelude::*;
                let (_, ow) = op.out_hw(*ih, *iw);
                let rows = *oh as usize;
                let band = rows.div_ceil(threads);
                let chunk_len = band * ow as usize * op.cout as usize;
                out.par_chunks_mut(chunk_len).enumerate().for_each(|(i, chunk)| {
                    let y0 = (i * band) as u32;
                    let y1 = ((i + 1) * band).min(rows) as u32;
                    conv::conv_std(
                        op, *ih, *iw, &parts,
                        &plan.weights[*w], &plan.weights[*b],
                        res_v, chunk, y0, y1,
                    );
                });
                return;
            }
            conv::conv_std(
                op, *ih, *iw, &parts,
                &plan.weights[*w], &plan.weights[*b],
                res_v, out, 0, *oh,
            );
        }
        PlanKind::ConvDw { op, ih, iw, oh, input, w, b, res } => {
            let in_v = view(slots, *input);
            let res_v = res.map(|r| view(slots, r));
            #[cfg(not(target_arch = "wasm32"))]
            if threads > 1 && *oh as usize >= threads {
                use rayon::prelude::*;
                let (_, ow) = op.out_hw(*ih, *iw);
                let rows = *oh as usize;
                let band = rows.div_ceil(threads);
                let chunk_len = band * ow as usize * op.cout as usize;
                out.par_chunks_mut(chunk_len).enumerate().for_each(|(i, chunk)| {
                    let y0 = (i * band) as u32;
                    let y1 = ((i + 1) * band).min(rows) as u32;
                    dw::conv_dw(
                        op, *ih, *iw, in_v,
                        &plan.weights[*w], &plan.weights[*b],
                        res_v, chunk, y0, y1,
                    );
                });
                return;
            }
            dw::conv_dw(
                op, *ih, *iw, in_v,
                &plan.weights[*w], &plan.weights[*b],
                res_v, out, 0, *oh,
            );
        }
        PlanKind::Binary { bop, a, operand, px, act } => {
            let av = view(slots, *a);
            match operand {
                Operand::Tensor(t) => {
                    elementwise::binary_tensor(*bop, av, view(slots, *t), *px, *act, out)
                }
                Operand::Scalar { v, first } => {
                    elementwise::binary_scalar(*bop, av, *v, *first, *px, *act, out)
                }
                Operand::CvecConst(i) => {
                    elementwise::binary_cvec(*bop, av, &plan.weights[*i], *px, *act, out)
                }
                Operand::CvecTensor(t) => {
                    let tv = view(slots, *t);
                    let vec = &tv.data[tv.c_off..tv.c_off + tv.c];
                    elementwise::binary_cvec(*bop, av, vec, *px, *act, out)
                }
            }
        }
        PlanKind::Gpool { input, px } => pool::global_avg(view(slots, *input), *px, out),
        PlanKind::Avgpool { op, ih, iw, input } => {
            pool::avg_pool(op, *ih, *iw, view(slots, *input), out)
        }
        PlanKind::Resize { op, ih, iw, parts } => {
            let views: Vec<View> = parts.iter().map(|p| view(slots, *p)).collect();
            resize::resize_bilinear(op, *ih, *iw, &views, out);
        }
        PlanKind::Concat { parts, px, out_stride } => {
            let mut c_off = 0usize;
            for p in parts {
                let v = view(slots, *p);
                shape::copy_view_into(v, *px, out, *out_stride, c_off);
                c_off += v.c;
            }
        }
        PlanKind::Chcopy { input, px } => {
            let v = view(slots, *input);
            shape::copy_view_into(v, *px, out, v.c, 0);
        }
        PlanKind::Act { input, px, act } => {
            elementwise::act(view(slots, *input), *px, *act, out)
        }
        PlanKind::Mix { z, a, b, px } => elementwise::mix(
            view(slots, *z),
            view(slots, *a),
            view(slots, *b),
            *px,
            out,
        ),
        PlanKind::SeGate { input, px, fc1, fc2 } => segate::se_gate(
            view(slots, *input),
            *px,
            &plan.fcs[*fc1],
            fc2.map(|i| &plan.fcs[i]),
            se_scratch,
            out,
        ),
    }
}
