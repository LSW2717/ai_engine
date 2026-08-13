//! 프레임 루프 — Plan 디스패치. 분석·재패킹은 전부 plan에서 끝났고,
//! 여기는 슬롯 꺼내기(mem::take) → 커널 호출 → 되돌리기만 한다.
//!
//! 상태 ping-pong은 **프레임 시작**에 swap한다: 직전 프레임 출력이 이번
//! 입력이 되고, 이번 출력 슬롯은 프레임이 끝나도 신선하게 남아
//! read_output이 항상 최신 값을 읽는다.

use ai_core::format::SwModel;

use crate::kernels::{conv, dw, elementwise, im2row, pool, pw_dot, resize, segate, shape};
use crate::plan::{self, Operand, Plan, PlanKind, Step, ViewRef};
use crate::view::View;
use crate::CpuError;

pub struct Model {
    sw: SwModel,
    plan: Plan,
    slots: Vec<Vec<f32>>,
    /// SeGate 재사용 버퍼 (프레임 중 할당 회피)
    se_scratch: (Vec<f32>, Vec<f32>),
    /// ConvStem im2row 패치 버퍼 (로드 시 최대 크기로 확보)
    im_scratch: Vec<f32>,
    /// conv/dw 행 밴드 분할 폭 (1 = 단일 스레드)
    threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pool: Option<rayon::ThreadPool>,
}

fn view<'s>(slots: &'s [Vec<f32>], vr: ViewRef) -> View<'s> {
    View { data: &slots[vr.slot], c_off: vr.c_off, stride: vr.stride, c: vr.c }
}

impl Model {
    pub fn load(bytes: &[u8]) -> Result<Self, CpuError> {
        let (sw, blob) = SwModel::parse_container(bytes)
            .map_err(|e| CpuError::Format(format!("{e:?}")))?;
        let plan = plan::build(&sw, blob)?;
        // +4 패딩: 커널의 4레인 로드가 텐서 마지막 픽셀에서 최대 3개를 초과
        // 읽어도 (K패딩 conv, 값은 가중치 0으로 소거) 슬라이스 안에 있게 한다.
        let slots = plan.slot_len.iter().map(|&l| vec![0f32; l + 4]).collect();
        let im_len = plan
            .steps
            .iter()
            .filter_map(|s| match &s.kind {
                PlanKind::ConvStem { px, k_pad, .. } => Some(px * k_pad + 4),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            sw,
            plan,
            slots,
            se_scratch: (vec![], vec![]),
            im_scratch: vec![0f32; im_len],
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
                &mut self.im_scratch,
                self.threads,
                &step.kind,
                &mut out[..step.out_len],
            );
            self.slots[step.out_slot] = out;
        }
        Ok(())
    }

    /// 스텝별 반복 벤치 — 각 스텝을 단독으로 reps회 돌려 1회 평균 ms를 잰다.
    /// wasm의 100µs 타이머 양자화를 반복 합산으로 우회한다 (같은 스텝 재실행은
    /// 같은 입력을 다시 읽어 같은 출력을 다시 쓰므로 안전). 캐시가 데워진
    /// 수치라는 점만 유의 — 절대값이 아니라 op 간 비중을 보는 도구.
    pub fn bench_steps(&mut self, reps: usize) -> Vec<StepProf> {
        #[cfg(target_arch = "wasm32")]
        use web_time::Instant;
        #[cfg(not(target_arch = "wasm32"))]
        use std::time::Instant;

        for &(a, b) in &self.plan.states {
            self.slots.swap(a, b);
        }
        let mut rows = Vec::with_capacity(self.plan.steps.len());
        for si in 0..self.plan.steps.len() {
            let step: &Step = &self.plan.steps[si];
            let mut out = std::mem::take(&mut self.slots[step.out_slot]);
            let t0 = Instant::now();
            for _ in 0..reps.max(1) {
                dispatch(
                    &self.slots,
                    &self.plan,
                    &mut self.se_scratch,
                    &mut self.im_scratch,
                    self.threads,
                    &step.kind,
                    &mut out[..step.out_len],
                );
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
            self.slots[step.out_slot] = out;
            let (label, mflop, mb) = prof_meta(&self.plan, &step.kind, step.out_len);
            rows.push(StepProf { label, ms, mflop, mb });
        }
        rows
    }

    /// per-op 계측 실행 — 예산표(실측 vs 이론하한)의 원료. 진단 전용, 네이티브만.
    /// 반환 순서는 실행 순서 그대로 (호출자가 rep마다 min을 모은다).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn infer_profiled(&mut self) -> Vec<StepProf> {
        for &(a, b) in &self.plan.states {
            self.slots.swap(a, b);
        }
        let mut rows = Vec::with_capacity(self.plan.steps.len());
        for si in 0..self.plan.steps.len() {
            let step: &Step = &self.plan.steps[si];
            let mut out = std::mem::take(&mut self.slots[step.out_slot]);
            let t0 = std::time::Instant::now();
            dispatch(
                &self.slots,
                &self.plan,
                &mut self.se_scratch,
                &mut self.im_scratch,
                self.threads,
                &step.kind,
                &mut out[..step.out_len],
            );
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            self.slots[step.out_slot] = out;
            let (label, mflop, mb) = prof_meta(&self.plan, &step.kind, step.out_len);
            rows.push(StepProf { label, ms, mflop, mb });
        }
        rows
    }

    /// 출력을 논리 NHWC로 복사해 반환 (진단·게이트용)
    pub fn read_output(&self, name: &str) -> Result<Vec<f32>, CpuError> {
        let mut out = vec![];
        self.read_output_into(name, &mut out)?;
        Ok(out)
    }

    /// 출력을 재사용 버퍼로 복사 — 프레임 루프용 (호출당 할당 없음)
    pub fn read_output_into(&self, name: &str, out: &mut Vec<f32>) -> Result<(), CpuError> {
        let (_, vr, c, px) = self
            .plan
            .outputs
            .iter()
            .find(|(n, ..)| n == name)
            .ok_or_else(|| CpuError::Other(format!("출력 아님: {name}")))?;
        out.resize(px * c, 0.0);
        shape::copy_view_into(view(&self.slots, *vr), *px, out, *c, 0);
        Ok(())
    }

    pub fn output_names(&self) -> Vec<String> {
        self.plan.outputs.iter().map(|(n, ..)| n.clone()).collect()
    }
}

/// per-op 계측 한 줄 — ms는 실측, mflop/mb는 이론하한 계산용 원료
pub struct StepProf {
    pub label: String,
    pub ms: f64,
    /// MAC×2 기준 부동소수 연산수 (백만)
    pub mflop: f64,
    /// 최소 트래픽 = 입력+가중치+출력 각 1회 (MB) — 재독은 하한에 안 넣는다
    pub mb: f64,
}

/// PlanKind → (라벨, MFLOP, MB). 경계 tap 감소는 무시(하한이므로 과대평가 없음 —
/// flops는 interior 기준 근사, bytes는 정확).
fn prof_meta(plan: &Plan, kind: &PlanKind, out_len: usize) -> (String, f64, f64) {
    let f32b = 4.0 / 1e6; // f32 → MB
    match kind {
        PlanKind::ConvStd { op, ih, iw, oh, parts, w, res, .. } => {
            let (_, ow) = op.out_hw(*ih, *iw);
            let px = (*oh * ow) as f64;
            let taps = (op.kh * op.kw) as f64;
            let mflop = 2.0 * px * op.cout as f64 * op.cin as f64 * taps / 1e6;
            let mut elems = (*ih * *iw) as f64 * op.cin as f64
                + plan.weights[*w].len() as f64
                + out_len as f64;
            if res.is_some() {
                elems += out_len as f64;
            }
            let cat = if parts.len() > 1 { format!(" cat{}", parts.len()) } else { String::new() };
            (
                format!(
                    "conv k{}x{} s{} {}->{} @{}x{}{}",
                    op.kh, op.kw, op.sh, op.cin, op.cout, ow, oh, cat
                ),
                mflop,
                elems * f32b,
            )
        }
        PlanKind::ConvDw { op, ih, iw, oh, w, res, .. } => {
            let (_, ow) = op.out_hw(*ih, *iw);
            let px = (*oh * ow) as f64;
            let taps = (op.kh * op.kw) as f64;
            let mflop = 2.0 * px * op.cout as f64 * taps / 1e6;
            let mut elems = (*ih * *iw) as f64 * op.cin as f64
                + plan.weights[*w].len() as f64
                + out_len as f64;
            if res.is_some() {
                elems += out_len as f64;
            }
            (
                format!("dw k{} s{} c{} @{}x{}", op.kh, op.sh, op.cout, ow, oh),
                mflop,
                elems * f32b,
            )
        }
        PlanKind::Binary { bop, a, operand, px, .. } => {
            let in2 = match operand {
                Operand::Tensor(t) => (*px * t.c) as f64,
                Operand::CvecTensor(t) => t.c as f64,
                Operand::CvecConst(i) => plan.weights[*i].len() as f64,
                Operand::Scalar { .. } => 0.0,
            };
            let elems = (*px * a.c) as f64 + in2 + out_len as f64;
            (
                format!("binary {bop:?} c{} px{px}", a.c),
                (*px * a.c) as f64 / 1e6,
                elems * f32b,
            )
        }
        PlanKind::ConvStem { op, ih, iw, oh, px, w, res, .. } => {
            let (_, ow) = op.out_hw(*ih, *iw);
            let taps = (op.kh * op.kw) as f64;
            let mut elems = (*ih * *iw) as f64 * op.cin as f64
                + plan.weights[*w].len() as f64
                + out_len as f64;
            if res.is_some() {
                elems += out_len as f64;
            }
            (
                format!(
                    "stem k{}x{} s{} {}->{} @{}x{}",
                    op.kh, op.kw, op.sh, op.cin, op.cout, ow, oh
                ),
                2.0 * *px as f64 * op.cout as f64 * op.cin as f64 * taps / 1e6,
                elems * f32b,
            )
        }
        PlanKind::PwDot { op, input, px, w, .. } => (
            format!("pwdot k1x1 {}->{} px{px}", op.cin, op.cout),
            2.0 * *px as f64 * op.cout as f64 * op.cin as f64 / 1e6,
            ((*px * input.c) as f64 + plan.weights[*w].len() as f64 + out_len as f64) * f32b,
        ),
        PlanKind::Gpool { input, px } => (
            format!("gpool c{} px{px}", input.c),
            (*px * input.c) as f64 / 1e6,
            ((*px * input.c) as f64 + out_len as f64) * f32b,
        ),
        PlanKind::Avgpool { op, ih, iw, input } => (
            format!("avgpool k{} s{} c{}", op.kh, op.sh, input.c),
            out_len as f64 * (op.kh * op.kw) as f64 / 1e6,
            ((*ih * *iw) as f64 * input.c as f64 + out_len as f64) * f32b,
        ),
        PlanKind::Maxpool { op, ih, iw, input } => (
            format!("maxpool k{} s{} c{}", op.kh, op.sh, input.c),
            out_len as f64 * (op.kh * op.kw) as f64 / 1e6,
            ((*ih * *iw) as f64 * input.c as f64 + out_len as f64) * f32b,
        ),
        PlanKind::Resize { op, ih, iw, parts } => {
            let cin: usize = parts.iter().map(|p| p.c).sum();
            let cat = if parts.len() > 1 { format!(" cat{}", parts.len()) } else { String::new() };
            (
                format!("resize {}x{}->{}x{} c{cin}{cat}", iw, ih, op.ow, op.oh),
                out_len as f64 * 7.0 / 1e6, // bilinear ≈ 픽셀당 7 flop
                ((*ih * *iw) as f64 * cin as f64 + out_len as f64) * f32b,
            )
        }
        PlanKind::Concat { parts, px, .. } => (
            format!("concat {}파트 px{px}", parts.len()),
            0.0,
            2.0 * out_len as f64 * f32b,
        ),
        PlanKind::Chcopy { input, px } => (
            format!("chcopy c{} px{px}", input.c),
            0.0,
            2.0 * out_len as f64 * f32b,
        ),
        PlanKind::Act { input, px, act } => (
            format!("act {act:?} c{} px{px}", input.c),
            out_len as f64 / 1e6,
            ((*px * input.c) as f64 + out_len as f64) * f32b,
        ),
        PlanKind::Mix { z, px, .. } => (
            format!("mix c{} px{px}", z.c),
            3.0 * out_len as f64 / 1e6,
            4.0 * out_len as f64 * f32b,
        ),
        PlanKind::SeGate { input, px, fc1, fc2 } => {
            let mut welems = plan.fcs[*fc1].w.len() as f64;
            if let Some(i) = fc2 {
                welems += plan.fcs[*i].w.len() as f64;
            }
            (
                format!("segate c{} px{px}", input.c),
                (2.0 * welems + (*px * input.c) as f64) / 1e6,
                ((*px * input.c) as f64 + welems + out_len as f64) * f32b,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    slots: &[Vec<f32>],
    plan: &Plan,
    se_scratch: &mut (Vec<f32>, Vec<f32>),
    im_scratch: &mut [f32],
    threads: usize,
    kind: &PlanKind,
    out: &mut [f32],
) {
    match kind {
        PlanKind::ConvStd { op, ih, iw, oh, parts, w, b, res } => {
            // 파트 소형 Vec — 프레임당 conv 수 × 수십 ns라 전체 예산에서 무시 가능
            let parts: Vec<conv::ConvPart> = parts
                .iter()
                .map(|p| conv::ConvPart { view: view(slots, p.view), ic0: p.ic0, c4: p.c4 })
                .collect();
            let res_v = res.map(|r| view(slots, r));
            // k1 s1 pad0는 행 구조가 무의미 — 전 픽셀을 한 행으로 접으면 MR 블록이
            // 행 끝에서 안 끊긴다 (7x7 같은 소형 맵의 mr1 낭비 제거. 밴드 분할과
            // 양립 안 해 1스레드 경로만)
            let pw1 = op.kh == 1 && op.kw == 1 && op.sh == 1 && op.sw == 1 && op.pad == [0; 4];
            let (ih, iw, oh) = if pw1 && threads <= 1 {
                (1u32, *ih * *iw, 1u32)
            } else {
                (*ih, *iw, *oh)
            };
            let (ih, iw, oh) = (&ih, &iw, &oh);
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
        PlanKind::ConvStem { op, pw, ih, iw, oh, px, k_pad, input, w, b, res } => {
            im2row::im2row(op, *ih, *iw, view(slots, *input), *k_pad, im_scratch);
            let (_, ow) = op.out_hw(*ih, *iw);
            let parts = [conv::ConvPart {
                view: View { data: im_scratch, c_off: 0, stride: *k_pad, c: *k_pad },
                ic0: 0,
                c4: *k_pad,
            }];
            let _ = px;
            let res_v = res.map(|r| view(slots, r));
            #[cfg(not(target_arch = "wasm32"))]
            if threads > 1 && *oh as usize >= threads {
                use rayon::prelude::*;
                let rows = *oh as usize;
                let band = rows.div_ceil(threads);
                let chunk_len = band * ow as usize * pw.cout as usize;
                out.par_chunks_mut(chunk_len).enumerate().for_each(|(i, chunk)| {
                    let y0 = (i * band) as u32;
                    let y1 = ((i + 1) * band).min(rows) as u32;
                    conv::conv_std(
                        pw, *oh, ow, &parts,
                        &plan.weights[*w], &plan.weights[*b],
                        res_v, chunk, y0, y1,
                    );
                });
                return;
            }
            conv::conv_std(
                pw, *oh, ow, &parts,
                &plan.weights[*w], &plan.weights[*b],
                res_v, out, 0, *oh,
            );
        }
        PlanKind::PwDot { op, input, px, w, k_pad, b } => pw_dot::conv_pw_dot(
            op,
            view(slots, *input),
            &plan.weights[*w],
            *k_pad,
            &plan.weights[*b],
            *px,
            out,
        ),
        PlanKind::Gpool { input, px } => pool::global_avg(view(slots, *input), *px, out),
        PlanKind::Avgpool { op, ih, iw, input } => {
            pool::avg_pool(op, *ih, *iw, view(slots, *input), out)
        }
        PlanKind::Maxpool { op, ih, iw, input } => {
            pool::max_pool(op, *ih, *iw, view(slots, *input), out)
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
