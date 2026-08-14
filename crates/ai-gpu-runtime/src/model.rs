//! Model — .sw 로드와 추론의 오케스트레이터.
//!
//! 로드 산출물: 슬롯 버퍼(liveness 재사용), 가중치 단일 버퍼, params 테이블,
//! even/odd OpDispatch 리스트(상태 ping-pong — 상태를 만지는 op만 bind group 2벌).
//! 추론: 인코더 1개 → record → 출력 스테이징 복사 → submit. 프레임 루프 생성물 0.

use std::collections::{HashMap, HashSet};

use ai_core::format::SwModel;
use ai_core::{pack, TensorDesc};
use ai_gpu::kernel::OpDispatch;
use ai_gpu::params::ParamsTable;
use ai_gpu::wgpu;
use ai_gpu::{readback, GpuContext};

use crate::compile;
use crate::error::RuntimeError;
use crate::lowering::{self, LoweredOp, RtBinding};
use crate::plan;

pub struct LoadReport {
    pub ops: usize,
    pub unique_pipelines: usize,
    pub slots: usize,
    pub arena_bytes: u64,
    pub weights_bytes: u64,
    pub load_ms: f64,
}

pub struct Model {
    pub sw: SwModel,
    buffers: Vec<wgpu::Buffer>,
    #[allow(dead_code)]
    weights: wgpu::Buffer,
    #[allow(dead_code)]
    params: ParamsTable,
    ops_even: Vec<OpDispatch>,
    ops_odd: Vec<OpDispatch>,
    labels: Vec<String>,
    frame: u32,
    name_to_tid: HashMap<String, u32>,
    slot_of: HashMap<u32, usize>,
    /// (출력 tid(실효), 이름, 스테이징)
    out_staging: Vec<(u32, String, wgpu::Buffer)>,
    state_slots: Vec<(usize, usize)>, // (in 슬롯, out 슬롯)
    state_tids: Vec<(u32, u32)>,
    pub report: LoadReport,
}

fn desc_of(sw: &SwModel, tid: u32) -> TensorDesc {
    let t = &sw.tensors[tid as usize];
    TensorDesc::new(t.h, t.w, t.c, t.dt)
}

impl Model {
    pub async fn load(ctx: &GpuContext, bytes: &[u8]) -> Result<Self, RuntimeError> {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();

        let (sw, blob) = SwModel::parse_container(bytes)?;

        // 가중치 단일 버퍼 (블롭 그대로 = memcpy 로드)
        let weights = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights"),
            size: (blob.len() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&weights, 0, blob);

        // 별칭 분류: 순수 rename → 백킹으로 매핑, 비정렬/부분 뷰 → 실체화 대상
        let mut rename: HashMap<u32, u32> = HashMap::new();
        let mut views: HashMap<u32, (u32, u32)> = HashMap::new(); // view → (backing, cg_off)
        for (i, t) in sw.tensors.iter().enumerate() {
            if t.alias.is_some() {
                let (root, cg_off) = sw.resolve_alias(i as u32);
                let root_c = sw.tensors[root as usize].c;
                if cg_off == 0 && t.c == root_c {
                    rename.insert(i as u32, root);
                } else {
                    views.insert(i as u32, (root, cg_off));
                }
            }
        }
        let map = |tid: u32| -> u32 { *rename.get(&tid).unwrap_or(&tid) };

        // lowering + 뷰 실체화 삽입 (첫 사용 직전)
        let mut lowered: Vec<LoweredOp> = Vec::new();
        let mut materialized: HashSet<u32> = HashSet::new();
        let mut ensure_view = |lowered: &mut Vec<LoweredOp>,
                               materialized: &mut HashSet<u32>,
                               tid: u32| {
            if let Some((backing, cg_off)) = views.get(&tid) {
                if materialized.insert(tid) {
                    lowered.push(lowering::materialize_view(&sw, tid, map(*backing), *cg_off));
                }
            }
        };
        for op in &sw.ops {
            // lowering이 흡수한 뷰는 백킹 tid로 바인딩돼 나오므로 ensure_view가 그냥 지나간다
            let lo = lowering::lower_op(&sw, op, &map, &views)?;
            for b in &lo.bindings {
                if let RtBinding::Tensor(t) = b {
                    ensure_view(&mut lowered, &mut materialized, *t);
                }
            }
            lowered.push(lo);
        }
        for &out in &sw.outputs {
            ensure_view(&mut lowered, &mut materialized, map(out));
        }
        // AI_RT_DUMP_OPS=1: lowering 결과(라벨+바인딩 tid)를 그대로 찍는다.
        // "이 op이 뷰를 흡수했나 실체화 버퍼를 읽나"를 추측 말고 확인하기 위한 것.
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("AI_RT_DUMP_OPS").is_ok() {
            for (i, lo) in lowered.iter().enumerate() {
                let b: Vec<String> = lo
                    .bindings
                    .iter()
                    .map(|x| match x {
                        RtBinding::Tensor(t) => format!("t{t}"),
                        RtBinding::Weights(w) => format!("w@{}", w.off),
                    })
                    .collect();
                eprintln!(
                    "LOWERED[{i}] {} | {} | key={}",
                    lo.label,
                    b.join(" "),
                    lo.spec.cache_key(&ctx.caps)
                );
            }
        }

        // liveness 슬롯 배정
        let mut storage_tids: HashSet<u32> = HashSet::new();
        for lo in &lowered {
            storage_tids.extend(lo.reads.iter());
            storage_tids.extend(lo.writes.iter());
        }
        let mut persistent: Vec<u32> = Vec::new();
        for &i in &sw.inputs {
            persistent.push(map(i));
        }
        for &o in &sw.outputs {
            persistent.push(map(o));
        }
        for s in &sw.states {
            persistent.push(map(s.input));
            persistent.push(map(s.output));
        }
        // 프리로드 상수 — 로드 시 1회 채우고 영원히 산다 (재사용 금지)
        for c in &sw.consts {
            persistent.push(map(c.tid));
        }
        let live_ops: Vec<plan::LiveOp> = lowered
            .iter()
            .map(|lo| plan::LiveOp { reads: lo.reads.clone(), writes: lo.writes.clone() })
            .collect();
        let storage_vec: Vec<u32> = storage_tids.iter().copied().collect();
        // AI_RT_NO_REUSE: 디버그용 — 슬롯 재사용을 꺼서 실행 후 모든 중간 텐서를 읽을 수 있게
        #[cfg(not(target_arch = "wasm32"))]
        let no_reuse = std::env::var("AI_RT_NO_REUSE").is_ok();
        #[cfg(target_arch = "wasm32")]
        let no_reuse = false;
        let mut persistent_eff: Vec<u32> =
            if no_reuse { storage_vec.clone() } else { persistent.clone() };
        // AI_RT_REUSE_FROM=K: 디버그 이진탐색 — op K 이전에 생산된 텐서는 재사용 금지
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(k) = std::env::var("AI_RT_REUSE_FROM") {
            let k: usize = k.parse().unwrap_or(0);
            for lo in lowered.iter().take(k) {
                persistent_eff.extend(lo.writes.iter());
            }
        }
        // 상태 tid는 전용 슬롯 (프레임 간 보존 — 재사용 완전 배제)
        let dedicated: Vec<u32> = sw
            .states
            .iter()
            .flat_map(|s| [map(s.input), map(s.output)])
            .collect();
        let slot_plan =
            plan::assign_slots(&live_ops, &storage_vec, &persistent_eff, &dedicated, |tid| {
                desc_of(&sw, tid).size_bytes()
            });

        let buffers: Vec<wgpu::Buffer> = slot_plan
            .slot_sizes
            .iter()
            .enumerate()
            .map(|(i, size)| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("t{i}")),
                    size: (*size).max(4),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let arena_bytes: u64 = slot_plan.slot_sizes.iter().sum();

        // params 테이블
        let params = ParamsTable::create(ctx, lowered.len() as u32);
        for (i, lo) in lowered.iter().enumerate() {
            params.write(ctx, i as u32, &lo.params);
        }

        // 파이프라인 병렬 컴파일
        let spec_refs: Vec<&dyn ai_gpu::kernel::KernelSpec> =
            lowered.iter().map(|l| l.spec.as_ref()).collect();
        let kernels = compile::compile_all(ctx, &spec_refs).await?;

        // 상태 슬롯 쌍
        let state_tids: Vec<(u32, u32)> =
            sw.states.iter().map(|s| (map(s.input), map(s.output))).collect();
        let state_slots: Vec<(usize, usize)> = state_tids
            .iter()
            .map(|(i, o)| (slot_plan.slot_of[i], slot_plan.slot_of[o]))
            .collect();
        let state_set: HashSet<u32> =
            state_tids.iter().flat_map(|(i, o)| [*i, *o]).collect();

        let resolve_slot = |tid: u32, parity: usize| -> usize {
            for (k, (it, ot)) in state_tids.iter().enumerate() {
                let (a, b) = state_slots[k];
                if tid == *it {
                    return if parity == 0 { a } else { b };
                }
                if tid == *ot {
                    return if parity == 0 { b } else { a };
                }
            }
            slot_plan.slot_of[&tid]
        };

        // bind group + OpDispatch (even/odd)
        let make_bg = |lo: &LoweredOp, kernel: &ai_gpu::kernel::CompiledKernel, parity: usize| {
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params.binding() }];
            for (i, b) in lo.bindings.iter().enumerate() {
                let resource = match b {
                    RtBinding::Tensor(t) => buffers[resolve_slot(*t, parity)].as_entire_binding(),
                    RtBinding::Weights(w) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weights,
                        offset: w.off,
                        size: std::num::NonZeroU64::new(w.len),
                    }),
                };
                entries.push(wgpu::BindGroupEntry { binding: (i + 1) as u32, resource });
            }
            std::sync::Arc::new(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &kernel.bgl,
                entries: &entries,
            }))
        };

        let mut ops_even = Vec::with_capacity(lowered.len());
        let mut ops_odd = Vec::with_capacity(lowered.len());
        let mut labels = Vec::with_capacity(lowered.len());
        for (i, lo) in lowered.iter().enumerate() {
            let key = lo.spec.cache_key(&ctx.caps);
            let kernel = kernels
                .get(&key)
                .ok_or_else(|| RuntimeError::Other(format!("커널 누락: {key}")))?
                .clone();
            let touches_state = lo
                .bindings
                .iter()
                .any(|b| matches!(b, RtBinding::Tensor(t) if state_set.contains(t)));
            let bg_even = make_bg(lo, &kernel, 0);
            let bg_odd = if touches_state { make_bg(lo, &kernel, 1) } else { bg_even.clone() };
            let groups = lo.spec.workgroups();
            let param_offset = ParamsTable::offset(i as u32);
            ops_even.push(OpDispatch {
                kernel: kernel.clone(),
                bind_group: bg_even,
                groups,
                param_offset,
                label: lo.label.clone(),
            });
            ops_odd.push(OpDispatch {
                kernel,
                bind_group: bg_odd,
                groups,
                param_offset,
                label: lo.label.clone(),
            });
            labels.push(lo.label.clone());
        }

        // 출력 스테이징
        let out_staging: Vec<(u32, String, wgpu::Buffer)> = sw
            .outputs
            .iter()
            .map(|&o| {
                let eff = map(o);
                let size = desc_of(&sw, eff).size_bytes();
                let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("out-{o}")),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (eff, sw.tensors[o as usize].name.clone(), staging)
            })
            .collect();

        let name_to_tid: HashMap<String, u32> = sw
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i as u32))
            .collect();

        let unique = kernels.len();
        let mut model = Self {
            sw,
            buffers,
            weights,
            params,
            ops_even,
            ops_odd,
            labels,
            frame: 0,
            name_to_tid,
            slot_of: slot_plan.slot_of,
            out_staging,
            state_slots,
            state_tids,
            report: LoadReport {
                ops: lowered.len(),
                unique_pipelines: unique,
                slots: slot_plan.slot_sizes.len(),
                arena_bytes,
                weights_bytes: blob.len() as u64,
                load_ms: 0.0,
            },
        };

        // 프리로드 상수 (학습된 토큰 등 — face_blendshapes AddExtraTokens):
        // 블롭의 밀집 f32 → NHWC-C4 팩 → 텐서 버퍼 (로드 1회)
        for cst in &model.sw.consts {
            let (root, _) = model.sw.resolve_alias(cst.tid);
            let desc = desc_of(&model.sw, root);
            let bytes = &blob[cst.w.off as usize..(cst.w.off + cst.w.len) as usize];
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect();
            let slot = model.slot_of[&root];
            ctx.queue.write_buffer(&model.buffers[slot], 0, &pack::pack_nhwc(&data, &desc));
        }

        // 워밍업 — 백엔드 컴파일 강제 + 그래프 전체 유효성 확인
        model.infer(ctx).await?;
        readback::wait_idle(ctx).await.map_err(RuntimeError::Gpu)?;
        model.reset(ctx);

        #[cfg(not(target_arch = "wasm32"))]
        {
            model.report.load_ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        Ok(model)
    }

    /// 상태·프레임 카운터 리셋 (워밍업 뒤, 또는 스트림 재시작 시)
    pub fn reset(&mut self, ctx: &GpuContext) {
        self.frame = 0;
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (a, b) in &self.state_slots {
            enc.clear_buffer(&self.buffers[*a], 0, None);
            enc.clear_buffer(&self.buffers[*b], 0, None);
        }
        ctx.queue.submit([enc.finish()]);
    }

    fn tid_by_name(&self, name: &str) -> Result<u32, RuntimeError> {
        self.name_to_tid
            .get(name)
            .copied()
            .ok_or_else(|| RuntimeError::Other(format!("텐서 없음: {name}")))
    }

    /// 논리 NHWC f32 입력 업로드
    pub fn upload_input(
        &self,
        ctx: &GpuContext,
        name: &str,
        data: &[f32],
    ) -> Result<(), RuntimeError> {
        let tid = self.tid_by_name(name)?;
        let (root, _) = self.sw.resolve_alias(tid);
        let desc = desc_of(&self.sw, root);
        let slot = self.slot_of[&root];
        ctx.queue.write_buffer(&self.buffers[slot], 0, &pack::pack_nhwc(data, &desc));
        Ok(())
    }

    /// 1프레임 추론 — 인코더 1개, 상태는 리스트 교대
    pub async fn infer(&mut self, ctx: &GpuContext) -> Result<(), RuntimeError> {
        let parity = (self.frame % 2) as usize;
        let ops = if parity == 0 { &self.ops_even } else { &self.ops_odd };
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        #[cfg(not(target_arch = "wasm32"))]
        let pass_per_op = std::env::var("AI_RT_PASS_PER_OP").is_ok();
        #[cfg(target_arch = "wasm32")]
        let pass_per_op = false;
        if pass_per_op {
            for op in ops {
                ai_gpu::kernel::record(&mut enc, std::slice::from_ref(op));
            }
        } else {
            ai_gpu::kernel::record(&mut enc, ops);
        }
        for (tid, _, staging) in &self.out_staging {
            let slot = self.resolve_slot_rt(*tid, parity);
            let size = desc_of(&self.sw, *tid).size_bytes();
            enc.copy_buffer_to_buffer(&self.buffers[slot], 0, staging, 0, size);
        }
        ctx.queue.submit([enc.finish()]);
        self.frame += 1;
        Ok(())
    }

    fn resolve_slot_rt(&self, tid: u32, parity: usize) -> usize {
        for (k, (it, ot)) in self.state_tids.iter().enumerate() {
            let (a, b) = self.state_slots[k];
            if tid == *it {
                return if parity == 0 { a } else { b };
            }
            if tid == *ot {
                return if parity == 0 { b } else { a };
            }
        }
        self.slot_of[&tid]
    }

    /// 직전 infer의 출력 읽기 (논리 NHWC f32)
    pub async fn read_output(
        &self,
        ctx: &GpuContext,
        name: &str,
    ) -> Result<Vec<f32>, RuntimeError> {
        let (tid, _, staging) = self
            .out_staging
            .iter()
            .find(|(_, n, _)| n == name)
            .ok_or_else(|| RuntimeError::Other(format!("출력 아님: {name}")))?;
        let bytes = readback::read_buffers(ctx, &[staging])
            .await
            .map_err(RuntimeError::Gpu)?
            .remove(0);
        Ok(pack::unpack_nhwc(&bytes, &desc_of(&self.sw, *tid)))
    }

    /// 출력 텐서가 실제로 들어있는 **스토리지 버퍼**와 desc.
    /// 리드백 없이 GPU에서 바로 그리려면(present) 이게 필요하다 —
    /// out_staging의 버퍼는 MAP_READ라 셰이더에 바인딩할 수 없다.
    /// 입력 텐서의 스토리지 버퍼 + desc — **GPU 전처리 직결용** (컴퓨트 패스가
    /// 프레임 텍스처에서 NHWC-C4로 바로 쓴다, CPU 픽셀 왕복 0). upload_input의
    /// GPU 짝. 상태 텐서가 아닌 그래프 입력은 parity와 무관하다.
    pub fn input_storage(&self, name: &str) -> Option<(&wgpu::Buffer, TensorDesc)> {
        let tid = *self.name_to_tid.get(name)?;
        let (root, _) = self.sw.resolve_alias(tid);
        let desc = desc_of(&self.sw, root);
        let slot = *self.slot_of.get(&root)?;
        Some((&self.buffers[slot], desc))
    }

    pub fn output_storage(&self, name: &str) -> Option<(&wgpu::Buffer, TensorDesc)> {
        let (tid, _, _) = self.out_staging.iter().find(|(_, n, _)| n == name)?;
        // pha 같은 그래프 출력은 상태 텐서가 아니라 parity와 무관하다
        let slot = self.resolve_slot_rt(*tid, 0);
        Some((&self.buffers[slot], desc_of(&self.sw, *tid)))
    }

    pub fn op_labels(&self) -> &[String] {
        &self.labels
    }

    /// 진단: op 하나만 `reps`번 반복 제출하고 1회당 wall 시간(ms)을 돌려준다.
    ///
    /// Apple GPU는 컴퓨트 패스 경계 타임스탬프를 샘플링하지 않아 `profile_infer`의
    /// 수치가 0이거나 부풀려진다. 반복 격리 측정은 타임스탬프에 전혀 의존하지 않으므로
    /// 이 기기에서 유일하게 믿을 수 있는 per-op 비용이다.
    /// 같은 op을 연달아 쓰므로 가중치가 캐시에 남아 실제보다 낙관적일 수 있다 —
    /// **반드시 합계를 실제 프레임타임과 대조할 것**.
    pub async fn bench_op(
        &mut self,
        ctx: &GpuContext,
        idx: usize,
        reps: usize,
    ) -> Result<f64, RuntimeError> {
        // ⚠ 컴퓨트 **패스** 1개에 디스패치 reps개를 넣어야 한다. record()를 reps번
        // 부르면 패스가 reps개 생기고, Metal은 패스마다 인코더를 새로 만들어
        // ~17µs가 붙는다 — 모든 op이 17µs로 보이는 가짜 바닥이 생긴다.
        let batch = vec![self.ops_even[idx].clone(); reps];
        for _ in 0..8 {
            let mut enc =
                ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            ai_gpu::kernel::record(&mut enc, &batch[..1]);
            ctx.queue.submit([enc.finish()]);
        }
        readback::wait_idle(ctx).await.map_err(RuntimeError::Gpu)?;
        // 인코딩 비용을 측정에서 빼려고 한 인코더에 reps개를 넣는다.
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ai_gpu::kernel::record(&mut enc, &batch);
        let cmd = enc.finish();
        let t0 = std::time::Instant::now();
        ctx.queue.submit([cmd]);
        readback::wait_idle(ctx).await.map_err(RuntimeError::Gpu)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
    }

    /// 진단: 앞 `n`개 op만 `frames`번 제출 — 프리픽스 차분 프로파일용.
    pub async fn bench_prefix(
        &mut self,
        ctx: &GpuContext,
        n: usize,
        frames: usize,
    ) -> Result<f64, RuntimeError> {
        let ops = &self.ops_even[..n];
        for _ in 0..2 {
            let mut enc =
                ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            ai_gpu::kernel::record(&mut enc, ops);
            ctx.queue.submit([enc.finish()]);
        }
        readback::wait_idle(ctx).await.map_err(RuntimeError::Gpu)?;
        let t0 = std::time::Instant::now();
        for _ in 0..frames {
            let mut enc =
                ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            ai_gpu::kernel::record(&mut enc, ops);
            ctx.queue.submit([enc.finish()]);
        }
        readback::wait_idle(ctx).await.map_err(RuntimeError::Gpu)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / frames as f64)
    }

    /// per-op 프로파일 추론 — op마다 별도 패스 + 타임스탬프. 느리므로 진단 전용.
    /// 반환: (label, ms) — op 순서대로.
    pub async fn profile_infer(
        &mut self,
        ctx: &GpuContext,
    ) -> Result<Vec<(String, f64)>, RuntimeError> {
        use ai_gpu::profile::GpuProfiler;
        let parity = (self.frame % 2) as usize;
        let ops = if parity == 0 { &self.ops_even } else { &self.ops_odd };
        let profiler = GpuProfiler::new(ctx, ops.len() as u32)
            .ok_or_else(|| RuntimeError::Other("timestamp 미지원".into()))?;
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (i, op) in ops.iter().enumerate() {
            let tw = profiler.timestamp_writes(i as u32);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(tw),
            });
            pass.set_pipeline(&op.kernel.pipeline);
            pass.set_bind_group(0, op.bind_group.as_ref(), &[op.param_offset]);
            pass.dispatch_workgroups(op.groups[0], op.groups[1], op.groups[2]);
        }
        profiler.resolve(&mut enc, ops.len() as u32);
        ctx.queue.submit([enc.finish()]);
        self.frame += 1;
        let ms = profiler.read_ms(ctx, ops.len() as u32).await.map_err(RuntimeError::Gpu)?;
        Ok(self.labels.iter().cloned().zip(ms).collect())
    }

    /// 디버그: 임의 텐서(tid)의 현재 슬롯 내용 읽기.
    /// 슬롯 재사용이 켜져 있으면 이후 op이 덮어쓴 값일 수 있다 — AI_RT_NO_REUSE와 함께 사용.
    pub async fn debug_read_tensor(
        &self,
        ctx: &GpuContext,
        tid: u32,
    ) -> Result<Vec<f32>, RuntimeError> {
        let (root, _) = self.sw.resolve_alias(tid);
        let slot = *self
            .slot_of
            .get(&root)
            .ok_or_else(|| RuntimeError::Other(format!("스토리지 없음 tid {tid}")))?;
        let desc = desc_of(&self.sw, root);
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: desc.size_bytes(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.buffers[slot], 0, &staging, 0, desc.size_bytes());
        ctx.queue.submit([enc.finish()]);
        let bytes = readback::read_buffers(ctx, &[&staging])
            .await
            .map_err(RuntimeError::Gpu)?
            .remove(0);
        Ok(pack::unpack_nhwc(&bytes, &desc))
    }
}
