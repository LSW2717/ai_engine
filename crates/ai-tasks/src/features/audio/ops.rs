//! 오디오 미니 실행기의 텐서/연산 — rank≤4 소형 밀집 f32 전용.
//!
//! fastenhancer spec2spec 서브그래프(최대 텐서 48×128)를 돌리기 위한 ONNX
//! 시맨틱 구현. 이미지 엔진(.sw NHWC)과 달리 여기 텐서는 미소해서 제네릭
//! 스트라이드 연산으로 충분하다 — hop 예산 10.7ms 대비 전체 ~1M MAC.

use ai_cpu::simd::F32x4;

use crate::error::TaskError;

#[derive(Clone, Debug, Default)]
pub struct Tens {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tens {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>().max(1), data.len().max(1));
        Tens { shape, data }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product::<usize>().max(1);
        Tens { shape, data: vec![0.0; n] }
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn err(msg: String) -> TaskError {
    TaskError::Other(format!("audio-graph: {msg}"))
}

/// numpy 브로드캐스트 결과 shape
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TaskError> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for i in 0..r {
        let av = *a.get(a.len().wrapping_sub(r - i)).unwrap_or(&1);
        let bv = *b.get(b.len().wrapping_sub(r - i)).unwrap_or(&1);
        out[i] = if av == bv || bv == 1 {
            av
        } else if av == 1 {
            bv
        } else {
            return Err(err(format!("브로드캐스트 불가 {a:?} × {b:?}")));
        };
    }
    Ok(out)
}

/// 브로드캐스트 이항 연산 (Mul/Add/Sub).
/// 핫패스: 같은 shape(잔차/게이트)와 스칼라는 flat 루프 — hop 예산의 주범이었던
/// per-element 나눗셈 인덱싱은 일반 브로드캐스트에만 남긴다 (odometer 방식).
pub fn ew_binary(op: &str, a: &Tens, b: &Tens) -> Result<Tens, TaskError> {
    #[inline(always)]
    fn apply(op: u8, x: f32, y: f32) -> f32 {
        match op {
            0 => x * y,
            1 => x + y,
            _ => x - y,
        }
    }
    let opc = match op {
        "Mul" => 0u8,
        "Add" => 1,
        "Sub" => 2,
        _ => return Err(err(format!("미지 ew op {op}"))),
    };
    if a.shape == b.shape {
        let out = a.data.iter().zip(&b.data).map(|(&x, &y)| apply(opc, x, y)).collect();
        return Ok(Tens::new(a.shape.clone(), out));
    }
    if b.data.len() == 1 {
        let y = b.data[0];
        let out = a.data.iter().map(|&x| apply(opc, x, y)).collect();
        return Ok(Tens::new(a.shape.clone(), out));
    }
    if a.data.len() == 1 {
        let x = a.data[0];
        let out = b.data.iter().map(|&y| apply(opc, x, y)).collect();
        return Ok(Tens::new(b.shape.clone(), out));
    }
    // 특례: b가 a의 마지막 축 벡터 (bias add 모양) — 행별 flat zip
    if !a.shape.is_empty()
        && b.data.len() == *a.shape.last().unwrap()
        && b.shape.last() == a.shape.last()
        && b.shape[..b.shape.len() - 1].iter().all(|&d| d == 1)
    {
        let n = b.data.len();
        let mut out = vec![0f32; a.data.len()];
        for (orow, arow) in out.chunks_exact_mut(n).zip(a.data.chunks_exact(n)) {
            for ((o, &x), &y) in orow.iter_mut().zip(arow).zip(&b.data) {
                *o = apply(opc, x, y);
            }
        }
        return Ok(Tens::new(a.shape.clone(), out));
    }
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let r = shape.len();
    let align = |t: &Tens| -> Vec<usize> {
        let ts = strides(&t.shape);
        (0..r)
            .map(|i| {
                let j = (t.shape.len() + i).wrapping_sub(r);
                if i + t.shape.len() < r || t.shape[j] == 1 { 0 } else { ts[j] }
            })
            .collect()
    };
    let (sa, sb) = (align(a), align(b));
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut out = vec![0f32; n];
    let mut ctr = vec![0usize; r];
    let (mut ia, mut ib) = (0usize, 0usize);
    for o in out.iter_mut() {
        *o = apply(opc, a.data[ia], b.data[ib]);
        // odometer 증가
        for d in (0..r).rev() {
            ctr[d] += 1;
            ia += sa[d];
            ib += sb[d];
            if ctr[d] < shape[d] {
                break;
            }
            ctr[d] = 0;
            ia -= sa[d] * shape[d];
            ib -= sb[d] * shape[d];
        }
    }
    Ok(Tens::new(shape, out))
}

/// Cephes 5차 다항 exp (상대오차 ~1e-7) — libm expf 대비 수 배.
/// ORT(MLAS MlasComputeLogistic/Tanh)와 같은 접근: 활성화가 wasm 프로파일의
/// ~17%를 먹던 것을 다항 근사로 잡는다. 오라클 게이트(1e-4)가 정확도를 감시.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.3, 88.7);
    let n = (x * LOG2E).round();
    let r = x - n * LN2_HI - n * LN2_LO;
    // exp(r), |r| ≤ ln2/2 — Cephes 계수
    let p = 1.987_569_1e-4f32;
    let p = p * r + 1.398_199_9e-3;
    let p = p * r + 8.333_452e-3;
    let p = p * r + 4.166_579_5e-2;
    let p = p * r + 1.666_666_6e-1;
    let p = p * r + 5e-1;
    let p = p * r * r + r + 1.0;
    // 2^n 스케일 — 지수 비트 직접 구성
    let bits = (((n as i32) + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// fast_exp의 F32x4판 — 다항은 전부 벡터 fma, 2^n 스케일만 레인별 정수 비트
/// 조작(to_array 경유). 라운딩은 매직넘버(1.5·2^23) 트릭 — f32.nearest와 동일.
#[inline(always)]
pub fn exp4(x: F32x4) -> F32x4 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const MAGIC: f32 = 12_582_912.0;
    let x = x.max(F32x4::splat(-87.3)).min(F32x4::splat(88.7));
    let n = x
        .mul(F32x4::splat(LOG2E))
        .add(F32x4::splat(MAGIC))
        .add(F32x4::splat(-MAGIC));
    let r = x.fma(n, F32x4::splat(-LN2_HI)).fma(n, F32x4::splat(-LN2_LO));
    let p = F32x4::splat(1.398_199_9e-3).fma(r, F32x4::splat(1.987_569_1e-4));
    let p = F32x4::splat(8.333_452e-3).fma(r, p);
    let p = F32x4::splat(4.166_579_5e-2).fma(r, p);
    let p = F32x4::splat(1.666_666_6e-1).fma(r, p);
    let p = F32x4::splat(0.5).fma(r, p);
    // exp(r) ≈ 1 + r + r²·p — fma(self,a,b) = self + a·b
    let p = F32x4::splat(1.0).add(r).fma(r.mul(r), p);
    let na = n.to_array();
    let pa = p.to_array();
    let mut out = [0f32; 4];
    for i in 0..4 {
        let bits = (((na[i] as i32) + 127) as u32) << 23;
        out[i] = pa[i] * f32::from_bits(bits);
    }
    F32x4::from_array(out)
}

#[inline(always)]
fn fast_sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-v))
}

#[inline(always)]
fn fast_tanh(v: f32) -> f32 {
    // tanh(v) = 2σ(2v) − 1 — fast_exp 재사용, 포화 구간은 clamp가 처리
    2.0 * fast_sigmoid(2.0 * v) - 1.0
}

pub fn unary(op: &str, a: &Tens) -> Result<Tens, TaskError> {
    let n = a.data.len();
    let mut data = vec![0f32; n];
    let one = F32x4::splat(1.0);
    let sig4 = |v: F32x4| -> F32x4 {
        let e = exp4(F32x4::splat(0.0).sub(v)); // −v
        one.div(one.add(e))
    };
    match op {
        "Sigmoid" => {
            let mut i = 0usize;
            while i + 4 <= n {
                sig4(F32x4::load(&a.data, i)).store(&mut data, i);
                i += 4;
            }
            for j in i..n {
                data[j] = fast_sigmoid(a.data[j]);
            }
        }
        "Tanh" => {
            let two = F32x4::splat(2.0);
            let mut i = 0usize;
            while i + 4 <= n {
                let s = sig4(F32x4::load(&a.data, i).mul(two));
                s.mul(two).sub(one).store(&mut data, i);
                i += 4;
            }
            for j in i..n {
                data[j] = fast_tanh(a.data[j]);
            }
        }
        _ => return Err(err(format!("미지 unary op {op}"))),
    }
    Ok(Tens::new(a.shape.clone(), data))
}

pub fn transpose(a: &Tens, perm: &[usize]) -> Tens {
    let r = a.shape.len();
    debug_assert_eq!(perm.len(), r);
    let out_shape: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
    let in_s = strides(&a.shape);
    let src_s: Vec<usize> = perm.iter().map(|&p| in_s[p]).collect();
    let mut out = vec![0f32; a.data.len()];
    // 특례: 마지막 축 유지 (attention perm [0,2,1,3] 류) — 행 단위 memcpy
    if r > 0 && perm[r - 1] == r - 1 {
        let row = a.shape[r - 1];
        let rows = a.data.len() / row.max(1);
        let mut ctr = vec![0usize; r - 1];
        let mut src = 0usize;
        for orow in 0..rows {
            out[orow * row..(orow + 1) * row].copy_from_slice(&a.data[src..src + row]);
            for d in (0..r - 1).rev() {
                ctr[d] += 1;
                src += src_s[d];
                if ctr[d] < out_shape[d] {
                    break;
                }
                ctr[d] = 0;
                src -= src_s[d] * out_shape[d];
            }
        }
        return Tens::new(out_shape, out);
    }
    let mut ctr = vec![0usize; r];
    let mut src = 0usize;
    for o in out.iter_mut() {
        *o = a.data[src];
        for d in (0..r).rev() {
            ctr[d] += 1;
            src += src_s[d];
            if ctr[d] < out_shape[d] {
                break;
            }
            ctr[d] = 0;
            src -= src_s[d] * out_shape[d];
        }
    }
    Tens::new(out_shape, out)
}

/// ONNX Reshape 목표 shape 해석 (-1 추론, 0=입력 차원 복사)
pub fn resolve_reshape(in_shape: &[usize], target: &[i64]) -> Result<Vec<usize>, TaskError> {
    let numel: usize = in_shape.iter().product::<usize>().max(1);
    let mut out: Vec<usize> = Vec::with_capacity(target.len());
    let mut infer = None;
    for (i, &t) in target.iter().enumerate() {
        match t {
            -1 => {
                if infer.is_some() {
                    return Err(err("-1이 둘".into()));
                }
                infer = Some(i);
                out.push(1);
            }
            0 => out.push(*in_shape.get(i).ok_or_else(|| err("0 차원 범위 밖".into()))?),
            v if v > 0 => out.push(v as usize),
            _ => return Err(err(format!("reshape 목표 {t}"))),
        }
    }
    if let Some(i) = infer {
        let rest: usize = out.iter().product::<usize>().max(1);
        out[i] = numel / rest;
    }
    if out.iter().product::<usize>().max(1) != numel {
        return Err(err(format!("reshape {in_shape:?} → {target:?} 크기 불일치")));
    }
    Ok(out)
}

/// 배치 MatMul (rank≤4, 배치 차원 브로드캐스트) — attention의 동적 양변 포함
pub fn matmul(a: &Tens, b: &Tens) -> Result<Tens, TaskError> {
    let (ar, br) = (a.shape.len(), b.shape.len());
    if ar < 2 || br < 2 {
        return Err(err(format!("matmul rank {ar}/{br}")));
    }
    let (m, ka) = (a.shape[ar - 2], a.shape[ar - 1]);
    let (kb, n) = (b.shape[br - 2], b.shape[br - 1]);
    if ka != kb {
        return Err(err(format!("matmul K 불일치 {ka}≠{kb}")));
    }
    let batch_shape = broadcast_shape(&a.shape[..ar - 2], &b.shape[..br - 2])?;
    let batch: usize = batch_shape.iter().product::<usize>().max(1);
    let a_batch: usize = a.shape[..ar - 2].iter().product::<usize>().max(1);
    let b_batch: usize = b.shape[..br - 2].iter().product::<usize>().max(1);
    let mut out_shape = batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);
    let mut out = vec![0f32; batch * m * n];
    for bi in 0..batch {
        // 배치 브로드캐스트: 배치 1이면 같은 행렬 재사용
        let ao = (bi % a_batch.max(1)) * m * ka;
        let bo = (bi % b_batch.max(1)) * ka * n;
        let oo = bi * m * n;
        for i in 0..m {
            let arow = &a.data[ao + i * ka..ao + (i + 1) * ka];
            // j 16블록을 레지스터에 유지, k 안쪽 (conv와 같은 블로킹) —
            // i-k-j 행 갱신은 orow 메모리 왕복이 지배해 wasm에서 밀렸다
            let mut j = 0usize;
            while j + 16 <= n {
                let (mut a0, mut a1, mut a2, mut a3) = (
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                );
                for (k, &av) in arow.iter().enumerate() {
                    let brow = &b.data[bo + k * n + j..];
                    let av4 = F32x4::splat(av);
                    a0 = a0.fma(F32x4::load(brow, 0), av4);
                    a1 = a1.fma(F32x4::load(brow, 4), av4);
                    a2 = a2.fma(F32x4::load(brow, 8), av4);
                    a3 = a3.fma(F32x4::load(brow, 12), av4);
                }
                let orow = &mut out[oo + i * n + j..];
                a0.store(orow, 0);
                a1.store(orow, 4);
                a2.store(orow, 8);
                a3.store(orow, 12);
                j += 16;
            }
            while j + 4 <= n {
                let mut acc = F32x4::splat(0.0);
                for (k, &av) in arow.iter().enumerate() {
                    acc = acc.fma(F32x4::load(&b.data, bo + k * n + j), F32x4::splat(av));
                }
                acc.store(&mut out[oo + i * n..], j);
                j += 4;
            }
            while j < n {
                let mut s = 0f32;
                for (k, &av) in arow.iter().enumerate() {
                    s += av * b.data[bo + k * n + j];
                }
                out[oo + i * n + j] = s;
                j += 1;
            }
        }
    }
    Ok(Tens::new(out_shape, out))
}

/// Gemm (rank-2): alpha·op(A)·op(B) + beta·C
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &Tens,
    b: &Tens,
    c: Option<&Tens>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tens, TaskError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(err(format!("gemm rank {:?}/{:?}", a.shape, b.shape)));
    }
    let (m, k) = if trans_a { (a.shape[1], a.shape[0]) } else { (a.shape[0], a.shape[1]) };
    let (kb, n) = if trans_b { (b.shape[1], b.shape[0]) } else { (b.shape[0], b.shape[1]) };
    if k != kb {
        return Err(err(format!("gemm K 불일치 {k}≠{kb}")));
    }
    let mut out = vec![0f32; m * n];
    if !trans_a && !trans_b {
        for i in 0..m {
            let arow = &a.data[i * k..(i + 1) * k];
            let orow = &mut out[i * n..(i + 1) * n];
            for (kk, &av) in arow.iter().enumerate() {
                let brow = &b.data[kk * n..(kk + 1) * n];
                for (o, &bv) in orow.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
            for o in orow.iter_mut() {
                *o *= alpha;
            }
        }
    } else if !trans_a && trans_b {
        // 가중치 관례 (W^T) — 2행×4열 페어링 (B 로드 재사용 2배 + 누산 8체인).
        // 1열 단일 체인은 실측 무이득이었다 (wasm Gemm 25%).
        let mut i2 = 0usize;
        while i2 + 2 <= m {
            let a0r = &a.data[i2 * k..(i2 + 1) * k];
            let a1r = &a.data[(i2 + 1) * k..(i2 + 2) * k];
            let mut j = 0usize;
            while j + 4 <= n {
                let b0 = &b.data[j * k..(j + 1) * k];
                let b1 = &b.data[(j + 1) * k..(j + 2) * k];
                let b2 = &b.data[(j + 2) * k..(j + 3) * k];
                let b3 = &b.data[(j + 3) * k..(j + 4) * k];
                let z = F32x4::splat(0.0);
                let (mut p0, mut p1, mut p2, mut p3) = (z, z, z, z);
                let (mut q0, mut q1, mut q2, mut q3) = (z, z, z, z);
                let mut kk = 0usize;
                while kk + 4 <= k {
                    let av0 = F32x4::load(a0r, kk);
                    let av1 = F32x4::load(a1r, kk);
                    let bv0 = F32x4::load(b0, kk);
                    let bv1 = F32x4::load(b1, kk);
                    let bv2 = F32x4::load(b2, kk);
                    let bv3 = F32x4::load(b3, kk);
                    p0 = p0.fma(av0, bv0);
                    p1 = p1.fma(av0, bv1);
                    p2 = p2.fma(av0, bv2);
                    p3 = p3.fma(av0, bv3);
                    q0 = q0.fma(av1, bv0);
                    q1 = q1.fma(av1, bv1);
                    q2 = q2.fma(av1, bv2);
                    q3 = q3.fma(av1, bv3);
                    kk += 4;
                }
                let mut s = [p0.sum(), p1.sum(), p2.sum(), p3.sum(),
                             q0.sum(), q1.sum(), q2.sum(), q3.sum()];
                while kk < k {
                    let (x0, x1) = (a0r[kk], a1r[kk]);
                    s[0] += x0 * b0[kk];
                    s[1] += x0 * b1[kk];
                    s[2] += x0 * b2[kk];
                    s[3] += x0 * b3[kk];
                    s[4] += x1 * b0[kk];
                    s[5] += x1 * b1[kk];
                    s[6] += x1 * b2[kk];
                    s[7] += x1 * b3[kk];
                    kk += 1;
                }
                for c4 in 0..4 {
                    out[i2 * n + j + c4] = alpha * s[c4];
                    out[(i2 + 1) * n + j + c4] = alpha * s[4 + c4];
                }
                j += 4;
            }
            while j < n {
                let brow = &b.data[j * k..(j + 1) * k];
                let s0: f32 = a0r.iter().zip(brow).map(|(&x, &y)| x * y).sum();
                let s1: f32 = a1r.iter().zip(brow).map(|(&x, &y)| x * y).sum();
                out[i2 * n + j] = alpha * s0;
                out[(i2 + 1) * n + j] = alpha * s1;
                j += 1;
            }
            i2 += 2;
        }
        for i in i2..m {
            let arow = &a.data[i * k..(i + 1) * k];
            let mut j = 0usize;
            while j + 4 <= n {
                let b0 = &b.data[j * k..(j + 1) * k];
                let b1 = &b.data[(j + 1) * k..(j + 2) * k];
                let b2 = &b.data[(j + 2) * k..(j + 3) * k];
                let b3 = &b.data[(j + 3) * k..(j + 4) * k];
                let (mut a0, mut a1, mut a2, mut a3) = (
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                    F32x4::splat(0.0),
                );
                let mut kk = 0usize;
                while kk + 4 <= k {
                    let av = F32x4::load(arow, kk);
                    a0 = a0.fma(av, F32x4::load(b0, kk));
                    a1 = a1.fma(av, F32x4::load(b1, kk));
                    a2 = a2.fma(av, F32x4::load(b2, kk));
                    a3 = a3.fma(av, F32x4::load(b3, kk));
                    kk += 4;
                }
                let (mut s0, mut s1, mut s2, mut s3) = (a0.sum(), a1.sum(), a2.sum(), a3.sum());
                while kk < k {
                    let av = arow[kk];
                    s0 += av * b0[kk];
                    s1 += av * b1[kk];
                    s2 += av * b2[kk];
                    s3 += av * b3[kk];
                    kk += 1;
                }
                out[i * n + j] = alpha * s0;
                out[i * n + j + 1] = alpha * s1;
                out[i * n + j + 2] = alpha * s2;
                out[i * n + j + 3] = alpha * s3;
                j += 4;
            }
            while j < n {
                let brow = &b.data[j * k..(j + 1) * k];
                let s: f32 = arow.iter().zip(brow).map(|(&x, &y)| x * y).sum();
                out[i * n + j] = alpha * s;
                j += 1;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    let av = if trans_a { a.data[kk * m + i] } else { a.data[i * k + kk] };
                    let bv = if trans_b { b.data[j * k + kk] } else { b.data[kk * n + j] };
                    s += av * bv;
                }
                out[i * n + j] = alpha * s;
            }
        }
    }
    let mut t = Tens::new(vec![m, n], out);
    if let Some(c) = c {
        if beta != 0.0 {
            let scaled = if beta == 1.0 {
                c.clone()
            } else {
                Tens::new(c.shape.clone(), c.data.iter().map(|&v| v * beta).collect())
            };
            t = ew_binary("Add", &t, &scaled)?;
        }
    }
    Ok(t)
}

/// Conv1d: x [1,C,W], w [O,C,k] → [1,O,W_out] (group 1)
pub fn conv1d(
    x: &Tens,
    w: &Tens,
    b: Option<&Tens>,
    pads: (usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Tens, TaskError> {
    let (c, wi) = (x.shape[1], x.shape[2]);
    let (o, wc, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if wc != c {
        return Err(err(format!("conv1d C 불일치 {wc}≠{c}")));
    }
    let span = dilation * (k - 1) + 1;
    let wi_p = wi + pads.0 + pads.1;
    let wo = (wi_p - span) / stride + 1;
    // shift-accumulate: 채널별 패딩 버퍼로 경계검사 제거, 내부 루프를 출력 연속으로
    // (자동벡터화) — 나이브 per-tap 검사 대비 ~8배 (프로파일: Conv가 hop의 86%였다)
    let mut xpad = vec![0f32; c * wi_p];
    for ic in 0..c {
        xpad[ic * wi_p + pads.0..ic * wi_p + pads.0 + wi]
            .copy_from_slice(&x.data[ic * wi..(ic + 1) * wi]);
    }
    let mut out = vec![0f32; o * wo];
    if stride == 1 {
        // ox 16블록(F32x4×4)을 **레지스터에 유지**하고 (ic,kk)를 안쪽 루프로 —
        // MAC당 메모리 접근이 x 로드 1회뿐이다. 이전 시도(kk 바깥, orow를
        // load-modify-store로 훑기)는 orow 트래픽이 지배해 무이득이었다
        // (wasm 프로파일: Conv 1.08ms/37% → 이 구조로 잡는다).
        // oc 4행 페어링: x 로드 1회가 네 출력행을 서빙 (MAC/로드 4배).
        // 레지스터: acc 16 + x 4 + w 4 = 24 — NEON 32 vreg 안, wasm은 V8이
        // 스필할 수 있어 실측으로 판정 (2행 대비 퇴행 시 cfg 분기).
        let mut oc = 0usize;
        while oc + 4 <= o {
            let biases = [
                b.map(|b| b.data[oc]).unwrap_or(0.0),
                b.map(|b| b.data[oc + 1]).unwrap_or(0.0),
                b.map(|b| b.data[oc + 2]).unwrap_or(0.0),
                b.map(|b| b.data[oc + 3]).unwrap_or(0.0),
            ];
            let wb = [oc * c * k, (oc + 1) * c * k, (oc + 2) * c * k, (oc + 3) * c * k];
            let mut ox = 0usize;
            while ox + 16 <= wo {
                let mut acc = [[F32x4::splat(0.0); 4]; 4]; // [행][x블록]
                for (row, a) in acc.iter_mut().enumerate() {
                    let bsp = F32x4::splat(biases[row]);
                    *a = [bsp, bsp, bsp, bsp];
                }
                for ic in 0..c {
                    let xr = &xpad[ic * wi_p + ox..];
                    for kk in 0..k {
                        let off = kk * dilation;
                        let x0 = F32x4::load(xr, off);
                        let x1 = F32x4::load(xr, off + 4);
                        let x2 = F32x4::load(xr, off + 8);
                        let x3 = F32x4::load(xr, off + 12);
                        for row in 0..4 {
                            let wv = F32x4::load_splat(&w.data, wb[row] + ic * k + kk);
                            acc[row][0] = acc[row][0].fma(x0, wv);
                            acc[row][1] = acc[row][1].fma(x1, wv);
                            acc[row][2] = acc[row][2].fma(x2, wv);
                            acc[row][3] = acc[row][3].fma(x3, wv);
                        }
                    }
                }
                for row in 0..4 {
                    let orow = &mut out[(oc + row) * wo + ox..];
                    acc[row][0].store(orow, 0);
                    acc[row][1].store(orow, 4);
                    acc[row][2].store(orow, 8);
                    acc[row][3].store(orow, 12);
                }
                ox += 16;
            }
            for row in 0..4 {
                for x in ox..wo {
                    let mut sv = biases[row];
                    for ic in 0..c {
                        let xr = &xpad[ic * wi_p..];
                        let wr = &w.data[wb[row] + ic * k..];
                        for kk in 0..k {
                            sv += wr[kk] * xr[x + kk * dilation];
                        }
                    }
                    out[(oc + row) * wo + x] = sv;
                }
            }
            oc += 4;
        }
        // 꼬리 행 (o % 4): 2행 페어링 경로
        while oc < o {
            let pair = oc + 1 < o;
            let bias0 = b.map(|b| b.data[oc]).unwrap_or(0.0);
            let bias1 = if pair { b.map(|b| b.data[oc + 1]).unwrap_or(0.0) } else { 0.0 };
            let wb0 = oc * c * k;
            let wb1 = (oc + 1).min(o - 1) * c * k;
            let mut ox = 0usize;
            while ox + 16 <= wo {
                let b0 = F32x4::splat(bias0);
                let b1 = F32x4::splat(bias1);
                let (mut p0, mut p1, mut p2, mut p3) = (b0, b0, b0, b0);
                let (mut q0, mut q1, mut q2, mut q3) = (b1, b1, b1, b1);
                for ic in 0..c {
                    let xr = &xpad[ic * wi_p + ox..];
                    for kk in 0..k {
                        let off = kk * dilation;
                        let x0 = F32x4::load(xr, off);
                        let x1 = F32x4::load(xr, off + 4);
                        let x2 = F32x4::load(xr, off + 8);
                        let x3 = F32x4::load(xr, off + 12);
                        let w0 = F32x4::load_splat(&w.data, wb0 + ic * k + kk);
                        p0 = p0.fma(x0, w0);
                        p1 = p1.fma(x1, w0);
                        p2 = p2.fma(x2, w0);
                        p3 = p3.fma(x3, w0);
                        if pair {
                            let w1 = F32x4::load_splat(&w.data, wb1 + ic * k + kk);
                            q0 = q0.fma(x0, w1);
                            q1 = q1.fma(x1, w1);
                            q2 = q2.fma(x2, w1);
                            q3 = q3.fma(x3, w1);
                        }
                    }
                }
                let orow = &mut out[oc * wo + ox..];
                p0.store(orow, 0);
                p1.store(orow, 4);
                p2.store(orow, 8);
                p3.store(orow, 12);
                if pair {
                    let orow = &mut out[(oc + 1) * wo + ox..];
                    q0.store(orow, 0);
                    q1.store(orow, 4);
                    q2.store(orow, 8);
                    q3.store(orow, 12);
                }
                ox += 16;
            }
            for row in 0..(1 + pair as usize) {
                let (bias, wbx) = if row == 0 { (bias0, wb0) } else { (bias1, wb1) };
                for x in ox..wo {
                    let mut sv = bias;
                    for ic in 0..c {
                        let xr = &xpad[ic * wi_p..];
                        let wr = &w.data[wbx + ic * k..];
                        for kk in 0..k {
                            sv += wr[kk] * xr[x + kk * dilation];
                        }
                    }
                    out[(oc + row) * wo + x] = sv;
                }
            }
            oc += 1 + pair as usize;
        }
    } else {
        for oc in 0..o {
            let bias = b.map(|b| b.data[oc]).unwrap_or(0.0);
            let orow = &mut out[oc * wo..(oc + 1) * wo];
            orow.fill(bias);
            for ic in 0..c {
                let xr = &xpad[ic * wi_p..(ic + 1) * wi_p];
                let wr = &w.data[(oc * c + ic) * k..(oc * c + ic + 1) * k];
                for (kk, &wv) in wr.iter().enumerate() {
                    let off = kk * dilation;
                    for (ox, o) in orow.iter_mut().enumerate() {
                        *o += wv * xr[off + ox * stride];
                    }
                }
            }
        }
    }
    Ok(Tens::new(vec![1, o, wo], out))
}

/// ConvTranspose1d: x [1,C,W], w [C,O,k] → [1,O,W_out]
pub fn conv_transpose1d(
    x: &Tens,
    w: &Tens,
    b: Option<&Tens>,
    pads: (usize, usize),
    stride: usize,
    output_padding: usize,
) -> Result<Tens, TaskError> {
    let (c, wi) = (x.shape[1], x.shape[2]);
    let (wc, o, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if wc != c {
        return Err(err(format!("convT C 불일치 {wc}≠{c}")));
    }
    let wo = (wi - 1) * stride + k + output_padding;
    let wo_final = wo - pads.0 - pads.1;
    // 출력별 gather (위치당 탭 ⌈k/s⌉개) + **프리팩**: w를 [kk][oc][ic], x를
    // [xi][ic]로 재배치해 ic 내적을 연속·벡터화 (원소 gather는 캐시미스로
    // 오히려 퇴행했었다 — 실측 0.069→0.076의 교훈)
    let mut wp = vec![0f32; k * o * c];
    for kk in 0..k {
        for oc in 0..o {
            for ic in 0..c {
                wp[(kk * o + oc) * c + ic] = w.data[(ic * o + oc) * k + kk];
            }
        }
    }
    let mut xt = vec![0f32; wi * c];
    for ic in 0..c {
        for xi in 0..wi {
            xt[xi * c + ic] = x.data[ic * wi + xi];
        }
    }
    let mut out = vec![0f32; o * wo_final];
    for oc in 0..o {
        let bias = b.map(|b| b.data[oc]).unwrap_or(0.0);
        for ox in 0..wo_final {
            let g = ox + pads.0;
            let mut acc = F32x4::splat(0.0);
            let mut tail = bias;
            let mut kk = g % stride;
            while kk < k {
                if g >= kk {
                    let xi = (g - kk) / stride;
                    if xi < wi {
                        let xr = &xt[xi * c..(xi + 1) * c];
                        let wr = &wp[(kk * o + oc) * c..(kk * o + oc + 1) * c];
                        let mut ic = 0usize;
                        while ic + 4 <= c {
                            acc = acc.fma(F32x4::load(xr, ic), F32x4::load(wr, ic));
                            ic += 4;
                        }
                        while ic < c {
                            tail += xr[ic] * wr[ic];
                            ic += 1;
                        }
                    }
                }
                kk += stride;
            }
            out[oc * wo_final + ox] = acc.sum() + tail;
        }
    }
    Ok(Tens::new(vec![1, o, wo_final], out))
}

/// Softmax (음수 축 허용)
pub fn softmax(a: &Tens, axis: isize) -> Result<Tens, TaskError> {
    let r = a.shape.len() as isize;
    let ax = if axis < 0 { (axis + r) as usize } else { axis as usize };
    if ax != a.shape.len() - 1 {
        return Err(err(format!("softmax 축 {ax} (마지막 축만)")));
    }
    let n = *a.shape.last().unwrap();
    let mut out = vec![0f32; a.data.len()];
    for (row_o, row_i) in out.chunks_exact_mut(n).zip(a.data.chunks_exact(n)) {
        let max = row_i.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let m4 = F32x4::splat(max);
        let mut acc = F32x4::splat(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let e = exp4(F32x4::load(row_i, i).sub(m4));
            e.store(row_o, i);
            acc = acc.add(e);
            i += 4;
        }
        let mut sum = acc.sum();
        for j in i..n {
            row_o[j] = fast_exp(row_i[j] - max);
            sum += row_o[j];
        }
        let inv = 1.0 / sum;
        for o in row_o.iter_mut() {
            *o *= inv;
        }
    }
    Ok(Tens::new(a.shape.clone(), out))
}

/// ONNX Slice (starts/ends/axes/steps, 음수·클램프 시맨틱)
pub fn slice(
    a: &Tens,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tens, TaskError> {
    let r = a.shape.len();
    // ⚠ 전부 i64로 계산 — ONNX는 ends에 INT64_MAX를 관례로 넣는데, wasm의
    // isize는 32비트라 `as isize` 캐스트가 −1로 뭉개져 마지막 원소가 잘린다
    // (네이티브 64비트에선 무증상 — audio-ab 게이트가 잡은 버그)
    let mut st: Vec<i64> = vec![0; r];
    let mut en: Vec<i64> = a.shape.iter().map(|&d| d as i64).collect();
    let mut sp: Vec<i64> = vec![1; r];
    for (i, &s) in starts.iter().enumerate() {
        let ax = axes.map_or(i as i64, |a| a[i]);
        let ax = if ax < 0 { (ax + r as i64) as usize } else { ax as usize };
        let d = a.shape[ax] as i64;
        let step = steps.map_or(1, |v| v[i]);
        if step <= 0 {
            return Err(err("slice 음수 step 미지원".into()));
        }
        let norm = |v: i64| -> i64 {
            if v < 0 { (v + d).clamp(0, d) } else { v.clamp(0, d) }
        };
        st[ax] = norm(s);
        en[ax] = norm(ends[i]);
        sp[ax] = step;
    }
    let out_shape: Vec<usize> = (0..r)
        .map(|d| (((en[d] - st[d]).max(0) as usize) + sp[d] as usize - 1) / sp[d] as usize)
        .collect();
    let in_s = strides(&a.shape);
    let n: usize = out_shape.iter().product::<usize>().max(1);
    // 특례: 마지막 축만 잘리고 step 1 — 행별 연속 memcpy (GRU 게이트/어텐션
    // 헤드 슬라이스가 전부 이 모양. odometer 대비 수 배)
    if r > 0
        && (0..r - 1).all(|d| st[d] == 0 && en[d] == a.shape[d] as i64 && sp[d] == 1)
        && sp[r - 1] == 1
    {
        let row_in = a.shape[r - 1];
        let row_out = out_shape[r - 1];
        let rows: usize = out_shape[..r - 1].iter().product::<usize>().max(1);
        let s0 = st[r - 1] as usize;
        let mut out = vec![0f32; n];
        for row in 0..rows {
            out[row * row_out..(row + 1) * row_out]
                .copy_from_slice(&a.data[row * row_in + s0..row * row_in + s0 + row_out]);
        }
        return Ok(Tens::new(out_shape, out));
    }
    let mut out = vec![0f32; n];
    if n > 0 {
        let step: Vec<usize> = (0..r).map(|d| sp[d] as usize * in_s[d]).collect();
        let mut src: usize = (0..r).map(|d| st[d] as usize * in_s[d]).sum();
        let mut ctr = vec![0usize; r];
        for o in out.iter_mut() {
            *o = a.data[src];
            for d in (0..r).rev() {
                ctr[d] += 1;
                src += step[d];
                if ctr[d] < out_shape[d].max(1) {
                    break;
                }
                ctr[d] = 0;
                src -= step[d] * out_shape[d].max(1);
            }
        }
    }
    Ok(Tens::new(out_shape, out))
}

/// Concat (같은 rank, axis 외 동일)
pub fn concat(inputs: &[&Tens], axis: isize) -> Result<Tens, TaskError> {
    let r = inputs[0].shape.len() as isize;
    let ax = if axis < 0 { (axis + r) as usize } else { axis as usize };
    let mut out_shape = inputs[0].shape.clone();
    out_shape[ax] = inputs.iter().map(|t| t.shape[ax]).sum();
    let outer: usize = out_shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = out_shape[ax + 1..].iter().product::<usize>().max(1);
    let mut out = vec![0f32; out_shape.iter().product::<usize>().max(1)];
    let total_ax = out_shape[ax];
    let mut off = 0usize;
    for t in inputs {
        let ta = t.shape[ax];
        for o in 0..outer {
            let src = &t.data[o * ta * inner..(o + 1) * ta * inner];
            let dst = &mut out[(o * total_ax + off) * inner..(o * total_ax + off + ta) * inner];
            dst.copy_from_slice(src);
        }
        off += ta;
    }
    Ok(Tens::new(out_shape, out))
}

/// Pad (constant, ONNX pads = [d0_b, d1_b, ..., d0_e, d1_e, ...])
pub fn pad_constant(a: &Tens, pads: &[i64], value: f32) -> Result<Tens, TaskError> {
    let r = a.shape.len();
    if pads.len() != 2 * r {
        return Err(err(format!("pads 길이 {} ≠ 2×{r}", pads.len())));
    }
    let out_shape: Vec<usize> = (0..r)
        .map(|d| a.shape[d] + pads[d] as usize + pads[r + d] as usize)
        .collect();
    let in_s = strides(&a.shape);
    let out_s = strides(&out_shape);
    let n: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![value; n];
    for idx in 0..a.data.len() {
        let mut dst = 0usize;
        for d in 0..r {
            let c = (idx / in_s[d]) % a.shape[d];
            dst += (c + pads[d] as usize) * out_s[d];
        }
        out[dst] = a.data[idx];
    }
    Ok(Tens::new(out_shape, out))
}
