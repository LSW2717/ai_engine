//! 오라클 .npy 디렉토리 vs CPU 실행 결과 비교 — 이름 매칭, 최초 발산 보고.

use std::collections::HashMap;
use std::path::Path;

use crate::error::ConvertError;
use crate::verify::npy;
use crate::verify::CpuExec;

pub struct TensorReport {
    pub name: String,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub passed: bool,
}

pub struct Manifest {
    /// ONNX 이름 → npy 상대경로
    pub outputs: HashMap<String, String>,
    pub intermediates: HashMap<String, String>,
    pub inputs: HashMap<String, String>,
}

pub fn load_manifest(dir: &Path) -> Result<Manifest, ConvertError> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)
            .map_err(|e| ConvertError::Other(e.to_string()))?;
    let get = |k: &str| -> HashMap<String, String> {
        v[k].as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(name, e)| {
                        e["file"].as_str().map(|f| (name.clone(), f.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    Ok(Manifest { outputs: get("outputs"), intermediates: get("intermediates"), inputs: get("inputs") })
}

/// 오라클 NCHW npy 로드 → NHWC
pub fn load_nhwc(dir: &Path, file: &str) -> Result<(Vec<usize>, Vec<f32>), ConvertError> {
    let (shape, data) = npy::read_npy_f32(&std::fs::read(dir.join(file))?)
        .map_err(ConvertError::Other)?;
    match shape.len() {
        4 => {
            let (c, h, w) = (shape[1], shape[2], shape[3]);
            Ok((shape, npy::nchw_to_nhwc(&data, c, h, w)))
        }
        _ => Ok((shape, data)),
    }
}

/// 실행 완료된 CpuExec의 텐서들을 오라클과 이름 매칭으로 비교.
/// `skip` = 융합으로 값 의미가 바뀐 이름 (Graph::semantic_changed).
pub fn compare_all(
    exec: &CpuExec,
    dir: &Path,
    manifest: &Manifest,
    skip: &std::collections::HashSet<String>,
    atol: f32,
    rtol: f32,
) -> Result<Vec<TensorReport>, ConvertError> {
    let mut reports = Vec::new();
    let all: Vec<(&String, &String)> =
        manifest.outputs.iter().chain(manifest.intermediates.iter()).collect();

    for (t_idx, t) in exec.model.tensors.iter().enumerate() {
        if skip.contains(&t.name) {
            continue;
        }
        let Some((_, file)) = all.iter().find(|(n, _)| **n == t.name) else { continue };
        let got = match exec.read(t_idx as u32) {
            Ok(g) => g,
            Err(_) => continue, // 미계산(융합으로 사라진 이름 등)
        };
        let (_, want) = load_nhwc(dir, file)?;
        if want.len() != got.len() {
            reports.push(TensorReport {
                name: t.name.clone(),
                max_abs: f32::INFINITY,
                mean_abs: f32::INFINITY,
                passed: false,
            });
            continue;
        }
        let mut max_abs = 0f32;
        let mut sum = 0f64;
        let mut passed = true;
        for (g, w) in got.iter().zip(&want) {
            let e = (g - w).abs();
            max_abs = max_abs.max(e);
            sum += e as f64;
            if !(e <= atol + rtol * w.abs()) {
                passed = false;
            }
        }
        reports.push(TensorReport {
            name: t.name.clone(),
            max_abs,
            mean_abs: (sum / want.len() as f64) as f32,
            passed,
        });
    }
    Ok(reports)
}
