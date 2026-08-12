//! .sw JSON 그래프 스키마 — 변환기(ai-convert)와 런타임(ai-runtime)의 공유 계약.
//!
//! 원칙: **런타임은 그래프 분석을 하지 않는다.** 변환기가 모든 것을 사전 계산한다 —
//! 토폴로지 정렬, 융합 결정, 뷰(alias) 판정, last_use(수명), 가중치 오프셋(사전 패킹).
//!
//! 뷰(alias)는 op가 아니라 **텐서 속성**이다: 항등 Resize/Expand/풀범위 Slice/
//! split-concat 상쇄/4배수 Split 파트가 전부 `alias {of, cg_off}`로 표현된다.
//! cg 정렬 시작 + 비4배수 채널 수의 뷰도 합법 — 모든 커널이 lane-safe(가중치
//! 제로패딩, 패딩 lane은 unpack에서 폐기)하기 때문이다.

use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::ops::{BinaryOp, CoordMode};
use crate::tensor::DType;

fn default_dil() -> u32 {
    1
}

/// 블롭 안의 바이트 구간 (off는 BLOB_ALIGN 배수)
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct WRef {
    pub off: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct SwAlias {
    /// 백킹 텐서 tid
    pub of: u32,
    /// 채널그룹(vec4) 단위 오프셋
    pub cg_off: u32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SwTensor {
    pub name: String,
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub dt: DType,
    #[serde(default)]
    pub alias: Option<SwAlias>,
    /// 이 텐서를 읽는 마지막 op 인덱스 (liveness — 뷰는 백킹 텐서 수명을 연장한 값)
    pub last_use: u32,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct SwState {
    /// 그래프 입력 tid (예: r1i)
    pub input: u32,
    /// 그래프 출력 tid (예: r1o) — 프레임 간 ping-pong 쌍
    pub output: u32,
}

/// binary의 두 번째 피연산자 — ai-gpu elementwise::EwOperand와 1:1
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum SwOperand {
    /// 같은 크기 텐서
    Tensor { tid: u32 },
    /// 스칼라 상수 (params 경유 — 재컴파일 없음). first = scalar∘tensor 순서
    Scalar { v: f32, first: bool },
    /// [1,1,C] 채널 벡터 상수 (블롭에 pack_nhwc 레이아웃으로 저장)
    Cvec { w: WRef, c: u32 },
    /// [1,1,C] 채널 벡터 런타임 텐서 (SE 게이트)
    CvecTensor { tid: u32 },
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SwOp {
    Conv {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        /// concat-into-conv 융합: 입력이 채널 concat의 파트들일 때 (그룹 정렬 필수).
        /// 비어 있으면 단일 입력(input). 채워져 있으면 input == srcs[0].input.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        srcs: Vec<SwConcatPart>,
        /// residual 에필로그 소스 (활성화 후 더해짐 — conv-tail 규약)
        #[serde(default)]
        res: Option<u32>,
        cin: u32,
        cout: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        pad: [u32; 4],
        /// 등방 dilation (기본 1)
        #[serde(default = "default_dil")]
        d: u32,
        /// 1 = 일반/pw, cin(=cout) = depthwise
        groups: u32,
        act: Activation,
        /// pack_weights_conv(kg_align=4) 또는 pack_weights_dw 레이아웃
        w: WRef,
        /// pack_bias — 항상 존재 (ONNX conv가 bias 없으면 0 벡터)
        b: WRef,
        /// 패킹 시 사용된 패딩 kg (일반/pw만 의미, dw는 0)
        kg_pad: u32,
    },
    Binary {
        a: u32,
        b: SwOperand,
        out: u32,
        op: BinaryOp,
        act: Activation,
    },
    Gpool {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
    },
    Avgpool {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        pad: [u32; 4],
    },
    Resize {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        /// concat-into-resize 융합 파트 (Conv.srcs와 동일 규약 — 그룹 정렬 필수)
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        srcs: Vec<SwConcatPart>,
        oh: u32,
        ow: u32,
        mode: CoordMode,
    },
    /// 채널 축 연결 — v1은 실제 복사 op (parts는 out의 채널 오프셋 순서)
    Concat {
        out: u32,
        parts: Vec<SwConcatPart>,
    },
    /// 서브 vec4 채널 슬라이스 (src_c부터 n채널)
    Chcopy {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        src_c: u32,
        n: u32,
    },
    /// 단독 활성화
    Act {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        act: Activation,
    },
    /// GRU mix(a,b,z) = a + z*(b-a) — fuse_mix 패스가 sub/mul/mul/add 체인에서 방출
    Mix {
        z: u32,
        a: u32,
        b: u32,
        out: u32,
    },
    /// SE 게이트: gpool(채널 평균) → FC1(act1) [→ FC2] 를 한 디스패치로 (fuse_se).
    /// 출력은 [1,1,C] 벡터 텐서 — 후속 cvec-mul이 게이트로 읽는다.
    SeGate {
        #[serde(rename = "in")]
        input: u32,
        out: u32,
        /// FC1 출력 채널 (fc2 없으면 곧 최종 채널)
        c_mid: u32,
        act1: Activation,
        /// pack_weights_conv(kg_align=4) 레이아웃 (1×1 conv과 동일)
        w1: WRef,
        b1: WRef,
        #[serde(default)]
        fc2: Option<SeFc>,
    },
}

/// SeGate의 두 번째 FC
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct SeFc {
    pub c_out: u32,
    pub act: Activation,
    pub w: WRef,
    pub b: WRef,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct SwConcatPart {
    #[serde(rename = "in")]
    pub input: u32,
    /// 이 파트의 논리 채널 수
    pub c: u32,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct SwSize {
    pub h: u32,
    pub w: u32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SwModel {
    pub name: String,
    /// 변환 시점 입력 크기 (정보용)
    pub size: SwSize,
    pub dt_default: DType,
    /// index = tid
    pub tensors: Vec<SwTensor>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    #[serde(default)]
    pub states: Vec<SwState>,
    /// 토폴로지 정렬 완료 상태
    pub ops: Vec<SwOp>,
}

impl SwModel {
    pub fn to_json(&self) -> Result<Vec<u8>, super::header::FormatError> {
        serde_json::to_vec(self).map_err(|e| super::header::FormatError::Json(e.to_string()))
    }

    pub fn from_json(json: &[u8]) -> Result<Self, super::header::FormatError> {
        serde_json::from_slice(json).map_err(|e| super::header::FormatError::Json(e.to_string()))
    }

    /// 컨테이너 전체 직렬화
    pub fn write_container(&self, blob: &[u8]) -> Result<Vec<u8>, super::header::FormatError> {
        Ok(super::header::write_container(&self.to_json()?, blob))
    }

    /// 컨테이너 파싱 → (모델, 블롭 슬라이스)
    pub fn parse_container(bytes: &[u8]) -> Result<(Self, &[u8]), super::header::FormatError> {
        let (json, blob) = super::header::parse_container(bytes)?;
        Ok((Self::from_json(json)?, blob))
    }

    /// alias 체인 해석: tid → (백킹 tid, 누적 cg_off)
    pub fn resolve_alias(&self, mut tid: u32) -> (u32, u32) {
        let mut cg_off = 0;
        while let Some(a) = self.tensors[tid as usize].alias {
            cg_off += a.cg_off;
            tid = a.of;
        }
        (tid, cg_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::header;

    fn tiny_model() -> SwModel {
        SwModel {
            name: "t".into(),
            size: SwSize { h: 4, w: 4 },
            dt_default: DType::F32,
            tensors: vec![
                SwTensor { name: "x".into(), h: 4, w: 4, c: 3, dt: DType::F32, alias: None, last_use: 0 },
                SwTensor { name: "y".into(), h: 4, w: 4, c: 8, dt: DType::F32, alias: None, last_use: 1 },
                SwTensor {
                    name: "y_view".into(),
                    h: 4,
                    w: 4,
                    c: 4,
                    dt: DType::F32,
                    alias: Some(SwAlias { of: 1, cg_off: 1 }),
                    last_use: 1,
                },
            ],
            inputs: vec![0],
            outputs: vec![2],
            states: vec![],
            ops: vec![SwOp::Conv {
                input: 0,
                out: 1,
                srcs: vec![],
                res: None,
                cin: 3,
                cout: 8,
                kh: 1,
                kw: 1,
                sh: 1,
                sw: 1,
                pad: [0; 4],
                d: 1,
                groups: 1,
                act: Activation::Relu,
                w: WRef { off: 0, len: 512 },
                b: WRef { off: 512, len: 32 },
                kg_pad: 4,
            }],
        }
    }

    #[test]
    fn container_roundtrip() {
        let m = tiny_model();
        let blob = vec![7u8; 544];
        let bytes = m.write_container(&blob).unwrap();
        let (m2, blob2) = SwModel::parse_container(&bytes).unwrap();
        assert_eq!(m, m2);
        assert_eq!(blob2, &blob[..]);
        // blob 정렬 확인
        let blob_off = bytes.len() - blob.len();
        assert_eq!(blob_off % header::BLOB_ALIGN as usize, 0);
    }

    #[test]
    fn alias_chain_resolution() {
        let m = tiny_model();
        assert_eq!(m.resolve_alias(2), (1, 1));
        assert_eq!(m.resolve_alias(0), (0, 0));
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let m = tiny_model();
        let mut bytes = m.write_container(&[]).unwrap();
        bytes[0] = b'X';
        assert!(matches!(
            SwModel::parse_container(&bytes),
            Err(header::FormatError::BadMagic)
        ));
        bytes[0] = b'S';
        bytes[4] = 99;
        assert!(matches!(
            SwModel::parse_container(&bytes),
            Err(header::FormatError::BadVersion(99))
        ));
    }

    /// 스키마 확장성: 미지 필드 무시 (v1 리더가 v1.x 파일을 읽을 수 있어야)
    #[test]
    fn unknown_json_fields_ignored() {
        let m = tiny_model();
        let mut v: serde_json::Value = serde_json::from_slice(&m.to_json().unwrap()).unwrap();
        v["future_field"] = serde_json::json!({"x": 1});
        let m2 = SwModel::from_json(serde_json::to_string(&v).unwrap().as_bytes()).unwrap();
        assert_eq!(m, m2);
    }
}
