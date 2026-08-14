//! 손조립 그래프 E2E — Model(계획·슬롯 재사용·뷰) vs CpuExec(순진 오라클).
//! 컨테이너 직렬화 왕복까지 실전 로드 경로 그대로 태운다.
//!
//! 그래프: conv(3→8) → dw(8) → mul 0.5 → resize 2× (+ conv 출력의 채널 뷰에 Act)
//! — alias 읽기, liveness 해제, 에필로그가 전부 걸리는 최소 구성.

use ai_convert::verify::CpuExec;
use ai_core::format::{SwAlias, SwModel, SwOp, SwOperand, SwSize, SwTensor, WRef};
use ai_core::ops::{BinaryOp, CoordMode};
use ai_core::rng::XorShift32;
use ai_core::{pack, Activation, DType};

fn tensor(name: &str, h: u32, w: u32, c: u32, last_use: u32) -> SwTensor {
    SwTensor { name: name.into(), h, w, c, dt: DType::F32, alias: None, last_use }
}

#[test]
fn synthetic_graph_matches_cpuexec() {
    let mut rng = XorShift32::new(42);
    let (h, w) = (6u32, 6u32);

    // 가중치 블롭 (컨테이너 규약대로 사전 패킹)
    let w0 = rng.vec_f32(8 * 3 * 3 * 3);
    let b0 = rng.vec_f32(8);
    let w1 = rng.vec_f32(8 * 3 * 3);
    let b1 = rng.vec_f32(8);
    let (w0_bytes, kg0) = pack::pack_weights_conv(&w0, 8, 3, 3, 3, 4, DType::F32);
    let b0_bytes = pack::pack_bias(&b0, 8, DType::F32);
    let w1_bytes = pack::pack_weights_dw(&w1, 8, 3, 3, DType::F32);
    let b1_bytes = pack::pack_bias(&b1, 8, DType::F32);

    let mut blob = Vec::new();
    let mut wref = |bytes: &[u8], blob: &mut Vec<u8>| {
        let off = blob.len() as u64;
        blob.extend_from_slice(bytes);
        WRef { off, len: bytes.len() as u64 }
    };
    let w0_ref = wref(&w0_bytes, &mut blob);
    let b0_ref = wref(&b0_bytes, &mut blob);
    let w1_ref = wref(&w1_bytes, &mut blob);
    let b1_ref = wref(&b1_bytes, &mut blob);

    let mut t_view = tensor("t1_view", h, w, 4, 3);
    t_view.alias = Some(SwAlias { of: 1, cg_off: 1 });

    let sw = SwModel {
        name: "synthetic".into(),
        size: SwSize { h, w },
        dt_default: DType::F32,
        dt_weights: None,
        tensors: vec![
            tensor("x", h, w, 3, 0),        // 0
            tensor("t1", h, w, 8, 3),       // 1 — op1 + op3(뷰 경유)이 읽는다
            tensor("t2", h, w, 8, 2),       // 2
            tensor("t3", h, w, 8, 4),       // 3
            t_view,                          // 4 — t1의 채널 4..8 뷰
            tensor("act_out", h, w, 4, 3),  // 5 (출력)
            tensor("resized", h * 2, w * 2, 8, 4), // 6 (출력)
        ],
        inputs: vec![0],
        outputs: vec![5, 6],
        states: vec![],
        consts: vec![],
        ops: vec![
            SwOp::Conv {
                input: 0, out: 1, srcs: vec![], res: None,
                cin: 3, cout: 8, kh: 3, kw: 3, sh: 1, sw: 1,
                pad: [1; 4], d: 1, groups: 1, act: Activation::Relu,
                w: w0_ref, b: b0_ref, kg_pad: kg0,
            },
            SwOp::Conv {
                input: 1, out: 2, srcs: vec![], res: None,
                cin: 8, cout: 8, kh: 3, kw: 3, sh: 1, sw: 1,
                pad: [1; 4], d: 1, groups: 8, act: Activation::Hardswish,
                w: w1_ref, b: b1_ref, kg_pad: 0,
            },
            SwOp::Binary {
                a: 2,
                b: SwOperand::Scalar { v: 0.5, first: false },
                out: 3,
                op: BinaryOp::Mul,
                act: Activation::None,
            },
            SwOp::Act { input: 4, out: 5, act: Activation::Sigmoid },
            SwOp::Resize {
                input: 3, out: 6, srcs: vec![],
                oh: h * 2, ow: w * 2,
                mode: CoordMode::HalfPixel,
            },
        ],
    };

    let input = XorShift32::new(7).vec_f32((h * w * 3) as usize);

    // 오라클
    let mut oracle = CpuExec::new(&sw, &blob);
    oracle.set_input(0, input.clone());
    oracle.run().unwrap();
    let want_act = oracle.read(5).unwrap();
    let want_resized = oracle.read(6).unwrap();

    // 실전 경로: 컨테이너 왕복 → Model
    let container = sw.write_container(&blob).unwrap();
    let mut m = ai_cpu::Model::load(&container).unwrap();
    m.set_input("x", &input).unwrap();
    m.infer().unwrap();
    let got_act = m.read_output("act_out").unwrap();
    let got_resized = m.read_output("resized").unwrap();

    let check = |want: &[f32], got: &[f32], tag: &str| {
        assert_eq!(want.len(), got.len(), "{tag} 길이");
        let mut max_err = 0f32;
        for (a, g) in want.iter().zip(got) {
            max_err = max_err.max((a - g).abs() / a.abs().max(1.0));
        }
        assert!(max_err <= 1e-4, "{tag} max_err {max_err}");
    };
    check(&want_act, &got_act, "act_out");
    check(&want_resized, &got_resized, "resized");

    // 두 번째 프레임(같은 입력)도 동일해야 한다 — 슬롯 재사용·상태 없음 확인
    m.infer().unwrap();
    check(&want_resized, &m.read_output("resized").unwrap(), "resized(2프레임)");
}
