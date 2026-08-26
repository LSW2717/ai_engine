//! clap_tune — 덤프(clap_dump) 리플레이 그리드 서치.
//!
//! GPU 없이 GestureClassifier(ClapTuning)를 스윕해 정답과의 오차를 최소화한다.
//! 네거티브 영상 오발화는 하드 제약(1건이라도 있으면 탈락).
//!
//! 사용: cargo run --release -p ai-bench --example clap_tune -- <dump_dir> <prefix>
//!   덤프 파일명: <dump_dir>/<prefix>_<n>.txt  (n = 1..=16)

use ai_tasks::features::hand::gesture::{
    ClapTuning, Gesture, GestureClassifier, Handedness, HandSnapshot,
};

/// 박수1~16 정답 (사용자 제공: 11·12는 네거티브)
const GT: [usize; 16] = [10, 20, 0, 3, 0, 4, 3, 0, 0, 0, 0, 0, 23, 10, 5, 15];

type Ticks = Vec<(f64, Vec<HandSnapshot>)>;

fn load_dump(path: &str) -> Ticks {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut ticks = Vec::new();
    for line in text.lines() {
        let mut cells = line.split('|');
        let t: f64 = cells.next().unwrap().parse().expect("t_ms");
        let mut hands = Vec::new();
        for cell in cells {
            let vals: Vec<f32> = cell.split(',').map(|v| v.parse().expect("f32")).collect();
            assert_eq!(vals.len(), 44, "hand cell len");
            let mut lmk = [[0f32; 2]; 21];
            for i in 0..21 {
                lmk[i] = [vals[2 + i * 2], vals[3 + i * 2]];
            }
            let handedness =
                if vals[0] > 0.5 { Handedness::Left } else { Handedness::Right };
            hands.push(HandSnapshot { landmarks: lmk, handedness, presence: vals[1] });
        }
        ticks.push((t, hands));
    }
    ticks
}

fn count_claps(t: ClapTuning, ticks: &Ticks) -> usize {
    let mut cls = GestureClassifier::with_clap_tuning(t);
    let mut n = 0;
    for (ts, hands) in ticks {
        for ev in cls.classify(hands, *ts) {
            if ev.gesture == Gesture::Clap {
                n += 1;
            }
        }
    }
    n
}

/// 스윕 축 — 이름과 후보값. 이름은 apply()의 match와 일치해야 한다.
struct Axis {
    name: &'static str,
    vals: &'static [f64],
}

fn apply(t: &mut ClapTuning, name: &str, v: f64) {
    let f = v as f32;
    match name {
        "close_threshold" => t.close_threshold = f,
        "rearm_d" => t.rearm_d = f,
        "cooldown_ms" => t.cooldown_ms = v,
        "fuse_window_ms" => t.fuse_window_ms = v,
        "mid_tol" => t.mid_tol = f,
        "sep_md" => t.sep_md = f,
        "rebound_ms" => t.rebound_ms = v,
        "strong_d" => t.strong_d = f,
        "strong_md" => t.strong_md = f,
        "sep_cd" => t.sep_cd = f,
        "axis_up_max_deg" => t.axis_up_max_deg = f,
        "axis_pair_diff_max" => t.axis_pair_diff_max = f,
        "fire_min_presence" => t.fire_min_presence = f,
        "same_label_need_horizontal" => t.same_label_need_horizontal = v > 0.5,
        "hop_need_horizontal" => t.hop_need_horizontal = v > 0.5,
        "seed_abs_cd_max" => t.seed_abs_cd_max = f,
        "approach_conv_min" => t.approach_conv_min = f,
        "approach_win_ms" => t.approach_win_ms = v,
        "entry_win_ms" => t.entry_win_ms = v,
        "session_ms" => t.session_ms = v,
        "entry_persist_ms" => t.entry_persist_ms = v,
        "entry_confirm_ms" => t.entry_confirm_ms = v,
        "rebound_disp_max" => t.rebound_disp_max = f,
        _ => panic!("미지 축: {name}"),
    }
}

fn fire_times(t: ClapTuning, ticks: &Ticks) -> Vec<f64> {
    let mut cls = GestureClassifier::with_clap_tuning(t);
    let mut out = Vec::new();
    for (ts, hands) in ticks {
        for ev in cls.classify(hands, *ts) {
            if ev.gesture == Gesture::Clap {
                out.push(ev.ts_ms);
            }
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("사용: clap_tune <dump_dir> <prefix> [detail]");
    let prefix = args.next().expect("prefix");
    let detail = args.next().is_some();

    let dumps: Vec<Ticks> =
        (1..=16).map(|i| load_dump(&format!("{dir}/{prefix}_{i}.txt"))).collect();

    if detail {
        // v5 base 우승 조합 — CLAP_T_<축이름> env로 개별 오버라이드
        let mut t = ClapTuning::default();
        for (n, v) in [
            ("cooldown_ms", 200.0),
            ("rebound_ms", 350.0),
            ("sep_md", 0.4),
            ("sep_cd", 1.35),
            ("strong_md", 1.4),
            ("rearm_d", 0.4),
            ("close_threshold", 0.3),
            ("mid_tol", 1.3),
            ("fuse_window_ms", 600.0),
            ("axis_up_max_deg", 15.0),
            ("axis_pair_diff_max", 1e9),
            ("seed_abs_cd_max", 0.30),
            ("fire_min_presence", 0.0),
            ("same_label_need_horizontal", 0.0),
            ("hop_need_horizontal", 0.0),
            ("entry_persist_ms", 200.0),
        ] {
            let v = std::env::var(format!("CLAP_T_{n}"))
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(v);
            apply(&mut t, n, v);
        }
        for (i, d) in dumps.iter().enumerate() {
            let ft = fire_times(t, d);
            println!(
                "박수{}: {}회  ({})",
                i + 1,
                ft.len(),
                ft.iter().map(|m| format!("{:.2}s", m / 1000.0)).collect::<Vec<_>>().join(", ")
            );
        }
        return;
    }

    // 스윕 공간 v6 — 확정 후보값 미니 스윕 (상호작용 검증)
    const AXES: &[Axis] = &[
        Axis { name: "cooldown_ms", vals: &[200.0] },
        Axis { name: "rebound_ms", vals: &[350.0] },
        Axis { name: "sep_md", vals: &[0.4] },
        Axis { name: "sep_cd", vals: &[1.35, 1.5] },
        Axis { name: "strong_md", vals: &[1.4] },
        Axis { name: "rearm_d", vals: &[0.4, 0.55] },
        Axis { name: "mid_tol", vals: &[1.3] },
        Axis { name: "fuse_window_ms", vals: &[600.0] },
        Axis { name: "axis_up_max_deg", vals: &[15.0, 30.0] },
        Axis { name: "axis_pair_diff_max", vals: &[15.0, 1e9] },
        Axis { name: "seed_abs_cd_max", vals: &[0.30] },
        Axis { name: "fire_min_presence", vals: &[0.0] },
        Axis { name: "entry_persist_ms", vals: &[200.0] },
        Axis { name: "hop_need_horizontal", vals: &[0.0, 1.0] },
    ];

    let mut idx = vec![0usize; AXES.len()];
    let total: usize = AXES.iter().map(|a| a.vals.len()).product();
    eprintln!("# 조합 {total}개 × 16영상");

    // (점수, 벡터, 튜닝) — 점수 = Σ|검출-정답| (양성), 네거티브 오발화는 탈락
    let mut best: Vec<(usize, Vec<usize>, ClapTuning)> = Vec::new();
    // 진단 — 오발화 포함 최선 (전멸 시 원인 표시용): (neg_fp, err, counts)
    let mut best_fp: Option<(usize, usize, Vec<usize>)> = None;
    let mut done = 0usize;
    loop {
        let mut t = ClapTuning::default();
        for (a, &i) in AXES.iter().zip(&idx) {
            apply(&mut t, a.name, a.vals[i]);
        }
        let counts: Vec<usize> = dumps.iter().map(|d| count_claps(t, d)).collect();
        let neg_fp = counts
            .iter()
            .zip(&GT)
            .filter(|&(_, &g)| g == 0)
            .map(|(&c, _)| c)
            .sum::<usize>();
        let err: usize = counts.iter().zip(&GT).map(|(&c, &g)| c.abs_diff(g)).sum();
        if neg_fp == 0 {
            best.push((err, counts, t));
            best.sort_by_key(|b| b.0);
            best.truncate(10);
        } else if best_fp.as_ref().is_none_or(|b| (neg_fp, err) < (b.0, b.1)) {
            best_fp = Some((neg_fp, err, counts));
        }
        done += 1;
        if done % 2000 == 0 {
            eprintln!("# {done}/{total}");
        }
        // 오도미터 증가
        let mut k = 0;
        loop {
            idx[k] += 1;
            if idx[k] < AXES[k].vals.len() {
                break;
            }
            idx[k] = 0;
            k += 1;
            if k == AXES.len() {
                // 종료 — 결과 출력
                println!("GT      : {GT:?}");
                for (err, counts, t) in &best {
                    println!("err={err:3}  {counts:?}");
                    println!("  {t:?}");
                }
                if best.is_empty() {
                    if let Some((fp, err, counts)) = &best_fp {
                        println!("전멸 — 최소 오발화 조합: neg_fp={fp} err={err} {counts:?}");
                    }
                }
                return;
            }
        }
    }
}
