//! liveness 슬롯 배정 — greedy interval, 크기별 free-list.
//!
//! 재사용 조건: 텐서의 last_use < 새 텐서의 first_def (엄격히 이전) —
//! 같은 dispatch 안에서 입력·출력이 같은 버퍼를 공유하는 일이 구조적으로 불가능하다
//! (WebGPU가 버퍼 단위 usage scope로 금지하는 바로 그 상황).

use std::collections::HashMap;

/// 실행 항목의 (입력 tid들, 출력 tid) — 실체화 op 포함 증강 리스트 기준
pub struct LiveOp {
    pub reads: Vec<u32>,
    pub writes: Vec<u32>,
}

pub struct SlotPlan {
    /// tid → 슬롯 인덱스 (storage가 필요한 tid만; 순수 rename 별칭은 호출자가 백킹으로 해석)
    pub slot_of: HashMap<u32, usize>,
    /// 슬롯 크기(바이트)
    pub slot_sizes: Vec<u64>,
}

/// `persistent` = 수명 무한(내 슬롯을 남이 재사용 금지) tid — 입출력.
/// `dedicated` = **전용 슬롯** tid — 남의 죽은 슬롯도 받으면 안 됨. 상태 텐서가 여기다:
/// 상태는 프레임을 넘어 값을 보존하므로 그래프 내 수명(생산 op부터)이 실제 수명이 아니다.
/// 죽은 중간 텐서의 슬롯을 물려받으면, 다음 프레임에서 그 중간 텐서의 생산자가
/// 지난 프레임의 상태를 읽기 전에 덮어쓴다.
/// `size_of` = tid → 바이트 (storage 필요 tid만 조회됨).
pub fn assign_slots(
    ops: &[LiveOp],
    storage_tids: &[u32],
    persistent: &[u32],
    dedicated: &[u32],
    size_of: impl Fn(u32) -> u64,
) -> SlotPlan {
    // first_def / last_use
    let mut first_def: HashMap<u32, usize> = HashMap::new();
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for &t in &op.writes {
            first_def.entry(t).or_insert(i);
        }
        for &t in &op.reads {
            last_use.insert(t, i);
        }
    }
    for &t in persistent {
        last_use.insert(t, usize::MAX);
    }

    let mut plan = SlotPlan { slot_of: HashMap::new(), slot_sizes: Vec::new() };
    let mut free: HashMap<u64, Vec<usize>> = HashMap::new();
    // 해제 예정 목록: op index → 그 시점 이후 재사용 가능해지는 tid들
    let mut expiring: HashMap<usize, Vec<u32>> = HashMap::new();
    for &t in storage_tids {
        if let Some(&lu) = last_use.get(&t) {
            if lu != usize::MAX {
                expiring.entry(lu).or_default().push(t);
            }
        }
    }

    // 입력처럼 op이 생산하지 않는 tid + 전용(상태) tid → 즉시 새 슬롯 배정
    for &t in storage_tids {
        if !first_def.contains_key(&t) || dedicated.contains(&t) {
            if plan.slot_of.contains_key(&t) {
                continue;
            }
            let size = size_of(t);
            let slot = plan.slot_sizes.len();
            plan.slot_sizes.push(size);
            plan.slot_of.insert(t, slot);
        }
    }

    for (i, op) in ops.iter().enumerate() {
        // i 이전에 수명이 끝난 텐서의 슬롯 반환 (last_use < i 조건은 직전 op까지 처리됨)
        if i > 0 {
            if let Some(done) = expiring.remove(&(i - 1)) {
                for t in done {
                    if let Some(&slot) = plan.slot_of.get(&t) {
                        free.entry(plan.slot_sizes[slot]).or_default().push(slot);
                    }
                }
            }
        }
        for &t in &op.writes {
            if plan.slot_of.contains_key(&t) || !storage_tids.contains(&t) {
                continue;
            }
            let size = size_of(t);
            let slot = match free.get_mut(&size).and_then(|v| v.pop()) {
                Some(s) => s,
                None => {
                    plan.slot_sizes.push(size);
                    plan.slot_sizes.len() - 1
                }
            };
            plan.slot_of.insert(t, slot);
        }
    }

    // 검증: 같은 슬롯의 두 텐서는 수명이 겹치면 안 된다
    #[cfg(debug_assertions)]
    {
        let interval = |t: u32| -> (usize, usize) {
            (
                first_def.get(&t).copied().unwrap_or(0),
                last_use.get(&t).copied().unwrap_or(0),
            )
        };
        let mut by_slot: HashMap<usize, Vec<u32>> = HashMap::new();
        for (&t, &s) in &plan.slot_of {
            by_slot.entry(s).or_default().push(t);
        }
        for (slot, tids) in by_slot {
            for (ai, &a) in tids.iter().enumerate() {
                for &b in &tids[ai + 1..] {
                    let (ad, au) = interval(a);
                    let (bd, bu) = interval(b);
                    let disjoint = au < bd || bu < ad;
                    assert!(
                        disjoint,
                        "슬롯 {slot} 수명 겹침: tid {a} [{ad},{au}] vs tid {b} [{bd},{bu}]"
                    );
                }
            }
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_expired_slots_but_not_persistent() {
        // op0: in(0) → 1 ; op1: 1 → 2 ; op2: 2 → 3(out)
        // 1은 op1 이후 죽음 → 3이 재사용 가능... 단 3은 persistent
        let ops = vec![
            LiveOp { reads: vec![0], writes: vec![1] },
            LiveOp { reads: vec![1], writes: vec![2] },
            LiveOp { reads: vec![2], writes: vec![3] },
        ];
        let plan = assign_slots(&ops, &[0, 1, 2, 3], &[0, 3], &[], |_| 1024);
        // 1(마지막 사용 op1)의 슬롯을 op2의 출력이... 3은 persistent지만 크기 같으면 재사용 가능?
        // persistent는 last_use=MAX일 뿐 배정은 동일 — 재사용 "당하지" 않는 게 핵심.
        let s1 = plan.slot_of[&1];
        let s3 = plan.slot_of[&3];
        assert_eq!(s1, s3, "죽은 1의 슬롯을 3이 재사용해야");
        // 0은 persistent — 누구도 재사용 못함
        let s0 = plan.slot_of[&0];
        assert!(s0 != plan.slot_of[&2] && s0 != s1);
        assert_eq!(plan.slot_sizes.len(), 3); // 4 텐서 → 3 슬롯
    }

    #[test]
    fn no_same_op_aliasing() {
        // op0 reads 0 writes 1 — 0의 last_use가 op0이면 1은 0의 슬롯을 못 씀
        let ops = vec![LiveOp { reads: vec![0], writes: vec![1] }];
        let plan = assign_slots(&ops, &[0, 1], &[], &[], |_| 64);
        assert_ne!(plan.slot_of[&0], plan.slot_of[&1]);
    }
}
