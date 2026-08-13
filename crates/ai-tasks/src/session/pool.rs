//! 핸들 기반 다중모델 풀 — vision 워커가 det+lm+게이즈를 동시 상주시키는 근거.
//!
//! 핸들은 단조 증가하고 **재사용하지 않는다**: 언로드된 핸들을 늦게 쥔 호스트가
//! 엉뚱한 새 모델을 조용히 치는 사고를 원천 차단한다 (u32 오버플로는 세션당
//! 40억 로드 — 도달 불가).
//!
//! `take`/`put`은 wasm async 경계용이다: RefCell 대여를 await 너머로 들고 갈 수
//! 없어 바인딩이 항목을 꺼냈다가 되돌려 놓는다. take된 동안 같은 핸들 접근은
//! "무효 핸들"로 실패한다 — 같은 모델에 대한 동시 호출은 원래 금지다.

use std::collections::HashMap;

use crate::error::TaskError;

pub struct Pool<T> {
    items: HashMap<u32, T>,
    next: u32,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool { items: HashMap::new(), next: 1 }
    }
}

fn bad(h: u32) -> TaskError {
    TaskError::Other(format!("무효 핸들 {h} — 언로드됐거나 사용 중"))
}

impl<T> Pool<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, item: T) -> u32 {
        let h = self.next;
        self.next += 1;
        self.items.insert(h, item);
        h
    }

    pub fn get(&self, h: u32) -> Result<&T, TaskError> {
        self.items.get(&h).ok_or_else(|| bad(h))
    }

    pub fn get_mut(&mut self, h: u32) -> Result<&mut T, TaskError> {
        self.items.get_mut(&h).ok_or_else(|| bad(h))
    }

    /// async 경계용 — 반드시 같은 핸들로 `put` 할 것
    pub fn take(&mut self, h: u32) -> Result<T, TaskError> {
        self.items.remove(&h).ok_or_else(|| bad(h))
    }

    pub fn put(&mut self, h: u32, item: T) {
        self.items.insert(h, item);
    }

    pub fn remove(&mut self, h: u32) -> Result<T, TaskError> {
        self.items.remove(&h).ok_or_else(|| bad(h))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_never_reused() {
        let mut p = Pool::new();
        let a = p.insert("a");
        p.remove(a).unwrap();
        let b = p.insert("b");
        assert_ne!(a, b);
        assert!(p.get(a).is_err());
        assert_eq!(*p.get(b).unwrap(), "b");
    }

    #[test]
    fn take_blocks_until_put() {
        let mut p = Pool::new();
        let h = p.insert(1);
        let v = p.take(h).unwrap();
        assert!(p.get(h).is_err());
        p.put(h, v);
        assert_eq!(*p.get(h).unwrap(), 1);
    }
}
