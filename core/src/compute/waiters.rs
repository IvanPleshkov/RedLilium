use std::sync::Mutex;
use std::task::Waker;

// All waiter state lives in the object, so registration and notification can
// cross dylib boundaries. Never invoke or drop an arbitrary waker under a lock.
#[derive(Default)]
pub(super) struct Waiters(Mutex<(u64, Vec<(u64, Waker)>)>);
impl Waiters {
    pub fn register(&self, key: &mut Option<u64>, waker: &Waker) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let id = *key.get_or_insert_with(|| {
            state.0 += 1;
            state.0
        });
        let old = if let Some((_, stored)) = state.1.iter_mut().find(|(k, _)| *k == id) {
            if stored.will_wake(waker) {
                return;
            }
            Some(std::mem::replace(stored, waker.clone()))
        } else {
            state.1.push((id, waker.clone()));
            None
        };
        drop(state);
        drop(old);
    }
    pub fn remove(&self, key: Option<u64>) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let old = key
            .and_then(|id| state.1.iter().position(|(k, _)| *k == id))
            .map(|i| state.1.swap_remove(i));
        drop(state);
        drop(old);
    }
    pub fn wake_all(&self) {
        let waiters = std::mem::take(&mut self.0.lock().unwrap_or_else(|e| e.into_inner()).1);
        for (_, waker) in waiters {
            waker.wake();
        }
    }
}
