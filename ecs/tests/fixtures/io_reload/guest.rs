use redlilium_ecs::{IoHandle, IoRunner, IoRuntime};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};
struct Tls(Arc<AtomicUsize>);
impl Drop for Tls {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
thread_local! {static TLS:RefCell<Option<Tls>>=const{RefCell::new(None)};}
struct Guest {
    mode: u32,
    drops: Arc<AtomicUsize>,
    tls: Arc<AtomicUsize>,
    wake: Arc<Mutex<Option<Waker>>>,
}
impl Future for Guest {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        TLS.with(|slot| {
            if slot.borrow().is_none() {
                *slot.borrow_mut() = Some(Tls(self.tls.clone()));
            }
        });
        *self.wake.lock().unwrap() = Some(cx.waker().clone());
        match self.mode {
            0 => Poll::Ready(7),
            1 => panic!("guest poll"),
            _ => Poll::Ready(8),
        }
    }
}
impl Drop for Guest {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        if self.mode != 0 {
            panic!("guest drop");
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spawn(
    io: *const IoRuntime,
    drops: *const Arc<AtomicUsize>,
    tls: *const Arc<AtomicUsize>,
    wake: *const Arc<Mutex<Option<Waker>>>,
    mode: u32,
) -> *mut IoHandle<u32> {
    let drops = unsafe { &*drops }.clone();
    let tls = unsafe { &*tls }.clone();
    let wake = unsafe { &*wake }.clone();
    let io = unsafe { &*io };
    let handle = if mode == 3 {
        io.run(async {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            33
        })
    } else {
        io.run(Guest {
            mode,
            drops,
            tls,
            wake,
        })
    };
    Box::into_raw(Box::new(handle))
}
