use redlilium_ecs::{IoHandle, IoRuntime};
use std::ffi::{CString, c_char, c_void};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::task::Waker;
use std::time::Duration;
unsafe extern "C" {
    fn dlopen(p: *const c_char, f: i32) -> *mut c_void;
    fn dlsym(h: *mut c_void, n: *const c_char) -> *mut c_void;
    fn dlclose(h: *mut c_void) -> i32;
}
fn main() {
    let path = CString::new(std::env::args().nth(1).unwrap()).unwrap();
    let io = IoRuntime::new();
    for _ in 0..3 {
        let module = unsafe { dlopen(path.as_ptr(), 2) };
        assert!(!module.is_null());
        let symbol = unsafe { dlsym(module, c"spawn".as_ptr()) };
        assert!(!symbol.is_null());
        let spawn: unsafe extern "C" fn(
            *const IoRuntime,
            *const Arc<AtomicUsize>,
            *const Arc<AtomicUsize>,
            *const Arc<Mutex<Option<Waker>>>,
            u32,
        ) -> *mut IoHandle<u32> = unsafe { std::mem::transmute(symbol) };
        let count = Arc::new(AtomicUsize::new(0));
        let tls = Arc::new(AtomicUsize::new(0));
        let mut wakes = Vec::new();
        for mode in 0..4 {
            let wake = Arc::new(Mutex::new(None));
            let handle = unsafe { Box::from_raw(spawn(&io, &count, &tls, &wake, mode)) };
            assert_eq!(
                handle.recv_timeout(Duration::from_secs(5)),
                match mode {
                    0 => Some(7),
                    1 => None,
                    2 => Some(8),
                    _ => Some(33),
                },
                "mode {mode}"
            );
            io.quiesce(Duration::from_secs(1)).unwrap();
            if mode != 3 {
                wakes.push(wake.lock().unwrap().take().unwrap());
            }
        }
        io.shutdown(Duration::from_secs(1)).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 3);
        assert_eq!(tls.load(Ordering::SeqCst), 1);
        assert_eq!(unsafe { dlclose(module) }, 0);
        for wake in wakes {
            wake.wake();
        }
    }
    println!(
        "IO cdylib: 3 cycles; timer, poll/drop panic, TLS joined, late wakes after unload passed"
    );
}
