use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crate::error::VfsError;
use crate::provider::VfsFuture;

/// Poll a VFS future once, expecting it to be immediately ready.
///
/// This is a convenience for synchronous contexts (editor UI, CLI tools,
/// tests) where the underlying provider does blocking I/O (e.g.
/// [`FileSystemProvider`](crate::FileSystemProvider),
/// [`MemoryProvider`](crate::MemoryProvider)) and the future completes
/// on the first poll.
///
/// # Panics
///
/// Panics if the future returns `Poll::Pending`. This should not happen
/// with blocking providers but would indicate a provider that requires
/// a real async runtime.
pub fn poll_now<T>(mut fut: VfsFuture<T>) -> Result<T, VfsError> {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    match Pin::new(&mut fut).poll(&mut cx) {
        Poll::Ready(val) => val,
        Poll::Pending => panic!("VFS future returned Pending — use IoRuntime for async providers"),
    }
}

fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
