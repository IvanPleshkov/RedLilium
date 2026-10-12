//! Fill gaps in winit 0.30's Win32 pointer cancellation and mouse promotion.
//! Subclassing also observes sent messages; an event-loop message hook only
//! observes queued messages and would miss synchronous capture changes.
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};
use windows_sys::Win32::{
    Foundation::*,
    UI::{
        Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass},
        WindowsAndMessaging::*,
    },
};
use winit::{
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

pub enum Cancellation {
    Mouse,
    Touch(u64),
}
struct CallbackData {
    pending: RefCell<Vec<Cancellation>>,
    attached: Cell<bool>,
}
pub struct WindowsInputBridge {
    // Keeps the HWND alive until our subclass is detached.
    _window: Arc<Window>,
    hwnd: HWND,
    data: Box<CallbackData>,
}
impl WindowsInputBridge {
    pub fn new(window: Arc<Window>) -> Option<Self> {
        let RawWindowHandle::Win32(handle) = window.window_handle().ok()?.as_raw() else {
            return None;
        };
        let hwnd = handle.hwnd.get() as HWND;
        let data = Box::new(CallbackData {
            pending: RefCell::default(),
            attached: Cell::new(true),
        });
        let pointer = &*data as *const CallbackData as usize;
        // SAFETY: called on the window thread. The boxed callback data has a
        // stable address and is retained until the subclass is removed in Drop.
        if unsafe { SetWindowSubclass(hwnd, Some(callback), pointer, pointer) } == 0 {
            log::error!("Failed to install Win32 input cancellation bridge");
            return None;
        }
        Some(Self {
            _window: window,
            hwnd,
            data,
        })
    }
    pub fn drain(&self) -> Vec<Cancellation> {
        std::mem::take(&mut *self.data.pending.borrow_mut())
    }
}
impl Drop for WindowsInputBridge {
    fn drop(&mut self) {
        if self.data.attached.replace(false) {
            // SAFETY: owned window is still alive; this is the installing thread.
            unsafe {
                RemoveWindowSubclass(
                    self.hwnd,
                    Some(callback),
                    &*self.data as *const CallbackData as usize,
                );
            }
        }
    }
}
unsafe extern "system" fn callback(
    hwnd: HWND,
    kind: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    id: usize,
    reference: usize,
) -> LRESULT {
    // SAFETY: SetWindowSubclass receives our Box's stable address; Drop removes
    // this callback before freeing it. No borrow is held while delegating to
    // DefSubclassProc, which can reenter the window procedure.
    let data = unsafe { &*(reference as *const CallbackData) };
    if kind == WM_NCDESTROY {
        data.attached.set(false);
        unsafe {
            RemoveWindowSubclass(hwnd, Some(callback), id);
        }
    }
    if kind == WM_CAPTURECHANGED && lparam != hwnd as isize {
        data.pending.borrow_mut().push(Cancellation::Mouse);
    }
    // GET_POINTERID_WPARAM / IS_POINTER_CANCELED_WPARAM from winuser.h.
    // https://learn.microsoft.com/windows/win32/inputmsg/wm-pointercapturechanged
    if kind == WM_POINTERCAPTURECHANGED
        || (matches!(kind, WM_POINTERUP | WM_POINTERUPDATE | WM_POINTERDOWN)
            && (wparam >> 16) & POINTER_MESSAGE_FLAG_CANCELED as usize != 0)
    {
        data.pending
            .borrow_mut()
            .push(Cancellation::Touch((wparam & 0xffff) as u64));
    }
    if (WM_MOUSEFIRST..=WM_MOUSELAST).contains(&kind) {
        // Filter the OS source tag, never timing/position heuristics. A real
        // concurrent mouse and pen retain their normal behavior.
        // https://learn.microsoft.com/windows/win32/tablet/system-events-and-mouse-messages
        let extra = unsafe { GetMessageExtraInfo() } as usize;
        if extra & 0xffff_ff80 == 0xff51_5780 {
            return 0;
        }
    }
    unsafe { DefSubclassProc(hwnd, kind, wparam, lparam) }
}
