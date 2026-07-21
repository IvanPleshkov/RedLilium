//! Display headroom query (#154 step 2): the window's current EDR headroom
//! `H = peak luminance / SDR-white`, in ×SDR-white units (`H >= 1`).
//!
//! The value is display state, not swapchain state — it tracks the physical
//! monitor and the OS brightness slider live — so callers poll it per frame and
//! smooth. Sources:
//!
//! - **macOS**: the window's `NSScreen`
//!   `maximumExtendedDynamicRangeColorComponentValue` (already in ×SDR-white
//!   units), backend-independent (works under both wgpu and Vulkan/MoltenVK).
//! - **Windows**: the window monitor's peak luminance from DXGI
//!   (`IDXGIOutput6::GetDesc1().MaxLuminance`, in nits) over the scRGB reference
//!   white (80 nits) — the surface is scRGB extended-linear, where `1.0` = 80
//!   nits.
//! - **elsewhere / any failure**: [`SDR_HEADROOM`] (`1.0`), the display-
//!   independent default the deferred display pass collapses to (a no-op
//!   roll-off).

use raw_window_handle::HasWindowHandle;

/// Headroom of an SDR display, and the safe fallback when the real value can't
/// be read: peak luminance equals SDR white, so highlights have no headroom.
pub const SDR_HEADROOM: f32 = 1.0;

/// The window's current display headroom `H` (`>= 1`). See the module docs.
///
/// Never panics and never blocks: any failure to resolve the screen yields
/// [`SDR_HEADROOM`].
#[cfg(target_os = "macos")]
pub fn window_display_headroom<W: HasWindowHandle>(window: &W) -> f32 {
    use objc2_app_kit::NSView;
    use raw_window_handle::RawWindowHandle;

    let Ok(handle) = window.window_handle() else {
        return SDR_HEADROOM;
    };
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return SDR_HEADROOM;
    };

    // SAFETY: `ns_view` points at this window's live `NSView`, and the render
    // loop that calls this runs on the main thread — where AppKit access is
    // sound. We only read scalar properties, retaining nothing past this scope.
    let view: &NSView = unsafe { appkit.ns_view.cast::<NSView>().as_ref() };
    let headroom = unsafe {
        view.window()
            .and_then(|w| w.screen())
            .map(|s| s.maximumExtendedDynamicRangeColorComponentValue())
            .unwrap_or(1.0)
    };

    // A window with no HDR-capable screen reports 1.0; guard NaN/inf/sub-1 too.
    if headroom.is_finite() && headroom >= 1.0 {
        headroom as f32
    } else {
        SDR_HEADROOM
    }
}

/// The window monitor's headroom from DXGI. See the module docs.
///
/// **Runtime-unverified**: written and compile-checked cross-target from macOS;
/// the DXGI walk and the scRGB white reference want validation on real Windows
/// HDR hardware (the SDR-white slider may shift the effective reference).
#[cfg(target_os = "windows")]
pub fn window_display_headroom<W: HasWindowHandle>(window: &W) -> f32 {
    use raw_window_handle::RawWindowHandle;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, IDXGIOutput6};
    use windows::Win32::Graphics::Gdi::{MONITOR_DEFAULTTONEAREST, MonitorFromWindow};
    use windows::core::Interface;

    // scRGB extended-linear (the swapchain colour space, see
    // `backend/vulkan/swapchain.rs`): `1.0` = 80 nits, the sRGB reference white.
    // Headroom is the panel peak over that reference.
    const SCRGB_SDR_WHITE_NITS: f32 = 80.0;

    let Ok(handle) = window.window_handle() else {
        return SDR_HEADROOM;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return SDR_HEADROOM;
    };
    let hwnd = HWND(win32.hwnd.get() as *mut core::ffi::c_void);

    // SAFETY: DXGI/GDI FFI. The HWND is the live window's; every call is
    // error-checked and any failure falls back to SDR. Nothing is retained.
    unsafe {
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else {
            return SDR_HEADROOM;
        };
        let mut adapter_index = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(adapter_index) {
            adapter_index += 1;
            let mut output_index = 0u32;
            while let Ok(output) = adapter.EnumOutputs(output_index) {
                output_index += 1;
                let Ok(desc) = output.GetDesc() else {
                    continue;
                };
                if desc.Monitor != monitor {
                    continue;
                }
                let Ok(output6) = output.cast::<IDXGIOutput6>() else {
                    continue;
                };
                let Ok(desc1) = output6.GetDesc1() else {
                    continue;
                };
                // MaxLuminance: the panel peak in nits. First cut — the Windows
                // SDR-content-brightness slider can shift the effective SDR
                // white; validate on hardware.
                let headroom = desc1.MaxLuminance / SCRGB_SDR_WHITE_NITS;
                return if headroom.is_finite() && headroom >= 1.0 {
                    headroom
                } else {
                    SDR_HEADROOM
                };
            }
        }
    }
    SDR_HEADROOM
}

/// No native EDR query on this platform yet (Linux/Wayland colour management is
/// a later step), so the headroom is always [`SDR_HEADROOM`].
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn window_display_headroom<W: HasWindowHandle>(_window: &W) -> f32 {
    SDR_HEADROOM
}
