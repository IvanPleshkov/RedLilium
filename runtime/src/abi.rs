//! The game-module ABI contract (ADR-020, #45): the two symbols a game cdylib
//! exports, the fingerprint that gates loading, and the [`GameModule`] handle
//! the host holds.
//!
//! A game cdylib exports exactly two symbols via [`redlilium_game_module!`]:
//!
//! - `redlilium_abi_fingerprint` — an `extern "C"` fn returning a C string.
//!   The C ABI is stable regardless of Rust version, so the host can call this
//!   **before** it trusts any Rust-ABI compatibility. It is the gate.
//! - `redlilium_game_module` — a Rust-ABI fn returning `Box<dyn Plugin>`. This
//!   hands a Rust trait object across the boundary and is therefore only sound
//!   once the fingerprint matched; the host calls it **only after** the gate.
//!
//! ## Linking model
//!
//! Rust has no stable ABI: two separate compilations of the "same" engine can
//! differ in layout, vtable order, and `TypeId` hashes. #45 does **not** use a
//! shared engine dylib (that was evaluated and deferred). Instead the game
//! cdylib is built in the **same `cargo build`** as the host, so both statically
//! embed the *same* engine rlibs: identical `StableCrateId` → identical
//! `TypeId`s → identical layouts → the `Box<dyn Plugin>` handoff and the
//! name-keyed snapshot round-trip are sound. There are two *copies* of the
//! engine code (host image + cdylib image); see [`GameModule::load`]'s Safety
//! notes for what that duplication does and does not permit.
//!
//! ## What the fingerprint does and does not guarantee
//!
//! The fingerprint encodes the engine version, the `rustc` version, and a
//! content id for the engine source (`build.rs`: `HEAD` rev + a hash of the
//! uncommitted diff), so a toolchain change, a version bump, or *any* engine
//! source edit — committed or not — shifts it. The game bakes its fingerprint
//! in as a `const` inlined from the engine metadata it compiled against (not a
//! call into engine code — that would resolve into the host image and
//! tautologically match), so a host built from different engine sources rejects
//! a stale game.
//!
//! Within a *single* `cargo build` the id is computed once and shared, so the
//! host and the game cdylib always agree and are mutually consistent — this is
//! what makes same-build loading (the reload harness) sound. The id only
//! discriminates across *separate* builds, which is the editor reload flow.
//!
//! It is still **necessary, not sufficient**. What it does NOT catch:
//!
//! - **Feature unification / profile differences.** These change layout and
//!   `-C metadata` without changing the source content the id hashes.
//! - **Duplicated statics.** Two engine copies means two of every `static`,
//!   thread-local, and global registry (allocator, `log` logger, panic
//!   bookkeeping). Fingerprint parity says nothing about this; see
//!   [`GameModule::load`]. The ECS's own component/resource locks sidestep it
//!   by using [`redlilium_ecs::sync`], which is cross-image safe by
//!   construction.
//!
//! Do NOT treat [`QualifiedTypeId`](redlilium_ecs::QualifiedTypeId) as the
//! backstop for the residual. `TypeId` derives from the crate's `StableCrateId`
//! (name + version + `-C metadata` + rustc) — **not** source contents — so a
//! layout-changing edit keeps the same `TypeId`. The load-time fail-fast catches
//! `TypeId` *drift* (a game type colliding with an engine type from another
//! source); it does **not** catch *layout drift under a stable `TypeId`*. That
//! residual is closed procedurally by the content-hash build id above: the
//! fingerprint is the check that the two sides were built from the same engine
//! sources, not a proof of it.

use std::ffi::c_char;

use crate::Plugin;

/// The canonical fingerprint string, baked in at compile time from the engine
/// version, the `rustc` version, and the engine source revision (`build.rs`).
/// A game cdylib inlines this (via [`ABI_FINGERPRINT`]) from the engine
/// metadata it compiled against; the host holds its own. A mismatch means the
/// two were built from incompatible engines.
macro_rules! fingerprint_str {
    () => {
        concat!(
            "redlilium-runtime ",
            env!("CARGO_PKG_VERSION"),
            " | rustc ",
            env!("REDLILIUM_RUSTC_VERSION"),
            " | build ",
            env!("REDLILIUM_BUILD_ID"),
        )
    };
}

/// This build's ABI fingerprint: engine version + `rustc` version + engine
/// source revision. The revision component means any engine rebuild past a
/// source edit changes the fingerprint, so a stale game cdylib is rejected;
/// see [`redlilium_game_module!`] for why the game bakes this in as a const.
pub const ABI_FINGERPRINT: &str = fingerprint_str!();

/// The exported symbol name for the game entry point.
///
/// Kept in sync with what [`redlilium_game_module!`] emits; the loader
/// (slice C) resolves this exact byte string.
pub const GAME_MODULE_SYMBOL: &[u8] = b"redlilium_game_module";

/// The exported symbol name for the ABI fingerprint.
pub const ABI_FINGERPRINT_SYMBOL: &[u8] = b"redlilium_abi_fingerprint";

/// Signature of the game entry symbol. Rust-ABI: only call after the
/// fingerprint gate passes.
pub type GameModuleFn = unsafe extern "Rust" fn() -> Box<dyn Plugin>;

/// Signature of the fingerprint symbol. C-ABI: always safe to call.
pub type AbiFingerprintFn = unsafe extern "C" fn() -> *const c_char;

/// This build's fingerprint as a Rust string, for comparison and display.
///
/// The host compares this against the C string a game cdylib returns from
/// `redlilium_abi_fingerprint` (see [`redlilium_game_module!`]); the game bakes
/// its own copy in at compile time rather than calling back into engine code.
pub fn abi_fingerprint() -> &'static str {
    ABI_FINGERPRINT
}

/// Why loading a game cdylib failed.
#[derive(Debug)]
pub enum GameModuleError {
    /// `dlopen`/symbol resolution failed (missing file, unresolved symbols —
    /// e.g. the shared engine dylib was not found, missing exported symbol).
    Load(libloading::Error),
    /// The cdylib's fingerprint did not match this host's — a different
    /// toolchain or engine version. Loading is refused before any Rust-ABI
    /// call, because the trait-object handoff would be unsound.
    FingerprintMismatch {
        /// This host's fingerprint.
        host: String,
        /// The fingerprint the cdylib reported.
        module: String,
    },
    /// The fingerprint symbol returned a null or non-UTF-8 C string.
    InvalidFingerprint,
    /// The game entry symbol panicked while constructing the plugin. The host
    /// stays alive; the panic payload was dropped before the library unmapped.
    EntryPanicked,
}

impl std::fmt::Display for GameModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameModuleError::Load(e) => write!(f, "failed to load game cdylib: {e}"),
            GameModuleError::FingerprintMismatch { host, module } => write!(
                f,
                "game cdylib ABI fingerprint mismatch:\n  host:   {host}\n  module: {module}\n\
                 rebuild the game against this engine with the same toolchain"
            ),
            GameModuleError::InvalidFingerprint => {
                write!(f, "game cdylib returned an invalid ABI fingerprint")
            }
            GameModuleError::EntryPanicked => {
                write!(f, "game cdylib panicked while constructing its plugin")
            }
        }
    }
}

impl std::error::Error for GameModuleError {}

/// A loaded game module: the plugin plus, for a dynamically loaded module, the
/// dylib it came from.
///
/// **Drop order is load-bearing.** The `Library` must outlive the `plugin`: the
/// plugin's code and vtable live *inside* the mapped library, so dropping the
/// library first would leave the boxed trait object pointing at unmapped
/// memory, and running its destructor is then UB. The field order here (plugin
/// before `_library`) makes Rust drop the plugin first, then the library.
///
/// **But the plugin is not the only thing that points into the dylib.** When a
/// plugin's `build`/`spawn_scene` runs, it plants game-owned `System`s,
/// serialize/restore `fn` pointers (component meta, snapshot hooks), event
/// update fns, command closures, and the drop glue of every game-component
/// storage into the `App`/`World`/`Schedules` — none of which have a lifetime
/// tie to this handle. **Invariant:** a `GameModule` must outlive every `App`,
/// `World`, and `Schedules` its plugin ever touched. Dropping (or reloading
/// over) a module while such state is still live means the next `run_frame`,
/// world drop, or snapshot capture calls into unmapped pages. The slice-C
/// reload driver enforces this structurally by owning `(GameModule, App)` and
/// tearing the `App` down before the module. (The captured `SerializedWorld` is
/// safe to hold across an unload — it is plain data, no foreign vtables.)
pub struct GameModule {
    plugin: Box<dyn Plugin>,
    /// Kept alive for the handle's lifetime; `None` for a static plugin.
    /// Declared after `plugin` so it drops *after* it (see the type note).
    _library: Option<libloading::Library>,
}

impl GameModule {
    /// Wrap a statically linked plugin — no dylib, no fingerprint check
    /// (host and plugin are one compilation).
    pub fn from_static(plugin: Box<dyn Plugin>) -> Self {
        Self {
            plugin,
            _library: None,
        }
    }

    /// Load a game cdylib from `path`, gating on the ABI fingerprint.
    ///
    /// Sequence: `dlopen` the library → call the **C-ABI** fingerprint symbol
    /// (safe regardless of Rust-ABI compatibility) → compare to this host's
    /// fingerprint → only on a match call the **Rust-ABI** entry symbol to get
    /// the plugin. The library is retained so the plugin's code stays mapped.
    ///
    /// # Safety
    ///
    /// #45 links statically: the cdylib must be built in the **same `cargo
    /// build`** as this host (see the module "Linking model" docs), so both
    /// embed the same engine rlibs. The caller must guarantee all of:
    ///
    /// - **Same build.** The cdylib and this host came from one `cargo build`
    ///   under the same `rustc`, engine version, and **feature set**, so their
    ///   `TypeId`s and type layouts agree. The fingerprint gate catches
    ///   version/rustc/source-content drift, but a match is necessary, not
    ///   sufficient (module docs): it does not prove identical feature
    ///   unification, and `QualifiedTypeId` does not backstop layout drift.
    /// - **Default allocator on both sides.** The plugin's `Box` is allocated in
    ///   the cdylib image and freed in the host image. With Rust's default
    ///   allocator both forward to the system allocator and this is sound; a
    ///   divergent `#[global_allocator]` on **either** side makes the cross-image
    ///   free UB.
    /// - **A genuine RedLilium module.** The file exports
    ///   `redlilium_abi_fingerprint` / `redlilium_game_module` with exactly the
    ///   [`AbiFingerprintFn`] / [`GameModuleFn`] signatures — i.e. it was
    ///   produced by [`redlilium_game_module!`]. Calling a foreign symbol of a
    ///   different type, or reading a non-NUL-terminated fingerprint, is UB.
    /// - **Lifetime.** The returned [`GameModule`] outlives every `App`/`World`/
    ///   `Schedules` its plugin touches (see the type docs).
    ///
    /// **Duplicated statics (not yet a shared engine).** Because there are two
    /// engine images, each has its own `static`s and thread-locals. This is
    /// benign for single-threaded, uncontended use (booting a game, the reload
    /// harness), but two hazards become live once game code runs under the
    /// multi-threaded runner alongside host code and must be resolved before
    /// then (by a shared engine dylib, or by confining game execution):
    ///
    /// - **`parking_lot` parking table.** A blocking lock/unlock on the *same*
    ///   lock instance from the two images uses two different global parking
    ///   tables → lost wakeups / deadlock under contention.
    /// - **Split panic bookkeeping.** A panic raised in the cdylib image and
    ///   caught here (the `catch_unwind` below, or any unwind through host
    ///   frames) touches two libstd panic counters → `thread::panicking()` can
    ///   misreport afterward.
    pub unsafe fn load(path: impl AsRef<std::ffi::OsStr>) -> Result<Self, GameModuleError> {
        unsafe {
            let library = libloading::Library::new(path).map_err(GameModuleError::Load)?;

            // Gate: read the C-ABI fingerprint before trusting any Rust ABI.
            let fingerprint_fn: libloading::Symbol<'_, AbiFingerprintFn> = library
                .get(ABI_FINGERPRINT_SYMBOL)
                .map_err(GameModuleError::Load)?;
            let module_fp_ptr = fingerprint_fn();
            if module_fp_ptr.is_null() {
                return Err(GameModuleError::InvalidFingerprint);
            }
            let module_fp = std::ffi::CStr::from_ptr(module_fp_ptr)
                .to_str()
                .map_err(|_| GameModuleError::InvalidFingerprint)?;
            if module_fp != ABI_FINGERPRINT {
                return Err(GameModuleError::FingerprintMismatch {
                    host: ABI_FINGERPRINT.to_owned(),
                    module: module_fp.to_owned(),
                });
            }

            // Gate passed: the Rust-ABI entry handoff is now sound under the
            // same-toolchain contract.
            let entry_fn: libloading::Symbol<'_, GameModuleFn> = library
                .get(GAME_MODULE_SYMBOL)
                .map_err(GameModuleError::Load)?;

            // A broken game build must not take down the host. If the plugin's
            // constructor panics, the payload is allocated inside the dylib with
            // dylib-side drop glue, so it MUST be dropped while the library is
            // still mapped — drop it here, inside the closure's error path,
            // before `library` (a local) unmaps at function scope exit.
            let plugin = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| entry_fn()))
                .map_err(|payload| {
                    drop(payload);
                    GameModuleError::EntryPanicked
                })?;

            Ok(Self {
                plugin,
                _library: Some(library),
            })
        }
    }

    /// The plugin to build into an [`App`](crate::App).
    pub fn plugin(&self) -> &dyn Plugin {
        self.plugin.as_ref()
    }
}

/// Export the two ABI symbols a game cdylib must provide.
///
/// `$plugin` is an expression evaluating to a value implementing
/// [`Plugin`](crate::Plugin):
///
/// ```ignore
/// redlilium_runtime::redlilium_game_module!(MyGame);
/// // or, when construction is needed:
/// redlilium_runtime::redlilium_game_module!(MyGame::new());
/// ```
///
/// The fingerprint is baked into the game cdylib as a compile-time `static`
/// built from [`ABI_FINGERPRINT`](crate::ABI_FINGERPRINT) — a `const` inlined
/// from the engine metadata **this game compiled against**. It is deliberately
/// *not* a call into engine code: under the shared-engine-dylib contract such a
/// call would resolve into the host's own dylib and always report the host's
/// fingerprint, making the gate tautological. Baking the const captures the
/// engine build the game was actually built against, so a host running a newer
/// engine detects a stale game.
#[macro_export]
macro_rules! redlilium_game_module {
    ($plugin:expr) => {
        /// Game entry point — returns the plugin. Rust ABI: the host calls
        /// this only after the fingerprint gate passes.
        #[unsafe(no_mangle)]
        pub extern "Rust" fn redlilium_game_module() -> ::std::boxed::Box<dyn $crate::Plugin> {
            ::std::boxed::Box::new($plugin)
        }

        /// ABI fingerprint — C ABI, safe for the host to call before trusting
        /// any Rust-ABI compatibility. Returns a pointer to a NUL-terminated
        /// `static` baked in at *this crate's* compile time.
        #[unsafe(no_mangle)]
        pub extern "C" fn redlilium_abi_fingerprint() -> *const ::std::ffi::c_char {
            // Const-eval the engine's fingerprint (inlined from metadata) into a
            // NUL-terminated byte array owned by this cdylib.
            static FINGERPRINT: [u8; $crate::ABI_FINGERPRINT.len() + 1] = {
                let src = $crate::ABI_FINGERPRINT.as_bytes();
                let mut buf = [0u8; $crate::ABI_FINGERPRINT.len() + 1];
                let mut i = 0;
                while i < src.len() {
                    buf[i] = src[i];
                    i += 1;
                }
                buf
            };
            FINGERPRINT.as_ptr() as *const ::std::ffi::c_char
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_names_engine_rustc_and_build() {
        let fp = abi_fingerprint();
        assert!(
            fp.contains(env!("CARGO_PKG_VERSION")),
            "carries engine version"
        );
        assert!(fp.contains("rustc "), "carries rustc version");
        assert!(!fp.contains("unknown"), "rustc version was captured: {fp}");
        assert!(fp.contains("build "), "carries the engine source revision");
    }

    #[test]
    fn static_module_hands_back_its_plugin() {
        struct Noop;
        impl Plugin for Noop {
            fn build(&self, _app: &mut crate::App) {}
        }
        let module = GameModule::from_static(Box::new(Noop));
        // The handle yields a usable &dyn Plugin (behavior exercised end-to-end
        // in slice C against a real cdylib).
        let _plugin: &dyn Plugin = module.plugin();
    }
}
