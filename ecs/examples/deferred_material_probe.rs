//! Public-API GPU acceptance probe, shared with the native integration test.
//! Build with `--features rendering`. On WASM call the exported `run_browser`.
#[path = "../tests/deferred_custom_material.rs"]
mod probe;
fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    probe::run_native();
}
