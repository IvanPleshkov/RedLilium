//! Shader library and Slang compiler support.

/// Offline-baked slang→WGSL lookup for wasm (no runtime Slang in a browser, #33).
pub mod baked;
pub mod library;
#[cfg(feature = "slang-shaders")]
pub mod slang_compiler;
/// Shader variant spaces & keys (#6, Decision 5).
pub mod variants;

pub use library::{EGUI_SHADER_SOURCE, ShaderLibrary};
#[cfg(feature = "slang-shaders")]
pub use slang_compiler::{ShaderReflectInput, SlangCompiler};
pub use variants::{
    MAX_VARIANTS_PER_SHADER, ShaderVariantSpace, VariantAxis, VariantAxisKind, VariantError,
    VariantKey, VariantSelection, VariantSelector, VariantValue,
};
