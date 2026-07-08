//! Shader library and Slang compiler support.

/// Offline-baked slang→WGSL lookup for wasm (no runtime Slang in a browser, #33).
pub mod baked;
pub mod library;
#[cfg(feature = "slang-shaders")]
pub mod slang_compiler;

pub use library::{EGUI_SHADER_SOURCE, ShaderLibrary};
#[cfg(feature = "slang-shaders")]
pub use slang_compiler::{ShaderReflectInput, SlangCompiler};
