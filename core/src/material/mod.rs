//! CPU-side material definitions with semantic-based properties.
//!
//! Materials store data as a list of [`MaterialProperty`] entries, each
//! with a [`MaterialSemantic`] tag and a typed [`MaterialValue`]. This
//! allows format-agnostic material representation that bridges loaders
//! (glTF, FBX, etc.) with the generic graphics material system.
//!
//! - [`CpuMaterial`] — Material with named properties and pipeline state
//! - [`MaterialSemantic`] — Well-known PBR semantics + custom extension
//! - [`MaterialValue`] — Typed property value (float, vec3, vec4, texture)
//! - [`TextureRef`] — Texture + sampler + UV set reference
//! - [`AlphaMode`] — Alpha rendering mode (opaque, mask, blend)

mod types;

pub use types::{
    AlphaMode, CpuMaterial, MaterialProperty, MaterialSemantic, MaterialValue, TextureRef,
    TextureSource,
};
