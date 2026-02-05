//! CPU-side texture types.
//!
//! Provides [`CpuTexture`] for holding raw pixel data, along with
//! [`TextureFormat`] and [`TextureDimension`] enums shared between
//! CPU and GPU code.

mod types;

pub use types::{CpuTexture, TextureDimension, TextureFormat};
