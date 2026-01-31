//! Common types and descriptors for graphics resources.
//!
//! This module contains format enums, usage flags, and descriptor structs
//! used throughout the graphics system.

mod buffer;
mod common;
mod sampler;
mod texture;

pub use buffer::{BufferDescriptor, BufferUsage};
pub use common::{ClearValue, Extent3d};
pub use sampler::SamplerDescriptor;
pub use texture::{TextureDescriptor, TextureFormat, TextureUsage};
