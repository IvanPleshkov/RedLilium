//! Common types and descriptors for graphics resources.
//!
//! This module contains format enums, usage flags, and descriptor structs
//! used throughout the graphics system.

mod buffer;
mod common;
mod sampler;
mod texture;

pub use buffer::{BufferDescriptor, BufferUsage, DrawIndexedIndirectArgs, DrawIndirectArgs};
pub use common::{ClearValue, Extent3d, ScissorRect, Viewport};
pub use sampler::{AddressMode, CompareFunction, FilterMode, SamplerDescriptor};
pub use texture::{TextureDescriptor, TextureDimension, TextureFormat, TextureUsage};
