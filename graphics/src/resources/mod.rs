//! GPU resources.
//!
//! This module contains the GPU resource types that are created by [`GraphicsDevice`]:
//! - [`Buffer`] - GPU memory buffer
//! - [`Texture`] - GPU texture/image
//! - [`Sampler`] - Texture sampler
//!
//! Resources are reference-counted with [`Arc`] and can be shared across threads.
//! Each resource holds a weak reference back to its parent device.
//!
//! [`GraphicsDevice`]: crate::GraphicsDevice
//! [`Arc`]: std::sync::Arc

mod buffer;
mod sampler;
mod texture;

pub use buffer::Buffer;
pub use sampler::Sampler;
pub use texture::Texture;
