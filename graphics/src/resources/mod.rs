//! GPU resources.
//!
//! This module contains the GPU resource types that are created by [`GraphicsDevice`]:
//! - [`Buffer`] - GPU memory buffer
//! - [`Texture`] - GPU texture/image
//! - [`Sampler`] - Texture sampler
//! - [`RingBuffer`] - Circular buffer for efficient GPU streaming
//!
//! Resources are reference-counted with [`Arc`] and can be shared across threads.
//! Each resource holds a weak reference back to its parent device.
//!
//! [`GraphicsDevice`]: crate::GraphicsDevice
//! [`Arc`]: std::sync::Arc

mod acceleration_structure;
mod buffer;
mod ring_buffer;
mod sampler;
mod texture;

pub(crate) use acceleration_structure::instance_buffer_size;
pub use acceleration_structure::{
    AccelBuildSizes, Blas, BlasDescriptor, BlasTriangles, Tlas, TlasDescriptor, TlasInstance,
};
pub use buffer::Buffer;
pub use ring_buffer::{RingAllocation, RingBuffer};
pub use sampler::Sampler;
pub use texture::Texture;
