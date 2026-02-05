//! CPU-side sampler types.
//!
//! Provides [`CpuSampler`] for describing texture sampling parameters,
//! along with [`FilterMode`], [`AddressMode`], and [`CompareFunction`]
//! enums shared between CPU and GPU code.

mod types;

pub use types::{AddressMode, CompareFunction, CpuSampler, FilterMode};
