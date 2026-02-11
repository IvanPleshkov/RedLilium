//! Physics integration with [rapier](https://rapier.rs/).
//!
//! Provides ECS resources, systems, and handle components for 2D and 3D physics.
//!
//! ## Features
//!
//! - `physics-3d` — 3D physics with `f64` precision (default)
//! - `physics-3d-f32` — 3D physics with `f32` precision
//! - `physics-2d` — 2D physics with `f64` precision
//! - `physics-2d-f32` — 2D physics with `f32` precision
//! - `physics` — both 3D and 2D (f64)

pub mod conversions;

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
pub mod physics3d;

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
pub mod components3d;

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
pub mod physics2d;

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
pub mod components2d;

// Re-export the active rapier crate under a unified name.

#[cfg(all(feature = "physics-3d", not(feature = "physics-3d-f32")))]
pub use rapier3d_f64 as rapier3d;

#[cfg(feature = "physics-3d-f32")]
pub use ::rapier3d;

#[cfg(all(feature = "physics-2d", not(feature = "physics-2d-f32")))]
pub use rapier2d_f64 as rapier2d;

#[cfg(feature = "physics-2d-f32")]
pub use ::rapier2d;
