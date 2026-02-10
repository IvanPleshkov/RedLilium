//! Physics math type aliases.
//!
//! Provides precision-aware type aliases for physics math types.
//! By default, uses `f64` precision. Enable the `physics-f32` feature
//! for `f32` precision.
//!
//! These types are based on [`nalgebra`] and are compatible with
//! [rapier](https://rapier.rs/) physics types.

pub use nalgebra;

/// Physics scalar type. `f64` by default, `f32` with `physics-f32` feature.
#[cfg(not(feature = "physics-f32"))]
pub type Real = f64;

/// Physics scalar type. `f32` with `physics-f32` feature.
#[cfg(feature = "physics-f32")]
pub type Real = f32;

/// 2D vector.
pub type Vector2 = nalgebra::Vector2<Real>;

/// 3D vector.
pub type Vector3 = nalgebra::Vector3<Real>;

/// 2D point.
pub type Point2 = nalgebra::Point2<Real>;

/// 3D point.
pub type Point3 = nalgebra::Point3<Real>;

/// 2D isometry (rotation + translation).
pub type Isometry2 = nalgebra::Isometry2<Real>;

/// 3D isometry (rotation + translation).
pub type Isometry3 = nalgebra::Isometry3<Real>;

/// 4x4 matrix.
pub type Matrix4 = nalgebra::Matrix4<Real>;

/// 2D rotation (unit complex number).
pub type UnitComplex = nalgebra::UnitComplex<Real>;

/// 3D rotation (unit quaternion).
pub type UnitQuaternion = nalgebra::UnitQuaternion<Real>;

/// 2D translation.
pub type Translation2 = nalgebra::Translation2<Real>;

/// 3D translation.
pub type Translation3 = nalgebra::Translation3<Real>;
