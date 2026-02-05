//! Scene graph types for representing loaded scenes.
//!
//! These types are format-agnostic and can be produced by any loader
//! (glTF, FBX, custom formats) or built programmatically.
//!
//! - [`Scene`] — A scene with nodes and resources
//! - [`SceneNode`] — A node in the scene tree
//! - [`NodeTransform`] — TRS transform using plain arrays
//! - [`SceneCamera`] / [`CameraProjection`] — Camera definitions
//! - [`SceneSkin`] — Skeletal animation skin
//! - [`Animation`] / [`AnimationChannel`] — Keyframe animations

mod types;

pub use types::{
    Animation, AnimationChannel, AnimationProperty, CameraProjection, Interpolation, NodeTransform,
    Scene, SceneCamera, SceneNode, SceneSkin,
};
