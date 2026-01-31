//! Scene rendering infrastructure for bridging ECS and render graphs.
//!
//! This module provides the glue between ECS worlds and the render graph system:
//!
//! - [`RenderWorld`] - Extracted render data from an ECS world
//! - [`CameraSystem`] - Manages per-camera render graphs and orchestrates rendering
//! - [`SceneRenderer`] - Simple single-camera renderer (legacy, prefer CameraSystem)
//! - Extract/Prepare/Render phases for efficient data flow
//!
//! # Architecture
//!
//! The scene rendering follows a three-phase approach:
//!
//! 1. **Extract Phase** - Copy render-relevant data from ECS to RenderWorld
//! 2. **Prepare Phase** - Process extracted data into GPU-ready formats
//! 3. **Render Phase** - Execute the render graph with prepared data
//!
//! This separation allows the main ECS world to continue simulation while
//! rendering uses a snapshot of the previous frame's state.
//!
//! # Camera System
//!
//! The [`CameraSystem`] handles multiple cameras with priority-based ordering:
//! - Texture-target cameras render first (for render-to-texture)
//! - Surface-target cameras render last
//! - Each camera has its own render graph and visibility filtering
//!
//! # Multiple Worlds
//!
//! A process can contain multiple ECS worlds, each with their own render graphs.
//! The backend is shared across all worlds for efficient GPU resource usage.

mod camera_system;
mod extracted;
mod render_world;

pub use camera_system::{CameraRenderContext, CameraSystem, ExtractedCamera};
pub use extracted::{ExtractedMaterial, ExtractedMesh, ExtractedTransform};
pub use render_world::RenderWorld;
