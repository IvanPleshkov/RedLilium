//! # PBR IBL Demo
//!
//! Demonstrates:
//! - Forward PBR rendering with Image-Based Lighting (IBL)
//! - HDR environment map converted to cubemap
//! - Irradiance cubemap for diffuse IBL
//! - Pre-filtered environment map for specular IBL
//! - BRDF Look-Up Table for split-sum approximation
//! - Orbit camera (no ECS)
//! - Grid of PBR spheres with varying metallic/roughness
//!
//! Based on LearnOpenGL IBL tutorials:
//! - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
//! - https://learnopengl.com/PBR/IBL/Specular-IBL

mod camera;
mod demo;
mod gbuffer;
mod ibl;
mod ibl_textures;
mod resolve_pass;
mod resources;
mod skybox_pass;
mod sphere_grid;
mod ui;
mod uniforms;

use redlilium_app::{App, AppArgs, DefaultAppArgs};
use redlilium_core::profiling::create_profiled_allocator;

// Enable memory allocation tracking with Tracy.
// Set callstack depth to 32 for detailed allocation tracking (0 for minimal overhead).
create_profiled_allocator!(GLOBAL_ALLOCATOR, 32);

// Demo configuration constants
pub const GRID_SIZE: usize = 5;
pub const SPHERE_SPACING: f32 = 1.5;
pub const IRRADIANCE_SIZE: u32 = 32;
pub const PREFILTER_SIZE: u32 = 128;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse().with_hdr(true);
    App::run(demo::PbrIblDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
