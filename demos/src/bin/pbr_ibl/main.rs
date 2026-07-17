//! # PBR IBL Demo
//!
//! Demonstrates:
//! - Forward PBR rendering with Image-Based Lighting (IBL)
//! - Baked IBL asset set (#137): sky cubemap, irradiance cubemap,
//!   GGX-prefiltered environment map, and BRDF LUT, loaded from the
//!   Zstd-supercompressed KTX2 files in `std-assets/textures/ibl/`
//! - Orbit camera (no ECS)
//! - Grid of PBR spheres with varying metallic/roughness
//!
//! Shading follows the LearnOpenGL IBL tutorials:
//! - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
//! - https://learnopengl.com/PBR/IBL/Specular-IBL

mod camera;
mod demo;
mod ecs_scene;
mod gbuffer;
mod ibl_textures;
mod resolve_pass;
mod shadow_pass;
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

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse().with_hdr(true);
    App::run(demo::PbrIblDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
