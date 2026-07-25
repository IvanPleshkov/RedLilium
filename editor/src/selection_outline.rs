//! Selection outline material — a fullscreen contour around the selected
//! entities' visible silhouette, drawn from the R8 mask the entity-index
//! (picking) pass writes as its second target. Editor-only, like picking:
//! the game runtime never links this pass.

use std::sync::Arc;

use redlilium_graphics::{
    GraphicsDevice, Material, MaterialDescriptor, ShaderSource, ShaderStage, TextureFormat,
};

// Kept in a file (not an inline string) so the offline shader bake reads the
// exact same bytes the runtime hashes — see `xtask bake-shaders`.
const SHADER_SLANG: &str = include_str!("../shaders/selection_outline.slang");

/// Per-frame outline parameters. Layout must match the Slang `Uniforms`
/// cbuffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelectionOutlineUniforms {
    /// Panel origin inside the (window-sized) mask texture, in pixels.
    pub mask_offset: [f32; 2],
    /// Mask texture dimensions, in pixels.
    pub mask_size: [f32; 2],
    /// Outline color.
    pub color: [f32; 4],
    /// Outline width in pixels.
    pub thickness: f32,
    pub _padding: [f32; 3],
}

/// Create the GPU [`Material`] for the selection outline pass.
///
/// Renders a fullscreen triangle onto the scene camera's color target
/// (`color_format`), discarding every pixel that is not within `thickness`
/// pixels outside the selection mask. Binding 0 (the uniforms) uses a
/// per-draw dynamic offset into the outline params ring.
pub fn create_selection_outline_material(
    device: &Arc<GraphicsDevice>,
    color_format: TextureFormat,
) -> Arc<Material> {
    device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::slang(
                    ShaderStage::Vertex,
                    SHADER_SLANG.as_bytes().to_vec(),
                    "vs_main",
                    vec![],
                ))
                .with_shader(ShaderSource::slang(
                    ShaderStage::Fragment,
                    SHADER_SLANG.as_bytes().to_vec(),
                    "fs_main",
                    vec![],
                ))
                .with_color_format(color_format)
                .with_dynamic_uniform(0, 0)
                .with_label("selection_outline"),
        )
        .expect("Failed to create selection outline material")
}
