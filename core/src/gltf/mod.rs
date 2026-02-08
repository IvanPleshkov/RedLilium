//! glTF 2.0 loader and exporter.
//!
//! Loads `.gltf`/`.glb` files into CPU-side data structures including
//! meshes, materials, textures, animations, skins, and scene graphs.
//! Exports scenes and materials back to binary glTF (`.glb`) format.
//!
//! # Callbacks
//!
//! The loader delegates resource creation to user-provided callbacks:
//!
//! - **`material_fn`** — Called per mesh primitive with a [`GltfMaterial`]
//!   (parsed PBR properties) and the native [`VertexLayout`] from the
//!   primitive's glTF attributes. Returns an `Arc<CpuMaterialInstance>`
//!   whose layout determines the mesh's final vertex format. If the
//!   material's layout differs from the native one, vertex data is adapted
//!   automatically. Results are cached by (material index, native layout).
//!
//! - **`sampler_fn`** — Called per glTF sampler with a [`CpuSampler`]
//!   containing the parsed filter and wrapping modes. Returns an
//!   `Arc<CpuSampler>` for sharing. The resolved samplers are embedded
//!   in [`TextureRef`] entries within [`GltfMaterial`].
//!
//! # Example
//!
//! ```ignore
//! use redlilium_core::gltf::{load_gltf, GltfMaterial};
//! use redlilium_core::material::*;
//! use redlilium_core::mesh::VertexLayout;
//! use redlilium_core::sampler::CpuSampler;
//! use std::sync::Arc;
//!
//! let data = std::fs::read("model.glb").unwrap();
//! let doc = load_gltf(
//!     &data,
//!     |mat: &GltfMaterial, layout: &VertexLayout| {
//!         let layout = Arc::new(layout.clone());
//!         let decl = CpuMaterial::pbr_metallic_roughness(
//!             layout,
//!             mat.alpha_mode, mat.double_sided,
//!             mat.base_color_texture.is_some(),
//!             mat.metallic_roughness_texture.is_some(),
//!             mat.normal_texture.is_some(),
//!             mat.occlusion_texture.is_some(),
//!             mat.emissive_texture.is_some(),
//!         );
//!         Arc::new(CpuMaterialInstance::new(Arc::new(decl)))
//!     },
//!     |s: &CpuSampler| Arc::new(s.clone()),
//! ).unwrap();
//!
//! println!("Scenes: {}", doc.scenes.len());
//! println!("Meshes: {}", doc.scenes[0].meshes.len());
//! ```

mod error;
mod exporter;
mod loader;
#[cfg(test)]
mod tests;
pub mod types;
mod vertex;

pub use error::GltfError;
pub use types::*;

use std::sync::Arc;

use crate::material::CpuMaterialInstance;
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;

/// Load a glTF document from binary data.
///
/// Supports both binary glTF (`.glb`) and JSON glTF (`.gltf` with embedded
/// data URIs). External file references are not supported.
///
/// # Arguments
///
/// * `data` - Raw bytes of the `.glb` or `.gltf` file.
/// * `material_fn` - Callback invoked per mesh primitive. Receives a
///   [`GltfMaterial`] with parsed PBR properties and a reference to the
///   native [`VertexLayout`] built from the primitive's glTF attributes.
///   Returns an `Arc<CpuMaterialInstance>` whose `CpuMaterial::vertex_layout`
///   determines the mesh's final vertex format. Results are cached by
///   (glTF material index, native layout).
/// * `sampler_fn` - Callback invoked per glTF sampler. Receives a
///   [`CpuSampler`] with parsed filter and wrapping modes. Returns an
///   `Arc<CpuSampler>` to embed in texture references. Called during
///   sampler loading, before material preparation.
///
/// # Returns
///
/// A [`GltfDocument`] containing all loaded scenes with meshes, cameras,
/// skins, and animations. Material instances are stored in each scene's
/// `materials` array. Meshes reference materials by index via
/// `CpuMesh::material()`.
pub fn load_gltf(
    data: &[u8],
    mut material_fn: impl FnMut(&GltfMaterial, &VertexLayout) -> Arc<CpuMaterialInstance>,
    mut sampler_fn: impl FnMut(&CpuSampler) -> Arc<CpuSampler>,
) -> Result<GltfDocument, GltfError> {
    let gltf = gltf_dep::Gltf::from_slice(data)?;
    let blob = gltf.blob.clone();

    let buffers = loader::resolve_buffers(&gltf.document, blob)?;
    let mut ctx = loader::LoadContext::new(gltf.document, buffers);

    ctx.load_textures()?;
    ctx.load_samplers(&mut sampler_fn);
    ctx.prepare_gltf_materials();
    let cameras = ctx.load_cameras();
    let meshes = ctx.load_meshes(&mut material_fn)?;
    let skins = ctx.load_skins()?;
    let animations = ctx.load_animations()?;
    let materials = ctx.material_instances();
    let scenes = ctx.load_scenes(meshes, materials, cameras, skins, animations);
    let default_scene = ctx.default_scene();

    Ok(GltfDocument {
        scenes,
        default_scene,
    })
}

/// Export scenes to a binary glTF (`.glb`) file.
///
/// Material instances are collected from each scene's `materials` array
/// and deduplicated using Arc pointer identity. Textures and samplers
/// are collected from those material instances.
///
/// # Texture handling
///
/// - [`TextureSource::Cpu`] — encodes RGBA8 data as PNG and embeds it in the GLB.
/// - [`TextureSource::Named`] — saved as an external texture URI reference.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::gltf::save_gltf;
///
/// let scenes: Vec<&Scene> = doc.scenes.iter().collect();
/// let glb = save_gltf(&scenes, doc.default_scene).unwrap();
/// std::fs::write("output.glb", &glb).unwrap();
/// ```
pub fn save_gltf(scenes: &[&Scene], default_scene: Option<usize>) -> Result<Vec<u8>, GltfError> {
    let mut ctx = exporter::ExportContext::new();

    ctx.collect_resources(scenes);
    ctx.build_images()?;
    ctx.build_samplers();
    ctx.build_materials();
    ctx.build_scenes(scenes)?;
    ctx.set_default_scene(default_scene);
    ctx.finalize_buffer();
    ctx.to_glb()
}
