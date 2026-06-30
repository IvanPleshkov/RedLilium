//! Draw-time pipeline specialization cache (`docs/MATERIAL_ASSETS.md` Decision 4).
//!
//! A material carries no vertex layout and no formats (Decision 2) — those belong
//! to the mesh and the render target, known only at draw time. So the compiled
//! GPU pipeline (graphics [`Material`]) is built **lazily in the render system**,
//! keyed by `(shader + vertex layout + formats)`, and cached here. This is Bevy's
//! `SpecializedMeshPipeline` pattern: only encountered combinations compile.
//!
//! For the first slice the shader entry points (`vs_main`/`fs_main`), the defines
//! (none), and which binding set is dynamic (group 0 = transform ring; group 1 =
//! static material props, Decision 7) are fixed for the one `opaque` model. The
//! variant/define system (Decision 5) and shader-declared `[UpdateRate]` set
//! classes (Decision 7) will replace those constants later.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_assets::Guid;
use redlilium_core::mesh::VertexLayout;
use redlilium_graphics::{
    GraphicsDevice, GraphicsError, Material, MaterialDescriptor, ShaderSource, ShaderStage,
    TextureFormat,
};

/// Cache key: the shader asset, the (shared, interned) vertex layout by pointer
/// identity, and the render target formats. Layouts are interned by the
/// [`VertexLayoutManager`](super::VertexLayoutManager), so pointer identity is
/// content identity — which is also what the renderer batches on.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader: Guid,
    layout: usize,
    color: TextureFormat,
    depth: Option<TextureFormat>,
}

/// Owns and shares specialized GPU pipelines (`Arc<Material>`) — an ECS resource
/// consulted by the render system at draw time.
pub struct PipelineCache {
    device: Arc<GraphicsDevice>,
    cache: HashMap<PipelineKey, Arc<Material>>,
}

impl PipelineCache {
    /// Create a pipeline cache for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            cache: HashMap::new(),
        }
    }

    /// Get (or build + cache) the pipeline specialized for this shader + mesh
    /// vertex layout + target formats. `shader_guid` keys the cache; `source` is
    /// the `.slang` content compiled on a miss.
    pub fn get_or_build(
        &mut self,
        shader_guid: Guid,
        source: &[u8],
        layout: &Arc<VertexLayout>,
        color: TextureFormat,
        depth: TextureFormat,
    ) -> Result<Arc<Material>, GraphicsError> {
        let key = PipelineKey {
            shader: shader_guid,
            layout: Arc::as_ptr(layout) as usize,
            color,
            depth: Some(depth),
        };
        if let Some(mat) = self.cache.get(&key) {
            return Ok(Arc::clone(mat));
        }

        let material = self.device.create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::slang(
                    ShaderStage::Vertex,
                    source.to_vec(),
                    "vs_main",
                    vec![],
                ))
                .with_shader(ShaderSource::slang(
                    ShaderStage::Fragment,
                    source.to_vec(),
                    "fs_main",
                    vec![],
                ))
                .with_vertex_layout(Arc::clone(layout))
                .with_color_format(color)
                .with_depth_format(depth)
                // Group 0 (per-entity transform) is the dynamic ring set; group 1
                // (material props) stays a static uniform buffer (Decision 7).
                .with_dynamic_uniform(0, 0)
                .with_label("opaque"),
        )?;
        self.cache.insert(key, Arc::clone(&material));
        Ok(material)
    }

    /// Number of distinct specialized pipelines currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}
