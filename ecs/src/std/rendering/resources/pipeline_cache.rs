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
    CullMode, GraphicsDevice, GraphicsError, Material, MaterialDescriptor, ShaderSource,
    ShaderStage, TextureFormat, VariantKey,
};

use crate::std::rendering::loaders::Shader;

/// Cache key: the shader asset, its variant selection (#6, Decision 5), the
/// (shared, interned) vertex layout by pointer identity, and the render target
/// formats — Decision 4's full formula `(shader + defines + layout + state +
/// formats)`. Layouts are interned by the
/// [`VertexLayoutManager`](super::VertexLayoutManager), so pointer identity is
/// content identity — which is also what the renderer batches on.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader: Guid,
    variant: VariantKey,
    layout: usize,
    /// Color attachment formats in attachment order — one for a regular pass,
    /// several for MRT (a G-buffer, #144), empty for depth-only pipelines
    /// (zero color attachments, #129).
    colors: Vec<TextureFormat>,
    depth: Option<TextureFormat>,
}

/// A cached specialization: the pipeline plus the shader `Arc` it was compiled
/// from (pointer identity is the version — a hot-reloaded shader recompiles).
/// `broken` remembers a shader `Arc` whose (re)compile failed so nothing
/// retries every frame: with a last-good pipeline (`material: Some`) that one
/// keeps serving; without one (`material: None` — the first build failed,
/// e.g. a single-output shader against an MRT pass, #144) the failure itself
/// is cached and re-reported cheaply until the shader `Arc` changes.
struct PipelineEntry {
    shader: Arc<Shader>,
    material: Option<Arc<Material>>,
    broken: Option<usize>,
}

/// Owns and shares specialized GPU pipelines (`Arc<Material>`) — an ECS resource
/// consulted by the render system at draw time.
pub struct PipelineCache {
    device: Arc<GraphicsDevice>,
    cache: HashMap<PipelineKey, PipelineEntry>,
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
    /// vertex layout + target formats. `colors` lists the pass's color
    /// attachment formats in order (one for a regular pass, several for MRT —
    /// a G-buffer, #144) and must not be empty (use
    /// [`get_or_build_depth_only`](Self::get_or_build_depth_only) for
    /// zero-color pipelines). `shader_guid` keys the cache; `shader` is
    /// the resident source compiled on a miss — and revalidated by pointer
    /// identity on a hit, so a hot-reloaded shader recompiles. If the recompile
    /// fails (broken source mid-edit), the last-good pipeline keeps serving.
    pub fn get_or_build(
        &mut self,
        shader_guid: Guid,
        shader: &Arc<Shader>,
        variant: &VariantKey,
        layout: &Arc<VertexLayout>,
        colors: &[TextureFormat],
        depth: TextureFormat,
    ) -> Result<Arc<Material>, GraphicsError> {
        debug_assert!(
            !colors.is_empty(),
            "get_or_build needs at least one color format; \
             use get_or_build_depth_only for zero-color pipelines"
        );
        let key = PipelineKey {
            shader: shader_guid,
            variant: variant.clone(),
            layout: Arc::as_ptr(layout) as usize,
            colors: colors.to_vec(),
            depth: Some(depth),
        };
        self.get_or_build_with(key, shader_guid, shader, |device, shader| {
            Self::build(device, shader, variant, layout, colors, depth)
        })
    }

    /// Get (or build + cache) the **depth-only** specialization of `shader`
    /// (#129): vertex stage only, zero color attachments. Used by depth-only
    /// phases (shadow maps, depth prepass) with a shared depth shader; same
    /// hot-reload/last-good semantics as [`get_or_build`](Self::get_or_build).
    pub fn get_or_build_depth_only(
        &mut self,
        shader_guid: Guid,
        shader: &Arc<Shader>,
        layout: &Arc<VertexLayout>,
        depth: TextureFormat,
    ) -> Result<Arc<Material>, GraphicsError> {
        let key = PipelineKey {
            shader: shader_guid,
            variant: VariantKey::default(),
            layout: Arc::as_ptr(layout) as usize,
            colors: Vec::new(),
            depth: Some(depth),
        };
        self.get_or_build_with(key, shader_guid, shader, |device, shader| {
            Self::build_depth_only(device, shader, layout, depth)
        })
    }

    /// The shared cache-hit / hot-reload / last-good-serving logic behind
    /// both specialization entry points.
    fn get_or_build_with(
        &mut self,
        key: PipelineKey,
        shader_guid: Guid,
        shader: &Arc<Shader>,
        build: impl Fn(&Arc<GraphicsDevice>, &Arc<Shader>) -> Result<Arc<Material>, GraphicsError>,
    ) -> Result<Arc<Material>, GraphicsError> {
        /// The error served for a cached failure (the real one was logged when
        /// it happened; rebuilding it every frame would repeat the compile).
        fn cached_failure(shader_guid: Guid) -> GraphicsError {
            GraphicsError::InvalidParameter(format!(
                "pipeline specialization of shader {shader_guid:?} previously \
                 failed (cached; edit the shader to retry)"
            ))
        }

        if let Some(entry) = self.cache.get_mut(&key) {
            if Arc::ptr_eq(&entry.shader, shader)
                || entry.broken == Some(Arc::as_ptr(shader) as usize)
            {
                // Same shader version, or a version already known broken:
                // serve the cached pipeline (or the cached failure) without
                // recompiling.
                return match &entry.material {
                    Some(material) => Ok(Arc::clone(material)),
                    None => Err(cached_failure(shader_guid)),
                };
            }
            // The shader was reloaded — recompile; keep the last-good pipeline
            // (and don't retry the same broken Arc every frame) on failure.
            match build(&self.device, shader) {
                Ok(material) => {
                    entry.shader = Arc::clone(shader);
                    entry.material = Some(Arc::clone(&material));
                    entry.broken = None;
                    return Ok(material);
                }
                Err(e) => {
                    log::warn!("shader {shader_guid:?} recompile failed (keeping last-good): {e}");
                    entry.broken = Some(Arc::as_ptr(shader) as usize);
                    return match &entry.material {
                        Some(material) => Ok(Arc::clone(material)),
                        None => Err(e),
                    };
                }
            }
        }

        match build(&self.device, shader) {
            Ok(material) => {
                self.cache.insert(
                    key,
                    PipelineEntry {
                        shader: Arc::clone(shader),
                        material: Some(Arc::clone(&material)),
                        broken: None,
                    },
                );
                Ok(material)
            }
            Err(e) => {
                // Cache the failure so the draw path doesn't recompile the
                // same broken combination every frame (#144: e.g. a pbr
                // material on the forward path, or an opaque material on the
                // deferred MRT pass).
                log::warn!(
                    "shader {shader_guid:?} pipeline specialization failed \
                     (cached, not retried until the shader changes): {e}"
                );
                self.cache.insert(
                    key,
                    PipelineEntry {
                        shader: Arc::clone(shader),
                        material: None,
                        broken: Some(Arc::as_ptr(shader) as usize),
                    },
                );
                Err(e)
            }
        }
    }

    /// Compile the pipeline for this shader source + variant + layout + formats.
    fn build(
        device: &Arc<GraphicsDevice>,
        shader: &Arc<Shader>,
        variant: &VariantKey,
        layout: &Arc<VertexLayout>,
        colors: &[TextureFormat],
        depth: TextureFormat,
    ) -> Result<Arc<Material>, GraphicsError> {
        let mut descriptor = MaterialDescriptor::new()
            .with_shader(ShaderSource::slang(
                ShaderStage::Vertex,
                shader.source.clone(),
                "vs_main",
                vec![],
            ))
            .with_shader(ShaderSource::slang(
                ShaderStage::Fragment,
                shader.source.clone(),
                "fs_main",
                vec![],
            ));
        for &color in colors {
            descriptor = descriptor.with_color_format(color);
        }
        device.create_material(
            &descriptor
                .with_variant(variant.clone())
                .with_vertex_layout(Arc::clone(layout))
                .with_depth_format(depth)
                // Cull back faces (#39): closed meshes shade ~half as many
                // fragments. Meshes use the engine's CCW-front convention
                // (glTF), normalized per backend. Double-sided materials would
                // need `CullMode::None`, but `double_sided` is not yet plumbed
                // into the material-asset path — a follow-up.
                .with_cull_mode(CullMode::Back)
                // Which set is dynamic/static/external is self-describing: the
                // shader's [UpdateRate] blocks classify each set through
                // reflection (Decision 7) — no hardcoded set indices here.
                .with_label("opaque"),
        )
    }

    /// Compile the depth-only pipeline: vertex stage only, no color formats
    /// (#129). Cull state matches [`build`](Self::build) so a depth pass
    /// covers exactly the faces the main pass shades.
    fn build_depth_only(
        device: &Arc<GraphicsDevice>,
        shader: &Arc<Shader>,
        layout: &Arc<VertexLayout>,
        depth: TextureFormat,
    ) -> Result<Arc<Material>, GraphicsError> {
        device.create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::slang(
                    ShaderStage::Vertex,
                    shader.source.clone(),
                    "vs_main",
                    vec![],
                ))
                .with_vertex_layout(Arc::clone(layout))
                .with_depth_format(depth)
                .with_cull_mode(CullMode::Back)
                .with_label("depth_only"),
        )
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
