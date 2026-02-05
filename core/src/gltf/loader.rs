//! Internal glTF loading logic.
//!
//! The [`LoadContext`] holds all state needed during loading: resolved buffer
//! data, layout cache, and the parsed glTF document.

use std::sync::Arc;

use crate::material::{
    AlphaMode, CpuMaterial, MaterialProperty, MaterialSemantic, MaterialValue, TextureRef,
};
use crate::mesh::{CpuMesh, VertexLayout};
use crate::sampler::{AddressMode, CpuSampler, FilterMode};
use crate::scene::{
    Animation, AnimationChannel, AnimationProperty, CameraProjection, Interpolation, NodeTransform,
    Scene, SceneCamera, SceneNode, SceneSkin,
};
use crate::texture::{CpuTexture, TextureFormat};

use super::error::GltfError;
use super::vertex;

/// Internal loading context that holds resolved data during loading.
pub(crate) struct LoadContext {
    /// The parsed glTF document.
    document: gltf_dep::Document,
    /// Resolved buffer data (one Vec<u8> per glTF buffer).
    buffers: Vec<Vec<u8>>,

    /// Shared layouts passed in by the caller.
    shared_layouts: Vec<Arc<VertexLayout>>,
    /// New layouts created during loading.
    new_layouts: Vec<Arc<VertexLayout>>,

    /// Shared samplers passed in by the caller.
    shared_samplers: Vec<Arc<CpuSampler>>,
    /// New samplers created during loading.
    new_samplers: Vec<Arc<CpuSampler>>,
    /// Loaded sampler Arcs indexed by glTF sampler index.
    sampler_arcs: Vec<Arc<CpuSampler>>,

    /// Mapping from glTF mesh index → range of flat CpuMesh indices.
    /// Populated by `load_meshes`, used by `load_scenes`.
    mesh_index_map: Vec<Vec<usize>>,
}

impl LoadContext {
    /// Create a new LoadContext from parsed glTF data.
    pub fn new(
        document: gltf_dep::Document,
        buffers: Vec<Vec<u8>>,
        shared_layouts: &[Arc<VertexLayout>],
        shared_samplers: &[Arc<CpuSampler>],
    ) -> Self {
        Self {
            document,
            buffers,
            shared_layouts: shared_layouts.to_vec(),
            new_layouts: Vec::new(),
            shared_samplers: shared_samplers.to_vec(),
            new_samplers: Vec::new(),
            sampler_arcs: Vec::new(),
            mesh_index_map: Vec::new(),
        }
    }

    /// Decode all images and build CpuTextures by inlining image data
    /// for each glTF texture.
    pub fn load_textures(&self) -> Result<Vec<CpuTexture>, GltfError> {
        let images = self.decode_images()?;

        let textures = self
            .document
            .textures()
            .map(|tex| {
                let image_index = tex.source().index();
                let image = &images[image_index];
                let mut cpu_tex = CpuTexture::new(
                    image.width,
                    image.height,
                    TextureFormat::Rgba8Unorm,
                    image.data.clone(),
                );
                if let Some(name) = tex.name() {
                    cpu_tex = cpu_tex.with_name(name);
                }
                cpu_tex
            })
            .collect();

        Ok(textures)
    }

    /// Decode all images to RGBA8.
    fn decode_images(&self) -> Result<Vec<DecodedImage>, GltfError> {
        let mut images = Vec::new();

        for image in self.document.images() {
            match image.source() {
                gltf_dep::image::Source::View { view, mime_type } => {
                    let buffer_index = view.buffer().index();
                    let buffer_data = self.buffers.get(buffer_index).ok_or_else(|| {
                        GltfError::BufferError(format!(
                            "image buffer index {buffer_index} out of range"
                        ))
                    })?;
                    let start = view.offset();
                    let end = start + view.length();
                    let image_bytes = &buffer_data[start..end];

                    images.push(decode_image(image_bytes, mime_type)?);
                }
                gltf_dep::image::Source::Uri { uri, mime_type } => {
                    if let Some(data) = parse_data_uri(uri) {
                        let mime = mime_type.unwrap_or("image/png");
                        images.push(decode_image(&data, mime)?);
                    } else {
                        return Err(GltfError::ImageDecode(format!(
                            "external URI images not supported: {uri}"
                        )));
                    }
                }
            }
        }

        Ok(images)
    }

    /// Load all samplers as shared `Arc<CpuSampler>`.
    ///
    /// Reuses structurally matching samplers from `shared_samplers`. New
    /// samplers are stored in `new_samplers`. The resulting `sampler_arcs`
    /// vector is indexed by glTF sampler index and used by
    /// `resolve_texture_sampler`.
    pub fn load_samplers(&mut self) {
        self.sampler_arcs = self
            .document
            .samplers()
            .map(|sampler| {
                let (min_filter, mipmap_filter) = sampler
                    .min_filter()
                    .map(map_min_filter)
                    .unwrap_or((FilterMode::Nearest, FilterMode::Nearest));
                let mag_filter = sampler
                    .mag_filter()
                    .map(map_mag_filter)
                    .unwrap_or(FilterMode::Nearest);

                let cpu_sampler = CpuSampler {
                    name: None,
                    mag_filter,
                    min_filter,
                    mipmap_filter,
                    address_mode_u: map_wrapping(sampler.wrap_s()),
                    address_mode_v: map_wrapping(sampler.wrap_t()),
                    address_mode_w: AddressMode::ClampToEdge,
                    ..Default::default()
                };

                find_or_create_sampler(cpu_sampler, &self.shared_samplers, &mut self.new_samplers)
            })
            .collect();
    }

    /// Load all materials as property-based CpuMaterials.
    pub fn load_materials(&self) -> Vec<CpuMaterial> {
        self.document
            .materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();

                // PBR scalar factors
                let mut props = vec![
                    MaterialProperty {
                        semantic: MaterialSemantic::BaseColorFactor,
                        value: MaterialValue::Vec4(pbr.base_color_factor()),
                    },
                    MaterialProperty {
                        semantic: MaterialSemantic::MetallicFactor,
                        value: MaterialValue::Float(pbr.metallic_factor()),
                    },
                    MaterialProperty {
                        semantic: MaterialSemantic::RoughnessFactor,
                        value: MaterialValue::Float(pbr.roughness_factor()),
                    },
                    MaterialProperty {
                        semantic: MaterialSemantic::EmissiveFactor,
                        value: MaterialValue::Vec3(mat.emissive_factor()),
                    },
                ];

                // PBR textures
                if let Some(t) = pbr.base_color_texture() {
                    props.push(self.map_material_texture(&t, MaterialSemantic::BaseColorTexture));
                }
                if let Some(t) = pbr.metallic_roughness_texture() {
                    props.push(
                        self.map_material_texture(&t, MaterialSemantic::MetallicRoughnessTexture),
                    );
                }
                if let Some(t) = mat.normal_texture() {
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::NormalTexture,
                        value: MaterialValue::Texture(
                            self.texture_ref_from(&t.texture(), t.tex_coord()),
                        ),
                    });
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::NormalScale,
                        value: MaterialValue::Float(t.scale()),
                    });
                }
                if let Some(t) = mat.occlusion_texture() {
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::OcclusionTexture,
                        value: MaterialValue::Texture(
                            self.texture_ref_from(&t.texture(), t.tex_coord()),
                        ),
                    });
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::OcclusionStrength,
                        value: MaterialValue::Float(t.strength()),
                    });
                }
                if let Some(t) = mat.emissive_texture() {
                    props.push(self.map_material_texture(&t, MaterialSemantic::EmissiveTexture));
                }

                // Alpha cutoff (only meaningful for Mask mode, but store it if present)
                if let Some(cutoff) = mat.alpha_cutoff() {
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::AlphaCutoff,
                        value: MaterialValue::Float(cutoff),
                    });
                } else if matches!(mat.alpha_mode(), gltf_dep::material::AlphaMode::Mask) {
                    props.push(MaterialProperty {
                        semantic: MaterialSemantic::AlphaCutoff,
                        value: MaterialValue::Float(0.5),
                    });
                }

                let alpha_mode = match mat.alpha_mode() {
                    gltf_dep::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf_dep::material::AlphaMode::Mask => AlphaMode::Mask,
                    gltf_dep::material::AlphaMode::Blend => AlphaMode::Blend,
                };

                let mut cpu_mat = CpuMaterial {
                    name: mat.name().map(String::from),
                    alpha_mode,
                    double_sided: mat.double_sided(),
                    properties: props,
                };
                if mat.name().is_none() {
                    cpu_mat.name = None;
                }
                cpu_mat
            })
            .collect()
    }

    /// Map a glTF texture info to a MaterialProperty with the given semantic.
    fn map_material_texture(
        &self,
        t: &gltf_dep::texture::Info<'_>,
        semantic: MaterialSemantic,
    ) -> MaterialProperty {
        let (texture, sampler) = self.resolve_texture_sampler(&t.texture());
        MaterialProperty {
            semantic,
            value: MaterialValue::Texture(TextureRef {
                texture,
                sampler,
                tex_coord: t.tex_coord(),
            }),
        }
    }

    /// Build a TextureRef from a glTF normal or occlusion texture.
    fn texture_ref_from(&self, tex: &gltf_dep::Texture<'_>, tex_coord: u32) -> TextureRef {
        let (texture, sampler) = self.resolve_texture_sampler(tex);
        TextureRef {
            texture,
            sampler,
            tex_coord,
        }
    }

    /// Resolve glTF texture index and shared sampler.
    fn resolve_texture_sampler(
        &self,
        tex: &gltf_dep::Texture<'_>,
    ) -> (usize, Option<Arc<CpuSampler>>) {
        let sampler = tex
            .sampler()
            .index()
            .map(|idx| Arc::clone(&self.sampler_arcs[idx]));
        (tex.index(), sampler)
    }

    /// Load all cameras.
    pub fn load_cameras(&self) -> Vec<SceneCamera> {
        self.document
            .cameras()
            .map(|cam| {
                let projection = match cam.projection() {
                    gltf_dep::camera::Projection::Perspective(p) => CameraProjection::Perspective {
                        yfov: p.yfov(),
                        aspect: p.aspect_ratio(),
                        znear: p.znear(),
                        zfar: p.zfar(),
                    },
                    gltf_dep::camera::Projection::Orthographic(o) => {
                        CameraProjection::Orthographic {
                            xmag: o.xmag(),
                            ymag: o.ymag(),
                            znear: o.znear(),
                            zfar: o.zfar(),
                        }
                    }
                };
                SceneCamera {
                    name: cam.name().map(String::from),
                    projection,
                }
            })
            .collect()
    }

    /// Load all meshes with vertex interleaving and layout sharing.
    ///
    /// Returns a flat list of `CpuMesh` (one per glTF primitive). Each mesh
    /// carries its material index via `CpuMesh::material()`. Also populates
    /// `self.mesh_index_map` so that `load_scenes` can map glTF mesh indices
    /// to flat CpuMesh indices.
    pub fn load_meshes(&mut self) -> Result<Vec<CpuMesh>, GltfError> {
        let mut result = Vec::new();
        let mut index_map = Vec::new();

        // Collect all accessors for data reading
        let accessors: Vec<gltf_dep::Accessor<'_>> = self.document.accessors().collect();

        for (mesh_idx, mesh) in self.document.meshes().enumerate() {
            let mut flat_indices = Vec::new();

            for (prim_idx, primitive) in mesh.primitives().enumerate() {
                // Check for POSITION attribute
                if primitive.get(&gltf_dep::Semantic::Positions).is_none() {
                    return Err(GltfError::MissingPositions {
                        mesh: mesh_idx,
                        primitive: prim_idx,
                    });
                }

                // Build layout from primitive attributes
                let (layout, attrs) = vertex::build_layout_from_primitive(&primitive)?;

                // Map topology
                let topology = vertex::map_topology(primitive.mode())?;

                // Find or create shared layout
                let shared_layout = vertex::find_or_create_layout(
                    layout,
                    &self.shared_layouts,
                    &mut self.new_layouts,
                );

                // Get vertex count from POSITION accessor
                let pos_accessor = primitive.get(&gltf_dep::Semantic::Positions).unwrap();
                let vertex_count = pos_accessor.count() as u32;

                // Get stride from the layout
                let stride = shared_layout.buffer_stride(0);

                // Interleave vertex data
                let vertex_data = vertex::interleave_vertices(
                    &attrs,
                    vertex_count,
                    stride,
                    &accessors,
                    &self.buffers,
                )?;

                // Read indices if present
                let index_data = if let Some(indices_accessor) = primitive.indices() {
                    Some(vertex::read_indices(
                        &indices_accessor,
                        &self.buffers,
                        vertex_count,
                    )?)
                } else {
                    None
                };

                // Build label
                let label = mesh.name().map(|name| {
                    if mesh.primitives().count() > 1 {
                        format!("{name}_prim{prim_idx}")
                    } else {
                        name.to_string()
                    }
                });

                // Build CpuMesh with material
                let mut cpu_mesh =
                    vertex::build_cpu_mesh(shared_layout, topology, vertex_data, index_data, label);
                if let Some(mat_idx) = primitive.material().index() {
                    cpu_mesh = cpu_mesh.with_material(mat_idx);
                }

                let flat_idx = result.len();
                flat_indices.push(flat_idx);
                result.push(cpu_mesh);
            }

            index_map.push(flat_indices);
        }

        self.mesh_index_map = index_map;
        Ok(result)
    }

    /// Load all skins.
    pub fn load_skins(&self) -> Result<Vec<SceneSkin>, GltfError> {
        let mut result = Vec::new();

        for skin in self.document.skins() {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();

            let inverse_bind_matrices = if let Some(accessor) = skin.inverse_bind_matrices() {
                read_mat4_accessor(&accessor, &self.buffers)?
            } else {
                // Default: identity matrices
                vec![
                    [
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0
                    ];
                    joints.len()
                ]
            };

            result.push(SceneSkin {
                name: skin.name().map(String::from),
                joints,
                inverse_bind_matrices,
                skeleton: skin.skeleton().map(|n| n.index()),
            });
        }

        Ok(result)
    }

    /// Load all animations.
    pub fn load_animations(&self) -> Result<Vec<Animation>, GltfError> {
        let mut result = Vec::new();

        for anim in self.document.animations() {
            let mut channels = Vec::new();

            for channel in anim.channels() {
                let target = channel.target();
                let target_node = target.node().index();
                let property = match target.property() {
                    gltf_dep::animation::Property::Translation => AnimationProperty::Translation,
                    gltf_dep::animation::Property::Rotation => AnimationProperty::Rotation,
                    gltf_dep::animation::Property::Scale => AnimationProperty::Scale,
                    gltf_dep::animation::Property::MorphTargetWeights => {
                        AnimationProperty::MorphTargetWeights
                    }
                };

                let sampler = channel.sampler();
                let interpolation = match sampler.interpolation() {
                    gltf_dep::animation::Interpolation::Linear => Interpolation::Linear,
                    gltf_dep::animation::Interpolation::Step => Interpolation::Step,
                    gltf_dep::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                };

                let input_accessor = sampler.input();
                let output_accessor = sampler.output();

                let timestamps = read_f32_accessor(&input_accessor, &self.buffers)?;
                let values = read_f32_accessor(&output_accessor, &self.buffers)?;

                channels.push(AnimationChannel {
                    target_node,
                    property,
                    interpolation,
                    timestamps,
                    values,
                });
            }

            let mut animation = Animation::new().with_channels(channels);
            if let Some(name) = anim.name() {
                animation = animation.with_name(name);
            }
            result.push(animation);
        }

        Ok(result)
    }

    /// Load all scenes as node trees with embedded resources.
    ///
    /// Must be called after `load_meshes` so that `mesh_index_map` is populated.
    /// Takes ownership of document-level resources and embeds them into each scene.
    pub fn load_scenes(
        &self,
        mut meshes: Vec<CpuMesh>,
        mut cameras: Vec<SceneCamera>,
        mut skins: Vec<SceneSkin>,
        mut animations: Vec<Animation>,
    ) -> Vec<Scene> {
        let scene_count = self.document.scenes().count();

        self.document
            .scenes()
            .enumerate()
            .map(|(i, scene)| {
                let is_last = i == scene_count - 1;
                Scene {
                    name: scene.name().map(String::from),
                    nodes: scene
                        .nodes()
                        .map(|n| load_node(&n, &self.mesh_index_map))
                        .collect(),
                    meshes: if is_last {
                        std::mem::take(&mut meshes)
                    } else {
                        meshes.clone()
                    },
                    cameras: if is_last {
                        std::mem::take(&mut cameras)
                    } else {
                        cameras.clone()
                    },
                    skins: if is_last {
                        std::mem::take(&mut skins)
                    } else {
                        skins.clone()
                    },
                    animations: if is_last {
                        std::mem::take(&mut animations)
                    } else {
                        animations.clone()
                    },
                }
            })
            .collect()
    }

    /// Get the default scene index.
    pub fn default_scene(&self) -> Option<usize> {
        self.document.default_scene().map(|s| s.index())
    }

    /// Consume the context and return new layouts and new samplers.
    pub fn into_new_resources(self) -> (Vec<Arc<VertexLayout>>, Vec<Arc<CpuSampler>>) {
        (self.new_layouts, self.new_samplers)
    }
}

// -- Helper functions --

/// Recursively load a node and its children.
///
/// `mesh_index_map` maps glTF mesh index → list of flat CpuMesh indices.
fn load_node(node: &gltf_dep::Node<'_>, mesh_index_map: &[Vec<usize>]) -> SceneNode {
    let (translation, rotation, scale) = node.transform().decomposed();

    let meshes = node
        .mesh()
        .map(|m| mesh_index_map[m.index()].clone())
        .unwrap_or_default();

    SceneNode {
        name: node.name().map(String::from),
        transform: NodeTransform {
            translation,
            rotation,
            scale,
        },
        meshes,
        camera: node.camera().map(|c| c.index()),
        skin: node.skin().map(|s| s.index()),
        children: node
            .children()
            .map(|c| load_node(&c, mesh_index_map))
            .collect(),
    }
}

/// Check if two samplers are structurally equal (ignoring name).
fn samplers_structurally_equal(a: &CpuSampler, b: &CpuSampler) -> bool {
    a.address_mode_u == b.address_mode_u
        && a.address_mode_v == b.address_mode_v
        && a.address_mode_w == b.address_mode_w
        && a.mag_filter == b.mag_filter
        && a.min_filter == b.min_filter
        && a.mipmap_filter == b.mipmap_filter
        && a.lod_min_clamp == b.lod_min_clamp
        && a.lod_max_clamp == b.lod_max_clamp
        && a.compare == b.compare
        && a.anisotropy_clamp == b.anisotropy_clamp
}

/// Find or create a shared sampler.
///
/// Searches `existing_samplers` for a structural match. If found, returns the
/// existing Arc. Otherwise, creates a new Arc and appends it to `new_samplers`.
fn find_or_create_sampler(
    sampler: CpuSampler,
    existing_samplers: &[Arc<CpuSampler>],
    new_samplers: &mut Vec<Arc<CpuSampler>>,
) -> Arc<CpuSampler> {
    // Search in pre-existing shared samplers
    for existing in existing_samplers {
        if samplers_structurally_equal(&sampler, existing) {
            return Arc::clone(existing);
        }
    }
    // Search in newly created samplers
    for new_sampler in new_samplers.iter() {
        if samplers_structurally_equal(&sampler, new_sampler) {
            return Arc::clone(new_sampler);
        }
    }
    // Create a new one
    let arc = Arc::new(sampler);
    new_samplers.push(Arc::clone(&arc));
    arc
}

/// Decoded image data (internal, not exposed).
struct DecodedImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

fn decode_image(bytes: &[u8], _mime_type: &str) -> Result<DecodedImage, GltfError> {
    let img = image::load_from_memory(bytes).map_err(|e| GltfError::ImageDecode(format!("{e}")))?;

    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    Ok(DecodedImage {
        data: rgba.into_raw(),
        width,
        height,
    })
}

/// Parse a data URI (e.g., `data:image/png;base64,...`) and return the decoded bytes.
fn parse_data_uri(uri: &str) -> Option<Vec<u8>> {
    let prefix = "data:";
    if !uri.starts_with(prefix) {
        return None;
    }
    let rest = &uri[prefix.len()..];
    let base64_start = rest.find(";base64,")?;
    let encoded = &rest[base64_start + 8..];
    base64_decode(encoded)
}

/// Simple base64 decoder (avoids adding a dependency).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn decode_char(c: u8) -> Option<u8> {
        TABLE.iter().position(|&b| b == c).map(|p| p as u8)
    }

    let input: Vec<u8> = input
        .bytes()
        .filter(|&b| b != b'\n' && b != b'\r' && b != b' ')
        .collect();
    let mut result = Vec::with_capacity(input.len() * 3 / 4);

    for chunk in input.chunks(4) {
        let mut buf = [0u8; 4];
        let mut pad = 0;

        for (i, &byte) in chunk.iter().enumerate() {
            if byte == b'=' {
                pad += 1;
                buf[i] = 0;
            } else {
                buf[i] = decode_char(byte)?;
            }
        }

        result.push((buf[0] << 2) | (buf[1] >> 4));
        if pad < 2 {
            result.push((buf[1] << 4) | (buf[2] >> 2));
        }
        if pad < 1 {
            result.push((buf[2] << 6) | buf[3]);
        }
    }

    Some(result)
}

/// Map glTF magnification filter to core FilterMode.
fn map_mag_filter(filter: gltf_dep::texture::MagFilter) -> FilterMode {
    match filter {
        gltf_dep::texture::MagFilter::Nearest => FilterMode::Nearest,
        gltf_dep::texture::MagFilter::Linear => FilterMode::Linear,
    }
}

/// Map glTF minification filter to core FilterMode pair (min_filter, mipmap_filter).
fn map_min_filter(filter: gltf_dep::texture::MinFilter) -> (FilterMode, FilterMode) {
    match filter {
        gltf_dep::texture::MinFilter::Nearest => (FilterMode::Nearest, FilterMode::Nearest),
        gltf_dep::texture::MinFilter::Linear => (FilterMode::Linear, FilterMode::Nearest),
        gltf_dep::texture::MinFilter::NearestMipmapNearest => {
            (FilterMode::Nearest, FilterMode::Nearest)
        }
        gltf_dep::texture::MinFilter::NearestMipmapLinear => {
            (FilterMode::Nearest, FilterMode::Linear)
        }
        gltf_dep::texture::MinFilter::LinearMipmapNearest => {
            (FilterMode::Linear, FilterMode::Nearest)
        }
        gltf_dep::texture::MinFilter::LinearMipmapLinear => {
            (FilterMode::Linear, FilterMode::Linear)
        }
    }
}

/// Map glTF wrapping mode to core AddressMode.
fn map_wrapping(wrap: gltf_dep::texture::WrappingMode) -> AddressMode {
    match wrap {
        gltf_dep::texture::WrappingMode::ClampToEdge => AddressMode::ClampToEdge,
        gltf_dep::texture::WrappingMode::MirroredRepeat => AddressMode::MirrorRepeat,
        gltf_dep::texture::WrappingMode::Repeat => AddressMode::Repeat,
    }
}

/// Read an accessor as a flat array of f32 values.
fn read_f32_accessor(
    accessor: &gltf_dep::Accessor,
    buffers: &[Vec<u8>],
) -> Result<Vec<f32>, GltfError> {
    let view = accessor.view().ok_or_else(|| {
        GltfError::AccessorError(format!("accessor {} has no buffer view", accessor.index()))
    })?;

    let buffer_index = view.buffer().index();
    let buffer_data = buffers.get(buffer_index).ok_or_else(|| {
        GltfError::BufferError(format!("buffer index {buffer_index} out of range"))
    })?;

    let component_count = accessor.dimensions().multiplicity();
    let element_size = 4 * component_count; // f32 = 4 bytes
    let stride = view.stride().unwrap_or(element_size);
    let start = view.offset() + accessor.offset();
    let count = accessor.count();

    let mut result = Vec::with_capacity(count * component_count);

    for i in 0..count {
        let offset = start + i * stride;
        for c in 0..component_count {
            let byte_offset = offset + c * 4;
            if byte_offset + 4 <= buffer_data.len() {
                let bytes = [
                    buffer_data[byte_offset],
                    buffer_data[byte_offset + 1],
                    buffer_data[byte_offset + 2],
                    buffer_data[byte_offset + 3],
                ];
                result.push(f32::from_le_bytes(bytes));
            }
        }
    }

    Ok(result)
}

/// Read an accessor as an array of [f32; 16] matrices.
fn read_mat4_accessor(
    accessor: &gltf_dep::Accessor,
    buffers: &[Vec<u8>],
) -> Result<Vec<[f32; 16]>, GltfError> {
    let values = read_f32_accessor(accessor, buffers)?;
    let count = accessor.count();
    let mut matrices = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i * 16;
        if offset + 16 <= values.len() {
            let mut mat = [0.0f32; 16];
            mat.copy_from_slice(&values[offset..offset + 16]);
            matrices.push(mat);
        }
    }

    Ok(matrices)
}

/// Resolve all buffer data from the glTF document.
///
/// For binary glTF (.glb), the first buffer is the embedded blob.
/// External URI buffers and data URIs are also supported.
pub(crate) fn resolve_buffers(
    document: &gltf_dep::Document,
    blob: Option<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, GltfError> {
    let mut buffers = Vec::new();

    for buffer in document.buffers() {
        match buffer.source() {
            gltf_dep::buffer::Source::Bin => {
                let data = blob.as_ref().ok_or_else(|| {
                    GltfError::BufferError("binary buffer referenced but no blob present".into())
                })?;
                buffers.push(data.clone());
            }
            gltf_dep::buffer::Source::Uri(uri) => {
                if let Some(data) = parse_data_uri(uri) {
                    buffers.push(data);
                } else {
                    return Err(GltfError::BufferError(format!(
                        "external buffer URIs not supported: {uri}"
                    )));
                }
            }
        }
    }

    Ok(buffers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_decode() {
        let encoded = "SGVsbG8gV29ybGQ=";
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"Hello World");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        let encoded = "YQ==";
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"a");
    }

    #[test]
    fn test_parse_data_uri() {
        let uri = "data:application/octet-stream;base64,AQID";
        let data = parse_data_uri(uri).unwrap();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_data_uri_not_data() {
        let uri = "file://some/path";
        assert!(parse_data_uri(uri).is_none());
    }

    #[test]
    fn test_default_transform() {
        let t = NodeTransform::default();
        assert_eq!(t.translation, [0.0, 0.0, 0.0]);
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
    }
}
