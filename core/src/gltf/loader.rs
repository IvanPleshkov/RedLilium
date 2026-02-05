//! Internal glTF loading logic.
//!
//! The [`LoadContext`] holds all state needed during loading: resolved buffer
//! data, layout cache, and the parsed glTF document.

use std::sync::Arc;

use crate::mesh::{CpuMesh, VertexLayout};

use super::error::GltfError;
use super::types::*;
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
    ) -> Self {
        Self {
            document,
            buffers,
            shared_layouts: shared_layouts.to_vec(),
            new_layouts: Vec::new(),
            mesh_index_map: Vec::new(),
        }
    }

    /// Load all images, decoding to RGBA8.
    pub fn load_images(&self) -> Result<Vec<GltfImage>, GltfError> {
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

                    let decoded = decode_image(image_bytes, mime_type)?;
                    images.push(GltfImage {
                        name: image.name().map(String::from),
                        data: decoded.data,
                        width: decoded.width,
                        height: decoded.height,
                    });
                }
                gltf_dep::image::Source::Uri { uri, mime_type } => {
                    // Check for embedded base64 data URI
                    if let Some(data) = parse_data_uri(uri) {
                        let mime = mime_type.unwrap_or("image/png");
                        let decoded = decode_image(&data, mime)?;
                        images.push(GltfImage {
                            name: image.name().map(String::from),
                            data: decoded.data,
                            width: decoded.width,
                            height: decoded.height,
                        });
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

    /// Load all samplers.
    pub fn load_samplers(&self) -> Vec<GltfSampler> {
        self.document
            .samplers()
            .map(|sampler| GltfSampler {
                mag_filter: sampler.mag_filter().map(map_mag_filter),
                min_filter: sampler.min_filter().map(map_min_filter),
                wrap_s: map_wrapping(sampler.wrap_s()),
                wrap_t: map_wrapping(sampler.wrap_t()),
            })
            .collect()
    }

    /// Load all textures.
    pub fn load_textures(&self) -> Vec<GltfTexture> {
        self.document
            .textures()
            .map(|tex| GltfTexture {
                name: tex.name().map(String::from),
                image: tex.source().index(),
                sampler: tex.sampler().index(),
            })
            .collect()
    }

    /// Load all materials.
    pub fn load_materials(&self) -> Vec<GltfMaterial> {
        self.document
            .materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();
                GltfMaterial {
                    name: mat.name().map(String::from),
                    base_color_factor: pbr.base_color_factor(),
                    base_color_texture: pbr.base_color_texture().map(|t| GltfTextureRef {
                        index: t.texture().index(),
                        tex_coord: t.tex_coord(),
                    }),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    metallic_roughness_texture: pbr.metallic_roughness_texture().map(|t| {
                        GltfTextureRef {
                            index: t.texture().index(),
                            tex_coord: t.tex_coord(),
                        }
                    }),
                    normal_texture: mat.normal_texture().map(|t| GltfNormalTextureRef {
                        index: t.texture().index(),
                        tex_coord: t.tex_coord(),
                        scale: t.scale(),
                    }),
                    occlusion_texture: mat.occlusion_texture().map(|t| GltfOcclusionTextureRef {
                        index: t.texture().index(),
                        tex_coord: t.tex_coord(),
                        strength: t.strength(),
                    }),
                    emissive_factor: mat.emissive_factor(),
                    emissive_texture: mat.emissive_texture().map(|t| GltfTextureRef {
                        index: t.texture().index(),
                        tex_coord: t.tex_coord(),
                    }),
                    alpha_mode: match mat.alpha_mode() {
                        gltf_dep::material::AlphaMode::Opaque => GltfAlphaMode::Opaque,
                        gltf_dep::material::AlphaMode::Mask => GltfAlphaMode::Mask,
                        gltf_dep::material::AlphaMode::Blend => GltfAlphaMode::Blend,
                    },
                    alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
                    double_sided: mat.double_sided(),
                }
            })
            .collect()
    }

    /// Load all cameras.
    pub fn load_cameras(&self) -> Vec<GltfCamera> {
        self.document
            .cameras()
            .map(|cam| {
                let projection = match cam.projection() {
                    gltf_dep::camera::Projection::Perspective(p) => {
                        GltfCameraProjection::Perspective {
                            yfov: p.yfov(),
                            aspect: p.aspect_ratio(),
                            znear: p.znear(),
                            zfar: p.zfar(),
                        }
                    }
                    gltf_dep::camera::Projection::Orthographic(o) => {
                        GltfCameraProjection::Orthographic {
                            xmag: o.xmag(),
                            ymag: o.ymag(),
                            znear: o.znear(),
                            zfar: o.zfar(),
                        }
                    }
                };
                GltfCamera {
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
    pub fn load_skins(&self) -> Result<Vec<GltfSkin>, GltfError> {
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

            result.push(GltfSkin {
                name: skin.name().map(String::from),
                joints,
                inverse_bind_matrices,
                skeleton: skin.skeleton().map(|n| n.index()),
            });
        }

        Ok(result)
    }

    /// Load all animations.
    pub fn load_animations(&self) -> Result<Vec<GltfAnimation>, GltfError> {
        let mut result = Vec::new();

        for anim in self.document.animations() {
            let mut channels = Vec::new();

            for channel in anim.channels() {
                let target = channel.target();
                let target_node = target.node().index();
                let property = match target.property() {
                    gltf_dep::animation::Property::Translation => {
                        GltfAnimationProperty::Translation
                    }
                    gltf_dep::animation::Property::Rotation => GltfAnimationProperty::Rotation,
                    gltf_dep::animation::Property::Scale => GltfAnimationProperty::Scale,
                    gltf_dep::animation::Property::MorphTargetWeights => {
                        GltfAnimationProperty::MorphTargetWeights
                    }
                };

                let sampler = channel.sampler();
                let interpolation = match sampler.interpolation() {
                    gltf_dep::animation::Interpolation::Linear => GltfInterpolation::Linear,
                    gltf_dep::animation::Interpolation::Step => GltfInterpolation::Step,
                    gltf_dep::animation::Interpolation::CubicSpline => {
                        GltfInterpolation::CubicSpline
                    }
                };

                let input_accessor = sampler.input();
                let output_accessor = sampler.output();

                let timestamps = read_f32_accessor(&input_accessor, &self.buffers)?;
                let values = read_f32_accessor(&output_accessor, &self.buffers)?;

                channels.push(GltfAnimationChannel {
                    target_node,
                    property,
                    interpolation,
                    timestamps,
                    values,
                });
            }

            result.push(GltfAnimation {
                name: anim.name().map(String::from),
                channels,
            });
        }

        Ok(result)
    }

    /// Load all scenes as node trees.
    ///
    /// Must be called after `load_meshes` so that `mesh_index_map` is populated.
    pub fn load_scenes(&self) -> Vec<GltfScene> {
        self.document
            .scenes()
            .map(|scene| GltfScene {
                name: scene.name().map(String::from),
                nodes: scene
                    .nodes()
                    .map(|n| load_node(&n, &self.mesh_index_map))
                    .collect(),
            })
            .collect()
    }

    /// Get the default scene index.
    pub fn default_scene(&self) -> Option<usize> {
        self.document.default_scene().map(|s| s.index())
    }

    /// Consume the context and return new layouts.
    pub fn into_new_layouts(self) -> Vec<Arc<VertexLayout>> {
        self.new_layouts
    }
}

// -- Helper functions --

/// Recursively load a node and its children.
///
/// `mesh_index_map` maps glTF mesh index → list of flat CpuMesh indices.
fn load_node(node: &gltf_dep::Node<'_>, mesh_index_map: &[Vec<usize>]) -> GltfNode {
    let (translation, rotation, scale) = node.transform().decomposed();

    let meshes = node
        .mesh()
        .map(|m| mesh_index_map[m.index()].clone())
        .unwrap_or_default();

    GltfNode {
        name: node.name().map(String::from),
        transform: GltfTransform {
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

/// Decode image bytes to RGBA8 using the `image` crate.
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

/// Map glTF magnification filter.
fn map_mag_filter(filter: gltf_dep::texture::MagFilter) -> GltfFilter {
    match filter {
        gltf_dep::texture::MagFilter::Nearest => GltfFilter::Nearest,
        gltf_dep::texture::MagFilter::Linear => GltfFilter::Linear,
    }
}

/// Map glTF minification filter (collapse mipmap variants to Nearest/Linear).
fn map_min_filter(filter: gltf_dep::texture::MinFilter) -> GltfFilter {
    match filter {
        gltf_dep::texture::MinFilter::Nearest
        | gltf_dep::texture::MinFilter::NearestMipmapNearest
        | gltf_dep::texture::MinFilter::NearestMipmapLinear => GltfFilter::Nearest,
        gltf_dep::texture::MinFilter::Linear
        | gltf_dep::texture::MinFilter::LinearMipmapNearest
        | gltf_dep::texture::MinFilter::LinearMipmapLinear => GltfFilter::Linear,
    }
}

/// Map glTF wrapping mode.
fn map_wrapping(wrap: gltf_dep::texture::WrappingMode) -> GltfWrapping {
    match wrap {
        gltf_dep::texture::WrappingMode::ClampToEdge => GltfWrapping::ClampToEdge,
        gltf_dep::texture::WrappingMode::MirroredRepeat => GltfWrapping::MirroredRepeat,
        gltf_dep::texture::WrappingMode::Repeat => GltfWrapping::Repeat,
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
        let t = GltfTransform::default();
        assert_eq!(t.translation, [0.0, 0.0, 0.0]);
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
    }
}
