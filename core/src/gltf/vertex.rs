//! Vertex layout building, sharing, and data interleaving for glTF primitives.

use std::sync::Arc;

use crate::mesh::{
    CpuMesh, IndexFormat, PrimitiveTopology, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
};

use super::error::GltfError;

/// Map a glTF semantic string to our engine's VertexAttributeSemantic.
fn map_semantic(semantic: gltf_dep::Semantic) -> Option<VertexAttributeSemantic> {
    match semantic {
        gltf_dep::Semantic::Positions => Some(VertexAttributeSemantic::Position),
        gltf_dep::Semantic::Normals => Some(VertexAttributeSemantic::Normal),
        gltf_dep::Semantic::Tangents => Some(VertexAttributeSemantic::Tangent),
        gltf_dep::Semantic::Colors(0) => Some(VertexAttributeSemantic::Color),
        gltf_dep::Semantic::TexCoords(0) => Some(VertexAttributeSemantic::TexCoord0),
        gltf_dep::Semantic::TexCoords(1) => Some(VertexAttributeSemantic::TexCoord1),
        gltf_dep::Semantic::Joints(0) => Some(VertexAttributeSemantic::Joints),
        gltf_dep::Semantic::Weights(0) => Some(VertexAttributeSemantic::Weights),
        _ => None, // Ignore additional sets (Colors(1+), TexCoords(2+), etc.)
    }
}

/// Determine the engine vertex format for a glTF accessor.
fn accessor_format(accessor: &gltf_dep::Accessor) -> Option<VertexAttributeFormat> {
    use gltf_dep::accessor::{DataType, Dimensions};

    match (accessor.dimensions(), accessor.data_type()) {
        (Dimensions::Scalar, DataType::F32) => Some(VertexAttributeFormat::Float),
        (Dimensions::Vec2, DataType::F32) => Some(VertexAttributeFormat::Float2),
        (Dimensions::Vec3, DataType::F32) => Some(VertexAttributeFormat::Float3),
        (Dimensions::Vec4, DataType::F32) => Some(VertexAttributeFormat::Float4),
        (Dimensions::Vec4, DataType::U8) => Some(VertexAttributeFormat::Unorm8x4),
        (Dimensions::Vec4, DataType::U16) => Some(VertexAttributeFormat::Uint4),
        _ => None,
    }
}

/// Information about one attribute collected from a glTF primitive.
pub(super) struct AttributeInfo {
    semantic: VertexAttributeSemantic,
    format: VertexAttributeFormat,
    offset: u32,
    accessor_index: usize,
}

/// Build a VertexLayout from a glTF primitive's attributes.
///
/// Returns the layout and per-attribute info needed for interleaving.
pub(crate) fn build_layout_from_primitive(
    primitive: &gltf_dep::Primitive<'_>,
) -> Result<(VertexLayout, Vec<AttributeInfo>), GltfError> {
    let mut attrs = Vec::new();
    let mut offset: u32 = 0;

    for (semantic, accessor) in primitive.attributes() {
        let Some(engine_semantic) = map_semantic(semantic.clone()) else {
            continue;
        };
        let Some(format) = accessor_format(&accessor) else {
            log::warn!(
                "Skipping attribute {:?}: unsupported format {:?}/{:?}",
                semantic,
                accessor.dimensions(),
                accessor.data_type()
            );
            continue;
        };

        attrs.push(AttributeInfo {
            semantic: engine_semantic,
            format,
            offset,
            accessor_index: accessor.index(),
        });
        offset += format.size() as u32;
    }

    let stride = offset;
    let mut layout = VertexLayout::new().with_buffer(VertexBufferLayout::new(stride));

    for attr in &attrs {
        layout = layout.with_attribute(VertexAttribute::new(
            attr.semantic,
            attr.format,
            attr.offset,
            0, // single interleaved buffer
        ));
    }

    Ok((layout, attrs))
}

/// Check if two layouts are structurally equal (ignoring label).
///
/// Compares buffer count, strides, step modes, and all attributes
/// (order-independent for attributes).
fn layouts_structurally_equal(a: &VertexLayout, b: &VertexLayout) -> bool {
    if a.buffers.len() != b.buffers.len() {
        return false;
    }
    for (ab, bb) in a.buffers.iter().zip(b.buffers.iter()) {
        if ab.stride != bb.stride || ab.step_mode != bb.step_mode {
            return false;
        }
    }
    if a.attributes.len() != b.attributes.len() {
        return false;
    }
    // Check that every attribute in a has a matching one in b
    a.attributes.iter().all(|aa| {
        b.attributes.iter().any(|ba| {
            aa.semantic == ba.semantic
                && aa.format == ba.format
                && aa.offset == ba.offset
                && aa.buffer_index == ba.buffer_index
        })
    })
}

/// Find or create a shared layout.
///
/// Searches `existing_layouts` for a structural match. If found, returns the
/// existing Arc. Otherwise, creates a new Arc and appends it to `new_layouts`.
pub(crate) fn find_or_create_layout(
    layout: VertexLayout,
    existing_layouts: &[Arc<VertexLayout>],
    new_layouts: &mut Vec<Arc<VertexLayout>>,
) -> Arc<VertexLayout> {
    // Search in pre-existing shared layouts
    for existing in existing_layouts {
        if layouts_structurally_equal(&layout, existing) {
            return Arc::clone(existing);
        }
    }
    // Search in newly created layouts
    for new_layout in new_layouts.iter() {
        if layouts_structurally_equal(&layout, new_layout) {
            return Arc::clone(new_layout);
        }
    }
    // Create a new one
    let arc = Arc::new(layout);
    new_layouts.push(Arc::clone(&arc));
    arc
}

/// Build adapted AttributeInfo list to match a target vertex layout.
///
/// For each attribute in the target layout's buffer 0, finds the matching
/// semantic in the primitive's native attributes. If found, creates an
/// `AttributeInfo` with the target layout's offset but the primitive's
/// accessor index. Missing attributes are omitted (zero-filled by the
/// caller since the output buffer starts zeroed).
///
/// Attributes in the primitive but not in the target layout are skipped.
pub(crate) fn adapt_attrs_to_target_layout(
    target_layout: &VertexLayout,
    native_attrs: &[AttributeInfo],
) -> Vec<AttributeInfo> {
    let mut adapted = Vec::new();

    for target_attr in target_layout.attributes_for_buffer(0) {
        if let Some(native) = native_attrs
            .iter()
            .find(|a| a.semantic == target_attr.semantic)
        {
            adapted.push(AttributeInfo {
                semantic: target_attr.semantic,
                format: target_attr.format,
                offset: target_attr.offset,
                accessor_index: native.accessor_index,
            });
        }
    }

    adapted
}

/// Map glTF primitive mode to our PrimitiveTopology.
pub(crate) fn map_topology(mode: gltf_dep::mesh::Mode) -> Result<PrimitiveTopology, GltfError> {
    match mode {
        gltf_dep::mesh::Mode::Points => Ok(PrimitiveTopology::PointList),
        gltf_dep::mesh::Mode::Lines => Ok(PrimitiveTopology::LineList),
        gltf_dep::mesh::Mode::LineStrip => Ok(PrimitiveTopology::LineStrip),
        gltf_dep::mesh::Mode::Triangles => Ok(PrimitiveTopology::TriangleList),
        gltf_dep::mesh::Mode::TriangleStrip => Ok(PrimitiveTopology::TriangleStrip),
        other => Err(GltfError::UnsupportedTopology(format!("{other:?}"))),
    }
}

/// Read accessor data as a flat slice of bytes from resolved buffers.
fn read_accessor_bytes<'a>(
    accessor: &gltf_dep::Accessor,
    buffers: &'a [Vec<u8>],
) -> Result<(&'a [u8], usize), GltfError> {
    let view = accessor.view().ok_or_else(|| {
        GltfError::AccessorError(format!(
            "accessor {} has no buffer view (sparse accessors not supported)",
            accessor.index()
        ))
    })?;
    let buffer_index = view.buffer().index();
    let buffer_data = buffers.get(buffer_index).ok_or_else(|| {
        GltfError::BufferError(format!("buffer index {buffer_index} out of range"))
    })?;

    let view_offset = view.offset();
    let accessor_offset = accessor.offset();
    let start = view_offset + accessor_offset;

    let component_size = accessor.data_type().size();
    let component_count = accessor.dimensions().multiplicity();
    let element_size = component_size * component_count;
    let stride = view.stride().unwrap_or(element_size);

    Ok((&buffer_data[start..], stride))
}

/// Interleave vertex data from glTF accessors into a single buffer.
///
/// For each vertex, writes each attribute's data at the correct offset
/// within the interleaved stride.
pub(crate) fn interleave_vertices(
    attrs: &[AttributeInfo],
    vertex_count: u32,
    stride: u32,
    accessors: &[gltf_dep::Accessor],
    buffers: &[Vec<u8>],
) -> Result<Vec<u8>, GltfError> {
    let total_size = vertex_count as usize * stride as usize;
    let mut result = vec![0u8; total_size];

    for attr in attrs {
        let accessor = &accessors[attr.accessor_index];
        let (src_data, src_stride) = read_accessor_bytes(accessor, buffers)?;

        let component_size = accessor.data_type().size();
        let component_count = accessor.dimensions().multiplicity();
        let element_size = component_size * component_count;

        // Handle type conversion for specific cases
        let needs_u8_to_f32 =
            accessor.data_type() == gltf_dep::accessor::DataType::U8 && accessor.normalized();
        let needs_u16_to_f32 =
            accessor.data_type() == gltf_dep::accessor::DataType::U16 && accessor.normalized();

        if needs_u8_to_f32 && attr.format == VertexAttributeFormat::Float4 {
            // Convert normalized u8x4 → float4
            for v in 0..vertex_count as usize {
                let src_offset = v * src_stride;
                let dst_offset = v * stride as usize + attr.offset as usize;
                if src_offset + 4 <= src_data.len() && dst_offset + 16 <= result.len() {
                    for c in 0..4 {
                        let val = src_data[src_offset + c] as f32 / 255.0;
                        result[dst_offset + c * 4..dst_offset + c * 4 + 4]
                            .copy_from_slice(&val.to_le_bytes());
                    }
                }
            }
        } else if needs_u16_to_f32 && attr.format == VertexAttributeFormat::Float4 {
            // Convert normalized u16x4 → float4
            for v in 0..vertex_count as usize {
                let src_offset = v * src_stride;
                let dst_offset = v * stride as usize + attr.offset as usize;
                if src_offset + 8 <= src_data.len() && dst_offset + 16 <= result.len() {
                    for c in 0..4 {
                        let bytes = [
                            src_data[src_offset + c * 2],
                            src_data[src_offset + c * 2 + 1],
                        ];
                        let val = u16::from_le_bytes(bytes) as f32 / 65535.0;
                        result[dst_offset + c * 4..dst_offset + c * 4 + 4]
                            .copy_from_slice(&val.to_le_bytes());
                    }
                }
            }
        } else {
            // Direct copy (most common path: f32 data)
            for v in 0..vertex_count as usize {
                let src_offset = v * src_stride;
                let dst_offset = v * stride as usize + attr.offset as usize;
                let copy_len = element_size.min(attr.format.size());
                if src_offset + copy_len <= src_data.len() && dst_offset + copy_len <= result.len()
                {
                    result[dst_offset..dst_offset + copy_len]
                        .copy_from_slice(&src_data[src_offset..src_offset + copy_len]);
                }
            }
        }
    }

    Ok(result)
}

/// Read index data from a glTF accessor, converting to u16 or u32.
pub(crate) fn read_indices(
    accessor: &gltf_dep::Accessor,
    buffers: &[Vec<u8>],
    vertex_count: u32,
) -> Result<(Vec<u8>, IndexFormat, u32), GltfError> {
    let (src_data, src_stride) = read_accessor_bytes(accessor, buffers)?;
    let count = accessor.count() as u32;

    // Choose output format based on vertex count
    let output_format = if vertex_count > 65535 {
        IndexFormat::Uint32
    } else {
        IndexFormat::Uint16
    };

    match accessor.data_type() {
        gltf_dep::accessor::DataType::U8 => {
            let mut indices = Vec::with_capacity(count as usize);
            for i in 0..count as usize {
                indices.push(src_data[i * src_stride] as u32);
            }
            match output_format {
                IndexFormat::Uint16 => {
                    let u16_indices: Vec<u16> = indices.iter().map(|&i| i as u16).collect();
                    Ok((
                        bytemuck::cast_slice(&u16_indices).to_vec(),
                        IndexFormat::Uint16,
                        count,
                    ))
                }
                IndexFormat::Uint32 => Ok((
                    bytemuck::cast_slice(&indices).to_vec(),
                    IndexFormat::Uint32,
                    count,
                )),
            }
        }
        gltf_dep::accessor::DataType::U16 => {
            if output_format == IndexFormat::Uint16 {
                // Direct copy
                let mut result = Vec::with_capacity(count as usize * 2);
                for i in 0..count as usize {
                    let offset = i * src_stride;
                    result.extend_from_slice(&src_data[offset..offset + 2]);
                }
                Ok((result, IndexFormat::Uint16, count))
            } else {
                // Convert u16 → u32
                let mut indices = Vec::with_capacity(count as usize);
                for i in 0..count as usize {
                    let offset = i * src_stride;
                    let val = u16::from_le_bytes([src_data[offset], src_data[offset + 1]]) as u32;
                    indices.push(val);
                }
                Ok((
                    bytemuck::cast_slice(&indices).to_vec(),
                    IndexFormat::Uint32,
                    count,
                ))
            }
        }
        gltf_dep::accessor::DataType::U32 => {
            let mut result = Vec::with_capacity(count as usize * 4);
            for i in 0..count as usize {
                let offset = i * src_stride;
                result.extend_from_slice(&src_data[offset..offset + 4]);
            }
            Ok((result, IndexFormat::Uint32, count))
        }
        other => Err(GltfError::AccessorError(format!(
            "unsupported index type: {other:?}"
        ))),
    }
}

/// Build a CpuMesh from a glTF primitive.
pub(crate) fn build_cpu_mesh(
    layout: Arc<VertexLayout>,
    topology: PrimitiveTopology,
    vertex_data: Vec<u8>,
    index_data: Option<(Vec<u8>, IndexFormat, u32)>,
    label: Option<String>,
) -> CpuMesh {
    let mut mesh = CpuMesh::new(layout)
        .with_vertex_data(0, vertex_data)
        .with_topology(topology);

    if let Some((data, format, count)) = index_data {
        match format {
            IndexFormat::Uint16 => {
                // Set raw index data directly
                mesh = mesh.with_raw_index_data(data, IndexFormat::Uint16, count);
            }
            IndexFormat::Uint32 => {
                mesh = mesh.with_raw_index_data(data, IndexFormat::Uint32, count);
            }
        }
    }

    if let Some(label) = label {
        mesh = mesh.with_label(label);
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layouts_structurally_equal() {
        let a = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12))
            .with_attribute(VertexAttribute::texcoord0(24))
            .with_label("layout_a");

        let b = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12))
            .with_attribute(VertexAttribute::texcoord0(24))
            .with_label("layout_b");

        // Labels differ but structure is the same
        assert!(layouts_structurally_equal(&a, &b));
    }

    #[test]
    fn test_layouts_not_equal_different_stride() {
        let a = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0));

        let b = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(24))
            .with_attribute(VertexAttribute::position(0));

        assert!(!layouts_structurally_equal(&a, &b));
    }

    #[test]
    fn test_layouts_not_equal_different_attrs() {
        let a = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(24))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        let b = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(24))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::texcoord0(12));

        assert!(!layouts_structurally_equal(&a, &b));
    }

    #[test]
    fn test_find_or_create_layout_shares_existing() {
        let existing = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(32))
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_attribute(VertexAttribute::texcoord0(24))
                .with_label("existing"),
        );

        let new_layout = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12))
            .with_attribute(VertexAttribute::texcoord0(24));

        let shared = &[Arc::clone(&existing)];
        let mut new_layouts = Vec::new();

        let result = find_or_create_layout(new_layout, shared, &mut new_layouts);

        // Should return the existing Arc (same pointer)
        assert!(Arc::ptr_eq(&result, &existing));
        assert!(new_layouts.is_empty());
    }

    #[test]
    fn test_find_or_create_layout_creates_new() {
        let existing = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(12))
                .with_attribute(VertexAttribute::position(0)),
        );

        let new_layout = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12))
            .with_attribute(VertexAttribute::texcoord0(24));

        let shared = &[existing];
        let mut new_layouts = Vec::new();

        let result = find_or_create_layout(new_layout, shared, &mut new_layouts);

        assert_eq!(new_layouts.len(), 1);
        assert!(Arc::ptr_eq(&result, &new_layouts[0]));
    }

    #[test]
    fn test_find_or_create_layout_reuses_new() {
        let layout1 = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        let layout2 = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        let shared: &[Arc<VertexLayout>] = &[];
        let mut new_layouts = Vec::new();

        let r1 = find_or_create_layout(layout1, shared, &mut new_layouts);
        let r2 = find_or_create_layout(layout2, shared, &mut new_layouts);

        assert_eq!(new_layouts.len(), 1);
        assert!(Arc::ptr_eq(&r1, &r2));
    }
}
