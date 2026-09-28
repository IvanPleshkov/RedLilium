//! Conversion helpers between f32 rendering types and physics-precision types.
//!
//! Also includes utilities for extracting physics collider data from [`CpuMesh`](redlilium_core::mesh::CpuMesh).

use redlilium_core::math::{Quat, Real, Vec2, Vec3, quat_from_xyzw, quat_to_array};

/// Converts a rendering `Vec3` (f32) to a physics `Vector3<Real>`.
pub fn vec3_to_na(v: Vec3) -> redlilium_core::math::Vector3 {
    redlilium_core::math::Vector3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// Converts a physics `Vector3<Real>` to a rendering `Vec3` (f32).
pub fn vec3_from_na(v: &redlilium_core::math::Vector3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Converts a rendering `Vec3` (f32) to a physics `Point3<Real>`.
pub fn point3_to_na(v: Vec3) -> redlilium_core::math::Point3 {
    redlilium_core::math::Point3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// Converts a physics `Point3<Real>` to a rendering `Vec3` (f32).
pub fn point3_from_na(p: &redlilium_core::math::Point3) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

/// Converts a rendering `Vec2` (f32) to a physics `Vector2<Real>`.
pub fn vec2_to_na(v: Vec2) -> redlilium_core::math::Vector2 {
    redlilium_core::math::Vector2::new(v.x as Real, v.y as Real)
}

/// Converts a physics `Vector2<Real>` to a rendering `Vec2` (f32).
pub fn vec2_from_na(v: &redlilium_core::math::Vector2) -> Vec2 {
    Vec2::new(v.x as f32, v.y as f32)
}

/// Converts a rendering `Quat` (f32) to a physics `UnitQuaternion<Real>`.
pub fn quat_to_na(q: Quat) -> redlilium_core::math::UnitQuaternion {
    use redlilium_core::math::nalgebra;
    let arr = quat_to_array(q);
    let quat = nalgebra::Quaternion::new(
        arr[3] as Real,
        arr[0] as Real,
        arr[1] as Real,
        arr[2] as Real,
    );
    // Normalize before handing to rapier: a drifted (non-unit) Transform
    // rotation would otherwise corrupt the simulation's orientation.
    redlilium_core::math::UnitQuaternion::new_normalize(quat)
}

/// Converts a physics `UnitQuaternion<Real>` to a rendering `Quat` (f32).
pub fn quat_from_na(q: &redlilium_core::math::UnitQuaternion) -> Quat {
    let q = q.quaternion();
    quat_from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32)
}

/// Converts a rendering `Vec3` + `Quat` to a physics `Isometry3<Real>`.
pub fn isometry3_to_na(translation: Vec3, rotation: Quat) -> redlilium_core::math::Isometry3 {
    redlilium_core::math::Isometry3::from_parts(
        redlilium_core::math::Translation3::new(
            translation.x as Real,
            translation.y as Real,
            translation.z as Real,
        ),
        quat_to_na(rotation),
    )
}

/// Extracts position and rotation from a physics `Isometry3<Real>` as `(Vec3, Quat)`.
pub fn isometry3_from_na(iso: &redlilium_core::math::Isometry3) -> (Vec3, Quat) {
    let t = &iso.translation;
    let pos = Vec3::new(t.x as f32, t.y as f32, t.z as f32);
    let rot = quat_from_na(&iso.rotation);
    (pos, rot)
}

/// Converts a rendering `Vec2` + angle to a physics `Isometry2<Real>`.
pub fn isometry2_to_na(translation: Vec2, angle: f32) -> redlilium_core::math::Isometry2 {
    redlilium_core::math::Isometry2::new(
        redlilium_core::math::Vector2::new(translation.x as Real, translation.y as Real),
        angle as Real,
    )
}

/// Extracts position and angle from a physics `Isometry2<Real>` as `(Vec2, f32)`.
pub fn isometry2_from_na(iso: &redlilium_core::math::Isometry2) -> (Vec2, f32) {
    let t = &iso.translation;
    let pos = Vec2::new(t.x as f32, t.y as f32);
    let angle = iso.rotation.angle() as f32;
    (pos, angle)
}

/// Extracts triangle mesh data (positions, triangle indices) from a [`CpuMesh`](redlilium_core::mesh::CpuMesh)
/// suitable for creating a trimesh collider.
///
/// Supports indexed `TriangleList` meshes with per-vertex `Float3` positions.
/// Returns `None` for missing/unsupported data, empty geometry, inconsistent
/// buffer sizes or counts, out-of-range indices, or non-finite positions.
/// Position attributes may have an offset and live in any vertex buffer slot.
pub fn extract_trimesh_data(
    mesh: &redlilium_core::mesh::CpuMesh,
) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    use redlilium_core::mesh::{
        IndexFormat, PrimitiveTopology, VertexAttributeFormat, VertexAttributeSemantic,
        VertexStepMode,
    };

    if mesh.topology() != PrimitiveTopology::TriangleList {
        return None;
    }
    let layout = mesh.layout();
    let pos_attr = layout
        .attributes
        .iter()
        .find(|a| a.semantic == VertexAttributeSemantic::Position)?;
    if pos_attr.format != VertexAttributeFormat::Float3 {
        return None;
    }
    let buffer_index = pos_attr.buffer_index as usize;
    let buffer = layout.buffers.get(buffer_index)?;
    let vertex_data = mesh.vertex_buffer_data(buffer_index)?;
    let stride = buffer.stride as usize;
    let offset = pos_attr.offset as usize;
    let vertex_count = mesh.vertex_count() as usize;
    if buffer.step_mode != VertexStepMode::Vertex
        || stride == 0
        || offset.checked_add(12)? > stride
        || vertex_count == 0
        || vertex_data.len() != vertex_count.checked_mul(stride)?
    {
        return None;
    }

    let indices_raw = mesh.index_data()?;
    let index_format = mesh.index_format()?;
    let index_size = match index_format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    };
    let index_count = mesh.index_count() as usize;
    if index_count == 0
        || !index_count.is_multiple_of(3)
        || indices_raw.len() != index_count.checked_mul(index_size)?
    {
        return None;
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    for vertex in vertex_data.chunks_exact(stride) {
        let position = &vertex[offset..offset + 12];
        let p = Vec3::new(
            f32::from_le_bytes(position[0..4].try_into().ok()?),
            f32::from_le_bytes(position[4..8].try_into().ok()?),
            f32::from_le_bytes(position[8..12].try_into().ok()?),
        );
        if !p.iter().all(|v| v.is_finite()) {
            return None;
        }
        vertices.push(p);
    }

    // Decode straight into triangles, without a temporary flat index allocation.
    let mut triangles = Vec::with_capacity(index_count / 3);
    for triangle in indices_raw.chunks_exact(3 * index_size) {
        let mut indices = [0; 3];
        for (dst, src) in indices.iter_mut().zip(triangle.chunks_exact(index_size)) {
            *dst = match index_format {
                IndexFormat::Uint16 => u16::from_le_bytes(src.try_into().ok()?) as u32,
                IndexFormat::Uint32 => u32::from_le_bytes(src.try_into().ok()?),
            };
            if *dst as usize >= vertex_count {
                return None;
            }
        }
        triangles.push(indices);
    }
    Some((vertices, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::quat_from_rotation_y;

    #[test]
    fn vec3_roundtrip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let na = vec3_to_na(v);
        let back = vec3_from_na(&na);
        assert!((v - back).norm() < 1e-6);
    }

    #[test]
    fn quat_roundtrip() {
        let q = quat_from_rotation_y(1.0);
        let na = quat_to_na(q);
        let back = quat_from_na(&na);
        assert!((q.coords - back.coords).norm() < 1e-5);
    }

    #[test]
    fn isometry3_roundtrip() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let rot = redlilium_core::math::quat_from_rotation_z(0.5);
        let iso = isometry3_to_na(pos, rot);
        let (pos2, rot2) = isometry3_from_na(&iso);
        assert!((pos - pos2).norm() < 1e-5);
        assert!((rot.coords - rot2.coords).norm() < 1e-5);
    }

    #[test]
    fn isometry2_roundtrip() {
        let pos = Vec2::new(1.0, 2.0);
        let angle = 0.7f32;
        let iso = isometry2_to_na(pos, angle);
        let (pos2, angle2) = isometry2_from_na(&iso);
        assert!((pos - pos2).norm() < 1e-5);
        assert!((angle - angle2).abs() < 1e-5);
    }
}

#[cfg(test)]
mod mesh_tests {
    use super::*;
    use redlilium_core::mesh::{
        CpuMesh, CpuMeshData, IndexFormat, PrimitiveTopology, VertexAttribute,
        VertexAttributeFormat, VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    };
    use std::sync::Arc;

    fn triangle() -> CpuMesh {
        CpuMesh::new(VertexLayout::position_only())
            .with_vertex_data(
                0,
                [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )
            .with_indices_u32(&[0, 1, 2])
    }

    #[test]
    fn extracts_positions_with_offset_in_a_separate_buffer_and_both_index_formats() {
        let layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(8))
                .with_buffer(VertexBufferLayout::new(20))
                .with_attribute(VertexAttribute::new(
                    VertexAttributeSemantic::Position,
                    VertexAttributeFormat::Float3,
                    4,
                    1,
                )),
        );
        let vertices = [
            99.0_f32, 0.0, 0.0, 0.0, 99.0, 99.0, 1.0, 0.0, 0.0, 99.0, 99.0, 0.0, 1.0, 0.0, 99.0,
        ];
        let mesh = CpuMesh::new(layout)
            .with_vertex_data(0, vec![0; 24])
            .with_vertex_data(1, vertices.into_iter().flat_map(f32::to_le_bytes).collect());
        for mesh in [
            mesh.clone().with_indices_u16(&[2, 0, 1]),
            mesh.with_indices_u32(&[2, 0, 1]),
        ] {
            let (positions, triangles) = extract_trimesh_data(&mesh).unwrap();
            assert_eq!(
                positions,
                vec![
                    Vec3::zeros(),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0)
                ]
            );
            assert_eq!(triangles, vec![[2, 0, 1]]);
        }
    }

    #[test]
    fn rejects_non_triangle_topology_and_non_float3_positions() {
        for topology in [
            PrimitiveTopology::LineList,
            PrimitiveTopology::TriangleStrip,
        ] {
            assert!(extract_trimesh_data(&triangle().with_topology(topology)).is_none());
        }
        let layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(8))
                .with_attribute(VertexAttribute::new(
                    VertexAttributeSemantic::Position,
                    VertexAttributeFormat::Float2,
                    0,
                    0,
                )),
        );
        let mesh = CpuMesh::new(layout)
            .with_vertex_data(0, vec![0; 24])
            .with_indices_u32(&[0, 1, 2]);
        assert!(extract_trimesh_data(&mesh).is_none());
    }

    #[test]
    fn rejects_truncated_mismatched_or_out_of_range_indices() {
        for format in [IndexFormat::Uint16, IndexFormat::Uint32] {
            let valid = match format {
                IndexFormat::Uint16 => triangle().with_indices_u16(&[0, 1, 2]),
                IndexFormat::Uint32 => triangle(),
            };
            let bytes = valid.index_data().unwrap();
            for (data, count) in [
                (bytes.to_vec(), 2),
                (bytes.to_vec(), 6),
                (bytes[..bytes.len() - 1].to_vec(), 3),
                ([bytes, &[0]].concat(), 3),
                (Vec::new(), 0),
            ] {
                assert!(
                    extract_trimesh_data(&valid.clone().with_raw_index_data(data, format, count))
                        .is_none()
                );
            }
        }
        assert!(extract_trimesh_data(&triangle().with_indices_u16(&[0, 1, 3])).is_none());
        assert!(extract_trimesh_data(&triangle().with_indices_u32(&[0, 1, u32::MAX])).is_none());
        let mut data = CpuMeshData::from_cpu_mesh(&triangle());
        data.index_data = None;
        assert!(extract_trimesh_data(&data.into_cpu_mesh(VertexLayout::position_only())).is_none());
    }

    #[test]
    fn rejects_malformed_vertex_storage_and_non_finite_positions() {
        let valid = triangle();
        let data = CpuMeshData::from_cpu_mesh(&valid);
        let mut cases = Vec::new();
        for count in [0, 2, 4] {
            let mut d = data.clone();
            d.vertex_count = count;
            cases.push(d);
        }
        let mut d = data.clone();
        d.vertex_buffers[0].pop();
        cases.push(d);
        let mut d = data.clone();
        d.vertex_buffers[0].push(0);
        cases.push(d);
        let mut d = data.clone();
        d.vertex_buffers.clear();
        cases.push(d);
        for value in [f32::NAN, f32::INFINITY] {
            let mut d = data.clone();
            d.vertex_buffers[0][..4].copy_from_slice(&value.to_le_bytes());
            cases.push(d);
        }
        for data in cases {
            assert!(extract_trimesh_data(&data.into_cpu_mesh(valid.layout().clone())).is_none());
        }
    }

    #[test]
    fn rejects_invalid_position_layouts() {
        let data = CpuMeshData::from_cpu_mesh(&triangle());
        for (stride, offset, slot, instance) in [
            (0, 0, 0, false),
            (8, 0, 0, false),
            (12, 4, 0, false),
            (12, u32::MAX, 0, false),
            (12, 0, 1, false),
            (12, 0, 0, true),
        ] {
            let mut buffer = VertexBufferLayout::new(stride);
            if instance {
                buffer = buffer.with_instance_step();
            }
            let layout = Arc::new(VertexLayout::new().with_buffer(buffer).with_attribute(
                VertexAttribute::new(
                    VertexAttributeSemantic::Position,
                    VertexAttributeFormat::Float3,
                    offset,
                    slot,
                ),
            ));
            assert!(extract_trimesh_data(&data.clone().into_cpu_mesh(layout)).is_none());
        }
        let layout = Arc::new(VertexLayout::new().with_buffer(VertexBufferLayout::new(12)));
        assert!(extract_trimesh_data(&data.into_cpu_mesh(layout)).is_none());
    }
}
