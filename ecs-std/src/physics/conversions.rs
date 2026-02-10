//! Conversion helpers between `glam` types and `nalgebra` types.
//!
//! Also includes utilities for extracting physics collider data from [`CpuMesh`].

use redlilium_core::math::Real;

/// Converts a `glam::Vec3` to a `nalgebra::Vector3<Real>`.
pub fn vec3_to_na(v: glam::Vec3) -> redlilium_core::math::Vector3 {
    redlilium_core::math::Vector3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// Converts a `nalgebra::Vector3<Real>` to a `glam::Vec3`.
pub fn vec3_from_na(v: &redlilium_core::math::Vector3) -> glam::Vec3 {
    glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Converts a `glam::Vec3` to a `nalgebra::Point3<Real>`.
pub fn point3_to_na(v: glam::Vec3) -> redlilium_core::math::Point3 {
    redlilium_core::math::Point3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// Converts a `nalgebra::Point3<Real>` to a `glam::Vec3`.
pub fn point3_from_na(p: &redlilium_core::math::Point3) -> glam::Vec3 {
    glam::Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

/// Converts a `glam::Vec2` to a `nalgebra::Vector2<Real>`.
pub fn vec2_to_na(v: glam::Vec2) -> redlilium_core::math::Vector2 {
    redlilium_core::math::Vector2::new(v.x as Real, v.y as Real)
}

/// Converts a `nalgebra::Vector2<Real>` to a `glam::Vec2`.
pub fn vec2_from_na(v: &redlilium_core::math::Vector2) -> glam::Vec2 {
    glam::Vec2::new(v.x as f32, v.y as f32)
}

/// Converts a `glam::Quat` to a `nalgebra::UnitQuaternion<Real>`.
pub fn quat_to_na(q: glam::Quat) -> redlilium_core::math::UnitQuaternion {
    use redlilium_core::math::nalgebra;
    let quat = nalgebra::Quaternion::new(q.w as Real, q.x as Real, q.y as Real, q.z as Real);
    redlilium_core::math::UnitQuaternion::new_unchecked(quat)
}

/// Converts a `nalgebra::UnitQuaternion<Real>` to a `glam::Quat`.
pub fn quat_from_na(q: &redlilium_core::math::UnitQuaternion) -> glam::Quat {
    let q = q.quaternion();
    glam::Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32)
}

/// Converts a `glam::Vec3` translation and `glam::Quat` rotation to a `nalgebra::Isometry3<Real>`.
pub fn isometry3_to_na(
    translation: glam::Vec3,
    rotation: glam::Quat,
) -> redlilium_core::math::Isometry3 {
    redlilium_core::math::Isometry3::from_parts(
        redlilium_core::math::Translation3::new(
            translation.x as Real,
            translation.y as Real,
            translation.z as Real,
        ),
        quat_to_na(rotation),
    )
}

/// Extracts position and rotation from a `nalgebra::Isometry3<Real>` as `(glam::Vec3, glam::Quat)`.
pub fn isometry3_from_na(iso: &redlilium_core::math::Isometry3) -> (glam::Vec3, glam::Quat) {
    let t = &iso.translation;
    let pos = glam::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
    let rot = quat_from_na(&iso.rotation);
    (pos, rot)
}

/// Converts a `glam::Vec2` translation and angle to a `nalgebra::Isometry2<Real>`.
pub fn isometry2_to_na(translation: glam::Vec2, angle: f32) -> redlilium_core::math::Isometry2 {
    redlilium_core::math::Isometry2::new(
        redlilium_core::math::Vector2::new(translation.x as Real, translation.y as Real),
        angle as Real,
    )
}

/// Extracts position and angle from a `nalgebra::Isometry2<Real>` as `(glam::Vec2, f32)`.
pub fn isometry2_from_na(iso: &redlilium_core::math::Isometry2) -> (glam::Vec2, f32) {
    let t = &iso.translation;
    let pos = glam::Vec2::new(t.x as f32, t.y as f32);
    let angle = iso.rotation.angle() as f32;
    (pos, angle)
}

/// Extracts triangle mesh data (positions, triangle indices) from a [`CpuMesh`]
/// suitable for creating a trimesh collider.
///
/// Returns `None` if the mesh has no position data or no index buffer.
///
/// Positions are extracted from the first vertex buffer using the mesh layout's
/// position attribute offset and stride. Returned as `glam::Vec3` — convert to
/// the rapier `Vector` type as needed.
pub fn extract_trimesh_data(
    mesh: &redlilium_core::mesh::CpuMesh,
) -> Option<(Vec<glam::Vec3>, Vec<[u32; 3]>)> {
    use redlilium_core::mesh::{IndexFormat, VertexAttributeSemantic};

    let layout = mesh.layout();

    // Find the position attribute
    let pos_attr = layout
        .attributes
        .iter()
        .find(|a| a.semantic == VertexAttributeSemantic::Position)?;

    let buffer_index = pos_attr.buffer_index as usize;
    let vertex_data = mesh.vertex_buffer_data(buffer_index)?;
    let stride = layout.buffers.get(buffer_index)?.stride as usize;
    let offset = pos_attr.offset as usize;
    let vertex_count = if stride > 0 {
        vertex_data.len() / stride
    } else {
        0
    };

    // Extract positions (assumed f32x3)
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let base = i * stride + offset;
        if base + 12 > vertex_data.len() {
            break;
        }
        let x = f32::from_le_bytes([
            vertex_data[base],
            vertex_data[base + 1],
            vertex_data[base + 2],
            vertex_data[base + 3],
        ]);
        let y = f32::from_le_bytes([
            vertex_data[base + 4],
            vertex_data[base + 5],
            vertex_data[base + 6],
            vertex_data[base + 7],
        ]);
        let z = f32::from_le_bytes([
            vertex_data[base + 8],
            vertex_data[base + 9],
            vertex_data[base + 10],
            vertex_data[base + 11],
        ]);
        vertices.push(glam::Vec3::new(x, y, z));
    }

    // Extract triangle indices
    let index_format = mesh.index_format()?;
    let indices_raw = mesh.index_data()?;
    let mut triangles = Vec::new();

    match index_format {
        IndexFormat::Uint16 => {
            let count = indices_raw.len() / 2;
            let mut idx = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * 2;
                idx.push(u16::from_le_bytes([indices_raw[base], indices_raw[base + 1]]) as u32);
            }
            for tri in idx.chunks_exact(3) {
                triangles.push([tri[0], tri[1], tri[2]]);
            }
        }
        IndexFormat::Uint32 => {
            let count = indices_raw.len() / 4;
            let mut idx = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * 4;
                idx.push(u32::from_le_bytes([
                    indices_raw[base],
                    indices_raw[base + 1],
                    indices_raw[base + 2],
                    indices_raw[base + 3],
                ]));
            }
            for tri in idx.chunks_exact(3) {
                triangles.push([tri[0], tri[1], tri[2]]);
            }
        }
    }

    Some((vertices, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_roundtrip() {
        let v = glam::Vec3::new(1.0, 2.0, 3.0);
        let na = vec3_to_na(v);
        let back = vec3_from_na(&na);
        assert!((v - back).length() < 1e-6);
    }

    #[test]
    fn quat_roundtrip() {
        let q = glam::Quat::from_rotation_y(1.0);
        let na = quat_to_na(q);
        let back = quat_from_na(&na);
        assert!((q - back).length() < 1e-5);
    }

    #[test]
    fn isometry3_roundtrip() {
        let pos = glam::Vec3::new(1.0, 2.0, 3.0);
        let rot = glam::Quat::from_rotation_z(0.5);
        let iso = isometry3_to_na(pos, rot);
        let (pos2, rot2) = isometry3_from_na(&iso);
        assert!((pos - pos2).length() < 1e-5);
        assert!((rot - rot2).length() < 1e-5);
    }

    #[test]
    fn isometry2_roundtrip() {
        let pos = glam::Vec2::new(1.0, 2.0);
        let angle = 0.7f32;
        let iso = isometry2_to_na(pos, angle);
        let (pos2, angle2) = isometry2_from_na(&iso);
        assert!((pos - pos2).length() < 1e-5);
        assert!((angle - angle2).abs() < 1e-5);
    }
}
