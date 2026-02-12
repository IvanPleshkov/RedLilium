//! Conversion helpers between f32 rendering types and physics-precision types.
//!
//! Also includes utilities for extracting physics collider data from [`CpuMesh`].

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
    redlilium_core::math::UnitQuaternion::new_unchecked(quat)
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

/// Extracts triangle mesh data (positions, triangle indices) from a [`CpuMesh`]
/// suitable for creating a trimesh collider.
///
/// Returns `None` if the mesh has no position data or no index buffer.
pub fn extract_trimesh_data(
    mesh: &redlilium_core::mesh::CpuMesh,
) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
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
        vertices.push(Vec3::new(x, y, z));
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
