//! Math type aliases and helper functions.
//!
//! Provides f32 rendering types (always available) and precision-aware
//! physics types (behind `physics-math` feature).

pub use nalgebra;

// ===== Rendering math (always f32) =====

/// 2D vector (f32).
pub type Vec2 = nalgebra::Vector2<f32>;

/// 3D vector (f32).
pub type Vec3 = nalgebra::Vector3<f32>;

/// 4D vector (f32).
pub type Vec4 = nalgebra::Vector4<f32>;

/// 4x4 matrix (f32).
pub type Mat4 = nalgebra::Matrix4<f32>;

/// Quaternion (f32). Stored as `[x, y, z, w]` in memory.
/// Use [`quat_from_xyzw`] or `Quaternion::new(w, x, y, z)` to construct.
pub type Quat = nalgebra::Quaternion<f32>;

// ===== Helper functions =====

/// Build a 4x4 TRS matrix from scale, rotation (quaternion), and translation.
pub fn mat4_from_scale_rotation_translation(
    scale: Vec3,
    rotation: Quat,
    translation: Vec3,
) -> Mat4 {
    let r = nalgebra::UnitQuaternion::new_unchecked(rotation);
    let m = r.to_rotation_matrix();
    let rm = m.matrix();
    #[rustfmt::skip]
    let result = Mat4::new(
        rm[(0, 0)] * scale.x, rm[(0, 1)] * scale.y, rm[(0, 2)] * scale.z, translation.x,
        rm[(1, 0)] * scale.x, rm[(1, 1)] * scale.y, rm[(1, 2)] * scale.z, translation.y,
        rm[(2, 0)] * scale.x, rm[(2, 1)] * scale.y, rm[(2, 2)] * scale.z, translation.z,
        0.0,                  0.0,                  0.0,                  1.0,
    );
    result
}

/// Build a right-handed perspective projection with depth range [0, 1] (wgpu/Vulkan convention).
pub fn perspective_rh(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
    let f = 1.0 / (yfov / 2.0).tan();
    let nf = 1.0 / (znear - zfar);
    #[rustfmt::skip]
    let result = Mat4::new(
        f / aspect, 0.0,  0.0,              0.0,
        0.0,        f,    0.0,              0.0,
        0.0,        0.0,  zfar * nf,        znear * zfar * nf,
        0.0,        0.0,  -1.0,             0.0,
    );
    result
}

/// Build a right-handed orthographic projection with depth range [0, 1] (wgpu/Vulkan convention).
pub fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    #[rustfmt::skip]
    let result = Mat4::new(
        2.0 / rml, 0.0,       0.0,         -(right + left) / rml,
        0.0,       2.0 / tmb, 0.0,         -(top + bottom) / tmb,
        0.0,       0.0,       -1.0 / fmn,  -near / fmn,
        0.0,       0.0,       0.0,          1.0,
    );
    result
}

/// Right-handed look-at view matrix.
pub fn look_at_rh(eye: &Vec3, target: &Vec3, up: &Vec3) -> Mat4 {
    let eye_point = nalgebra::Point3::from(*eye);
    let target_point = nalgebra::Point3::from(*target);
    nalgebra::Isometry3::look_at_rh(&eye_point, &target_point, up).to_homogeneous()
}

/// Build a translation-only 4x4 matrix.
pub fn mat4_from_translation(t: Vec3) -> Mat4 {
    Mat4::new_translation(&t)
}

/// Create a quaternion from x, y, z, w components.
pub fn quat_from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Quat {
    nalgebra::Quaternion::new(w, x, y, z)
}

/// Create a quaternion from a `[x, y, z, w]` array.
pub fn quat_from_array(a: [f32; 4]) -> Quat {
    nalgebra::Quaternion::new(a[3], a[0], a[1], a[2])
}

/// Convert a quaternion to a `[x, y, z, w]` array.
pub fn quat_to_array(q: Quat) -> [f32; 4] {
    [q.coords.x, q.coords.y, q.coords.z, q.coords.w]
}

/// Create a quaternion from rotation around the X axis.
pub fn quat_from_rotation_x(angle: f32) -> Quat {
    nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), angle).into_inner()
}

/// Create a quaternion from rotation around the Y axis.
pub fn quat_from_rotation_y(angle: f32) -> Quat {
    nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), angle).into_inner()
}

/// Create a quaternion from rotation around the Z axis.
pub fn quat_from_rotation_z(angle: f32) -> Quat {
    nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), angle).into_inner()
}

/// Rotate a vector by a quaternion.
pub fn quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3 {
    nalgebra::UnitQuaternion::new_unchecked(q) * v
}

/// Convert a 4x4 matrix to a column-major `[[f32; 4]; 4]` array.
pub fn mat4_to_cols_array_2d(m: &Mat4) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

/// Decompose a 4x4 matrix into (scale, rotation, translation).
pub fn to_scale_rotation_translation(m: &Mat4) -> (Vec3, Quat, Vec3) {
    let translation = Vec3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);
    let col0 = Vec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]);
    let col1 = Vec3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]);
    let col2 = Vec3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]);
    let sx = col0.norm();
    let sy = col1.norm();
    let sz = col2.norm();
    let scale = Vec3::new(sx, sy, sz);
    let rot_mat = nalgebra::Matrix3::from_columns(&[col0 / sx, col1 / sy, col2 / sz]);
    let rotation = nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rot_mat),
    )
    .into_inner();
    (scale, rotation, translation)
}

// ===== Physics math (precision-aware) =====

/// Physics scalar type. `f64` by default, `f32` with `physics-f32` feature.
#[cfg(all(feature = "physics-math", not(feature = "physics-f32")))]
pub type Real = f64;

/// Physics scalar type. `f32` with `physics-f32` feature.
#[cfg(all(feature = "physics-math", feature = "physics-f32"))]
pub type Real = f32;

/// 2D physics vector.
#[cfg(feature = "physics-math")]
pub type Vector2 = nalgebra::Vector2<Real>;

/// 3D physics vector.
#[cfg(feature = "physics-math")]
pub type Vector3 = nalgebra::Vector3<Real>;

/// 2D physics point.
#[cfg(feature = "physics-math")]
pub type Point2 = nalgebra::Point2<Real>;

/// 3D physics point.
#[cfg(feature = "physics-math")]
pub type Point3 = nalgebra::Point3<Real>;

/// 2D physics isometry (rotation + translation).
#[cfg(feature = "physics-math")]
pub type Isometry2 = nalgebra::Isometry2<Real>;

/// 3D physics isometry (rotation + translation).
#[cfg(feature = "physics-math")]
pub type Isometry3 = nalgebra::Isometry3<Real>;

/// 4x4 physics matrix.
#[cfg(feature = "physics-math")]
pub type Matrix4 = nalgebra::Matrix4<Real>;

/// 2D physics rotation (unit complex number).
#[cfg(feature = "physics-math")]
pub type UnitComplex = nalgebra::UnitComplex<Real>;

/// 3D physics rotation (unit quaternion).
#[cfg(feature = "physics-math")]
pub type UnitQuaternion = nalgebra::UnitQuaternion<Real>;

/// 2D physics translation.
#[cfg(feature = "physics-math")]
pub type Translation2 = nalgebra::Translation2<Real>;

/// 3D physics translation.
#[cfg(feature = "physics-math")]
pub type Translation3 = nalgebra::Translation3<Real>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_trs_matrix() {
        let m = mat4_from_scale_rotation_translation(
            Vec3::new(1.0, 1.0, 1.0),
            Quat::identity(),
            Vec3::zeros(),
        );
        assert!((m - Mat4::identity()).norm() < 1e-6);
    }

    #[test]
    fn translation_matrix() {
        let t = Vec3::new(1.0, 2.0, 3.0);
        let m = mat4_from_translation(t);
        assert_eq!(m[(0, 3)], 1.0);
        assert_eq!(m[(1, 3)], 2.0);
        assert_eq!(m[(2, 3)], 3.0);
    }

    #[test]
    fn quat_xyzw_roundtrip() {
        let q = quat_from_xyzw(0.1, 0.2, 0.3, 0.9);
        let arr = quat_to_array(q);
        assert!((arr[0] - 0.1).abs() < 1e-6);
        assert!((arr[1] - 0.2).abs() < 1e-6);
        assert!((arr[2] - 0.3).abs() < 1e-6);
        assert!((arr[3] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rotation_y_90() {
        let q = quat_from_rotation_y(FRAC_PI_2);
        let v = quat_rotate_vec3(q, Vec3::new(1.0, 0.0, 0.0));
        assert!((v.x - 0.0).abs() < 1e-5);
        assert!((v.z - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn decompose_trs_roundtrip() {
        let s = Vec3::new(2.0, 3.0, 4.0);
        let r = quat_from_rotation_y(1.0);
        let t = Vec3::new(5.0, 6.0, 7.0);
        let m = mat4_from_scale_rotation_translation(s, r, t);
        let (s2, r2, t2) = to_scale_rotation_translation(&m);
        assert!((s - s2).norm() < 1e-5);
        assert!((t - t2).norm() < 1e-5);
        // Compare rotations by rotating a test vector
        let test = Vec3::new(1.0, 0.0, 0.0);
        assert!((quat_rotate_vec3(r, test) - quat_rotate_vec3(r2, test)).norm() < 1e-5);
    }

    #[test]
    fn cols_array_2d_identity() {
        let m = Mat4::identity();
        let cols = mat4_to_cols_array_2d(&m);
        assert_eq!(cols[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cols[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cols[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(cols[3], [0.0, 0.0, 0.0, 1.0]);
    }
}
