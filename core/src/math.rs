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
    // Normalize: a non-unit quaternion (e.g. drifted from accumulated edits or
    // loaded data) would otherwise embed its squared norm as extra scale and
    // silently produce a wrong matrix. `new_normalize` is a no-op for unit input.
    let r = nalgebra::UnitQuaternion::new_normalize(rotation);
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

/// Depth mapping convention for projection matrices and depth attachments
/// (ADR-038, #125).
///
/// The engine default is [`ReversedZ`](Self::ReversedZ): near plane → depth 1,
/// far plane → depth 0, clear to 0.0, compare `GreaterEqual`. With a float
/// depth buffer this distributes precision near-uniformly across the view
/// range instead of cramming it at the near plane. [`Classic`](Self::Classic)
/// (near → 0, far → 1, clear 1.0, `LessEqual`) is the opt-out.
///
/// Everything depth-related derives from this enum in one place: the
/// projection matrix flavor ([`perspective`](Self::perspective) /
/// [`orthographic`](Self::orthographic)) and the default clear depth
/// ([`clear_depth`](Self::clear_depth)); the graphics crate maps it to the
/// default depth compare op. Do not scatter `if reversed` logic — route it
/// through these methods.
///
/// # One convention per depth target
///
/// The convention must stay **consistent within one render target**: every
/// camera/pass writing or testing against a given depth buffer must use the
/// same convention (projection + clear + compare as a unit). Mixing
/// conventions on one depth buffer is a logic error the engine does not
/// detect — it renders garbage, not a validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthConvention {
    /// Near → 1, far → 0, clear 0.0, compare `GreaterEqual`. Engine default.
    #[default]
    ReversedZ,
    /// Near → 0, far → 1, clear 1.0, compare `LessEqual`. Opt-out.
    Classic,
}

impl DepthConvention {
    /// The depth value a fresh depth attachment is cleared to under this
    /// convention: 0.0 for reversed-Z ("everything is behind"), 1.0 classic.
    pub fn clear_depth(self) -> f32 {
        match self {
            Self::ReversedZ => 0.0,
            Self::Classic => 1.0,
        }
    }

    /// The default depth comparison for this convention: `GreaterEqual` for
    /// reversed-Z (closer fragments have *larger* depth), `LessEqual` classic.
    pub fn compare(self) -> crate::sampler::CompareFunction {
        match self {
            Self::ReversedZ => crate::sampler::CompareFunction::GreaterEqual,
            Self::Classic => crate::sampler::CompareFunction::LessEqual,
        }
    }

    /// Right-handed perspective projection ([0, 1] depth) under this convention.
    pub fn perspective(self, yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
        match self {
            Self::ReversedZ => perspective_rh_reversed(yfov, aspect, znear, zfar),
            Self::Classic => perspective_rh(yfov, aspect, znear, zfar),
        }
    }

    /// Right-handed orthographic projection ([0, 1] depth) under this convention.
    #[allow(clippy::too_many_arguments)]
    pub fn orthographic(
        self,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Mat4 {
        match self {
            Self::ReversedZ => orthographic_rh_reversed(left, right, bottom, top, near, far),
            Self::Classic => orthographic_rh(left, right, bottom, top, near, far),
        }
    }
}

/// Build a right-handed perspective projection with depth range [0, 1] (wgpu/Vulkan convention).
///
/// Classic depth mapping (near → 0, far → 1). The engine default is
/// reversed-Z — prefer [`DepthConvention::perspective`] or
/// [`perspective_rh_reversed`] unless the target explicitly opts out.
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

/// Build a right-handed **reversed-Z** perspective projection with depth range
/// [0, 1]: near plane → depth 1, far plane → depth 0 (ADR-038, #125).
///
/// Same as [`perspective_rh`] with the z-row remapped so depth decreases with
/// distance. Pair with clear depth 0.0 and a `GreaterEqual` depth compare.
pub fn perspective_rh_reversed(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
    let f = 1.0 / (yfov / 2.0).tan();
    // Classic z-row with znear and zfar swapped: depth(znear) = 1, depth(zfar) = 0.
    let fn_ = 1.0 / (zfar - znear);
    #[rustfmt::skip]
    let result = Mat4::new(
        f / aspect, 0.0,  0.0,              0.0,
        0.0,        f,    0.0,              0.0,
        0.0,        0.0,  znear * fn_,      znear * zfar * fn_,
        0.0,        0.0,  -1.0,             0.0,
    );
    result
}

/// Build a right-handed orthographic projection with depth range [0, 1] (wgpu/Vulkan convention).
///
/// Classic depth mapping (near → 0, far → 1). The engine default is
/// reversed-Z — prefer [`DepthConvention::orthographic`] or
/// [`orthographic_rh_reversed`] unless the target explicitly opts out.
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

/// Build a right-handed **reversed-Z** orthographic projection with depth range
/// [0, 1]: near plane → depth 1, far plane → depth 0 (ADR-038, #125).
///
/// Same as [`orthographic_rh`] with the z-row remapped so depth decreases with
/// distance. Pair with clear depth 0.0 and a `GreaterEqual` depth compare.
pub fn orthographic_rh_reversed(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    #[rustfmt::skip]
    let result = Mat4::new(
        2.0 / rml, 0.0,       0.0,         -(right + left) / rml,
        0.0,       2.0 / tmb, 0.0,         -(top + bottom) / tmb,
        0.0,       0.0,       1.0 / fmn,    far / fmn,
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
    // Normalize so a non-unit quaternion does not also scale the vector.
    nalgebra::UnitQuaternion::new_normalize(q) * v
}

/// Rotation whose forward axis (**-Z**, the engine's transform convention)
/// points along `dir`, with roll fixed by the world up (+Y). Degenerate for
/// near-vertical directions, where +Z is used as the roll reference instead.
///
/// The canonical way to aim a directional light or camera at a direction.
pub fn quat_looking_along(dir: Vec3) -> Quat {
    let dir = dir.normalize();
    let up = if dir.y.abs() > 0.999 {
        Vec3::z()
    } else {
        Vec3::y()
    };
    // face_towards points the local +Z at its target; forward is -Z.
    nalgebra::UnitQuaternion::face_towards(&-dir, &up).into_inner()
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

    /// NDC depth of a view-space point at distance `dist` in front of the camera.
    fn projected_depth(proj: &Mat4, dist: f32) -> f32 {
        let clip = proj * Vec4::new(0.0, 0.0, -dist, 1.0);
        clip.z / clip.w
    }

    #[test]
    fn perspective_classic_depth_range() {
        let proj = perspective_rh(1.0, 1.0, 0.1, 100.0);
        assert!(projected_depth(&proj, 0.1).abs() < 1e-5);
        assert!((projected_depth(&proj, 100.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn perspective_reversed_depth_range() {
        let proj = perspective_rh_reversed(1.0, 1.0, 0.1, 100.0);
        assert!((projected_depth(&proj, 0.1) - 1.0).abs() < 1e-5);
        assert!(projected_depth(&proj, 100.0).abs() < 1e-5);
    }

    #[test]
    fn perspective_reversed_matches_classic_xy() {
        // Reversed-Z only remaps depth; x/y projection is identical.
        let classic = perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0);
        let reversed = perspective_rh_reversed(1.0, 16.0 / 9.0, 0.1, 100.0);
        let p = Vec4::new(1.5, -2.0, -10.0, 1.0);
        let c = classic * p;
        let r = reversed * p;
        assert!((c.x - r.x).abs() < 1e-6);
        assert!((c.y - r.y).abs() < 1e-6);
        assert!((c.w - r.w).abs() < 1e-6);
    }

    #[test]
    fn orthographic_classic_depth_range() {
        let proj = orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        assert!(projected_depth(&proj, 0.1).abs() < 1e-5);
        assert!((projected_depth(&proj, 100.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthographic_reversed_depth_range() {
        let proj = orthographic_rh_reversed(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        assert!((projected_depth(&proj, 0.1) - 1.0).abs() < 1e-5);
        assert!(projected_depth(&proj, 100.0).abs() < 1e-5);
    }

    #[test]
    fn looking_along_points_forward_at_dir() {
        for dir in [
            Vec3::new(0.75, 0.40, 0.75),
            Vec3::new(-0.5, -0.3, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0), // near-vertical fallback branch
        ] {
            let q = quat_looking_along(dir);
            assert!((q.norm() - 1.0).abs() < 1e-6, "unit quaternion for {dir}");
            let forward = quat_rotate_vec3(q, Vec3::new(0.0, 0.0, -1.0));
            assert!(
                (forward - dir.normalize()).norm() < 1e-5,
                "forward {forward} should match {dir}"
            );
        }
    }

    #[test]
    fn convention_dispatch() {
        assert_eq!(DepthConvention::default(), DepthConvention::ReversedZ);
        assert_eq!(DepthConvention::ReversedZ.clear_depth(), 0.0);
        assert_eq!(DepthConvention::Classic.clear_depth(), 1.0);
        assert_eq!(
            DepthConvention::ReversedZ.compare(),
            crate::sampler::CompareFunction::GreaterEqual
        );
        assert_eq!(
            DepthConvention::Classic.compare(),
            crate::sampler::CompareFunction::LessEqual
        );
        assert_eq!(
            DepthConvention::ReversedZ.perspective(1.0, 1.0, 0.1, 100.0),
            perspective_rh_reversed(1.0, 1.0, 0.1, 100.0)
        );
        assert_eq!(
            DepthConvention::Classic.perspective(1.0, 1.0, 0.1, 100.0),
            perspective_rh(1.0, 1.0, 0.1, 100.0)
        );
        assert_eq!(
            DepthConvention::ReversedZ.orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
            orthographic_rh_reversed(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)
        );
        assert_eq!(
            DepthConvention::Classic.orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
            orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)
        );
    }

    #[test]
    fn reversed_precision_beats_classic_far_field() {
        // The rationale for reversed-Z (ADR-038): with a float depth buffer and
        // a large far/near ratio, classic depth collapses distant points onto
        // indistinguishable values while reversed keeps them apart. Two points
        // 1 unit apart near the far plane of a 0.01..10000 frustum:
        let classic = perspective_rh(1.0, 1.0, 0.01, 10000.0);
        let reversed = perspective_rh_reversed(1.0, 1.0, 0.01, 10000.0);
        let (a, b) = (9000.0, 9001.0);
        let sep_classic =
            (projected_depth(&classic, a) as f64 - projected_depth(&classic, b) as f64).abs();
        let sep_reversed =
            (projected_depth(&reversed, a) as f64 - projected_depth(&reversed, b) as f64).abs();
        // f32 resolution around classic depth (~1.0) is ~1.2e-7; around the
        // reversed value (~1.1e-6) it is ~1e-13 (subnormal-adjacent exponent).
        // The classic separation must sit below its representable step while
        // the reversed separation sits far above its own.
        let f32_step_classic = 2.0_f64.powi(-23); // ulp near 1.0
        assert!(
            sep_classic < f32_step_classic * 4.0,
            "classic separation {sep_classic} unexpectedly large"
        );
        assert!(
            sep_reversed > sep_classic * 100.0,
            "reversed separation {sep_reversed} not better than classic {sep_classic}"
        );
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
