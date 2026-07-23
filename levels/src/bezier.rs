//! Bicubic Bézier road patches derived from node cross-sections.
//!
//! A road between two [`RoadNode`](crate::RoadNode)s is a 4×4 Bézier patch:
//! rows 0 and 3 are the two cross-sections (4 points uniformly along each
//! node's local-X segment), rows 1 and 2 extend from them along the nodes'
//! +Z headings by the segment's tangent lengths. `u` runs along the road
//! (row 0 → row 3), `v` across it.

use redlilium_core::math::{Mat4, Vec3, Vec4};

/// Control grid: `points[u_row][v_column]`.
pub type Patch = [[Vec3; 4]; 4];

/// The 4 boundary control points of a node's cross-section, in world space:
/// local X at fractions −1, −1/3, 1/3, 1 of `half_width` through `world`.
pub fn cross_section(world: &Mat4, half_width: f32) -> [Vec3; 4] {
    [-1.0f32, -1.0 / 3.0, 1.0 / 3.0, 1.0].map(|f| {
        let p = world * Vec4::new(f * half_width, 0.0, 0.0, 1.0);
        Vec3::new(p.x, p.y, p.z)
    })
}

/// A node's world-space +Z heading (unit length).
pub fn heading(world: &Mat4) -> Vec3 {
    let d = world * Vec4::new(0.0, 0.0, 1.0, 0.0);
    Vec3::new(d.x, d.y, d.z).normalize()
}

/// Rotation whose local **X runs along `x_axis` in full 3D** — a chord
/// between two contour points keeps its tilt, it is never projected onto
/// a ground plane — and whose local +Z points as close to `toward_z` as
/// orthogonality allows. `None` when either input degenerates or they are
/// (anti)parallel.
pub fn rotation_with_x_along(x_axis: Vec3, toward_z: Vec3) -> Option<redlilium_core::math::Quat> {
    use redlilium_core::math::nalgebra;
    if x_axis.norm() < 1e-6 {
        return None;
    }
    let x = x_axis.normalize();
    let z0 = toward_z - x * x.dot(&toward_z);
    if z0.norm() < 1e-6 {
        return None;
    }
    let z = z0.normalize();
    let y = z.cross(&x);
    let m = nalgebra::Matrix3::from_columns(&[x, y, z]);
    let rotation = nalgebra::Rotation3::from_matrix_unchecked(m);
    Some(nalgebra::UnitQuaternion::from_rotation_matrix(&rotation).into_inner())
}

/// Build the road patch between two nodes.
///
/// The end row's point order is flipped when connecting same-index corners
/// would cross the patch over itself (a node rotated ~180° should widen the
/// road's twist, not bowtie it). Headings are taken as-is: hairpins where a
/// node faces "back at" its partner are legitimate authoring.
/// `b_from_front`: the patch meets `b` on its +Z side instead of the
/// default −Z — used when `b` is a socket whose +Z faces the road network.
pub fn patch_from_nodes(
    a_world: &Mat4,
    a_half_width: f32,
    tangent_a: f32,
    b_world: &Mat4,
    b_half_width: f32,
    tangent_b: f32,
    b_from_front: bool,
) -> Patch {
    let row0 = cross_section(a_world, a_half_width);
    let mut row3 = cross_section(b_world, b_half_width);

    let straight = (row3[0] - row0[0]).norm() + (row3[3] - row0[3]).norm();
    let crossed = (row3[3] - row0[0]).norm() + (row3[0] - row0[3]).norm();
    if crossed < straight {
        row3.reverse();
    }

    let fwd_a = heading(a_world);
    let fwd_b = heading(b_world);
    let b_sign = if b_from_front { 1.0 } else { -1.0 };
    let row1 = row0.map(|p| p + fwd_a * tangent_a);
    let row2 = row3.map(|p| p + fwd_b * (b_sign * tangent_b));
    [row0, row1, row2, row3]
}

/// Cubic Bernstein basis at `t`.
fn bernstein(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t]
}

/// Evaluate the patch surface at `(u, v)`, both in `[0, 1]`.
pub fn eval(patch: &Patch, u: f32, v: f32) -> Vec3 {
    let bu = bernstein(u);
    let bv = bernstein(v);
    let mut p = Vec3::zeros();
    for (i, bi) in bu.iter().enumerate() {
        for (j, bj) in bv.iter().enumerate() {
            p += patch[i][j] * (bi * bj);
        }
    }
    p
}

/// Derivative of the cubic Bernstein basis at `t`.
fn bernstein_dt(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [
        -3.0 * s * s,
        3.0 * s * s - 6.0 * t * s,
        6.0 * t * s - 3.0 * t * t,
        3.0 * t * t,
    ]
}

/// Surface derivative across the road, `∂P/∂v` at `(u, v)`. This is the
/// direction the surface leaves its side edge in — edge attachments use it
/// so a driveway continues the road's cross-slope (G1 off the edge).
pub fn eval_dv(patch: &Patch, u: f32, v: f32) -> Vec3 {
    let bu = bernstein(u);
    let bv = bernstein_dt(v);
    let mut d = Vec3::zeros();
    for (i, bi) in bu.iter().enumerate() {
        for (j, bj) in bv.iter().enumerate() {
            d += patch[i][j] * (bi * bj);
        }
    }
    d
}

/// Four points uniformly spanning `[u_min, u_max]` on a side edge of the
/// patch (`v = 0` or `v = 1`). Sampled directly from the surface — an
/// attachment's boundary row lies exactly on the road's edge curve.
pub fn sample_edge_row(patch: &Patch, v_edge: f32, u_min: f32, u_max: f32) -> [Vec3; 4] {
    [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 1.0].map(|f| eval(patch, u_min + (u_max - u_min) * f, v_edge))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f32, y: f32, z: f32) -> Mat4 {
        Mat4::new_translation(&Vec3::new(x, y, z))
    }

    #[test]
    fn corners_interpolate_control_grid() {
        let patch = patch_from_nodes(
            &translation(0.0, 0.0, 0.0),
            3.0,
            4.0,
            &translation(0.0, 0.0, 10.0),
            3.0,
            4.0,
            false,
        );
        assert!((eval(&patch, 0.0, 0.0) - patch[0][0]).norm() < 1e-5);
        assert!((eval(&patch, 0.0, 1.0) - patch[0][3]).norm() < 1e-5);
        assert!((eval(&patch, 1.0, 0.0) - patch[3][0]).norm() < 1e-5);
        assert!((eval(&patch, 1.0, 1.0) - patch[3][3]).norm() < 1e-5);
    }

    #[test]
    fn straight_road_stays_planar_and_spans_nodes() {
        let patch = patch_from_nodes(
            &translation(0.0, 0.0, 0.0),
            3.0,
            4.0,
            &translation(0.0, 0.0, 12.0),
            3.0,
            4.0,
            false,
        );
        let mid = eval(&patch, 0.5, 0.5);
        assert!(mid.y.abs() < 1e-5);
        assert!(mid.x.abs() < 1e-5);
        assert!((mid.z - 6.0).abs() < 1e-4);
        // Edge rows sit on the node segments.
        assert!((patch[0][0] - Vec3::new(-3.0, 0.0, 0.0)).norm() < 1e-5);
        assert!((patch[3][3] - Vec3::new(3.0, 0.0, 12.0)).norm() < 1e-5);
    }

    #[test]
    fn dv_matches_finite_difference() {
        let patch = patch_from_nodes(
            &translation(0.0, 0.0, 0.0),
            3.0,
            4.0,
            &(translation(4.0, 1.0, 12.0)
                * Mat4::from_axis_angle(&redlilium_core::math::nalgebra::Vector3::y_axis(), 0.7)),
            2.0,
            5.0,
            false,
        );
        let (u, v) = (0.37, 0.62);
        let h = 1e-3;
        let numeric = (eval(&patch, u, v + h) - eval(&patch, u, v - h)) / (2.0 * h);
        let analytic = eval_dv(&patch, u, v);
        assert!((numeric - analytic).norm() < 1e-2, "dv mismatch");
    }

    #[test]
    fn edge_row_lies_on_the_surface_edge() {
        let patch = patch_from_nodes(
            &translation(0.0, 0.0, 0.0),
            3.0,
            4.0,
            &translation(0.0, 0.0, 12.0),
            3.0,
            4.0,
            false,
        );
        let row = sample_edge_row(&patch, 1.0, 0.25, 0.75);
        for (i, p) in row.iter().enumerate() {
            let u = 0.25 + 0.5 * (i as f32 / 3.0);
            assert!((p - eval(&patch, u, 1.0)).norm() < 1e-5);
        }
    }

    #[test]
    fn b_from_front_puts_the_end_tangent_on_plus_z() {
        // b's +Z faces BACK at a (a socket looking at the road network).
        let b = translation(0.0, 0.0, 10.0)
            * Mat4::from_axis_angle(
                &redlilium_core::math::nalgebra::Vector3::y_axis(),
                std::f32::consts::PI,
            );
        let default_side =
            patch_from_nodes(&translation(0.0, 0.0, 0.0), 3.0, 4.0, &b, 3.0, 4.0, false);
        let front_side =
            patch_from_nodes(&translation(0.0, 0.0, 0.0), 3.0, 4.0, &b, 3.0, 4.0, true);
        // Default: control row behind b (z > 10, the far side — an S-kink
        // for a socket). Front: control row between a and b (z < 10).
        assert!(default_side[2].iter().all(|p| p.z > 10.0));
        assert!(front_side[2].iter().all(|p| p.z < 10.0));
    }

    #[test]
    fn flipped_end_node_does_not_bowtie() {
        // Node b rotated 180° around Y: its local X points at −X. Without the
        // uncrossing fix, corner 0 would connect to world +X — a bowtie.
        let b = translation(0.0, 0.0, 10.0)
            * Mat4::from_axis_angle(
                &redlilium_core::math::nalgebra::Vector3::y_axis(),
                std::f32::consts::PI,
            );
        let patch = patch_from_nodes(&translation(0.0, 0.0, 0.0), 3.0, 4.0, &b, 3.0, 4.0, false);
        // Same-side corners stay on the same side of the X=0 plane.
        assert!(patch[0][0].x < 0.0 && patch[3][0].x < 0.0);
        assert!(patch[0][3].x > 0.0 && patch[3][3].x > 0.0);
    }
}
