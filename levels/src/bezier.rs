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

/// Build the road patch between two nodes.
///
/// The end row's point order is flipped when connecting same-index corners
/// would cross the patch over itself (a node rotated ~180° should widen the
/// road's twist, not bowtie it). Headings are taken as-is: hairpins where a
/// node faces "back at" its partner are legitimate authoring.
pub fn patch_from_nodes(
    a_world: &Mat4,
    a_half_width: f32,
    tangent_a: f32,
    b_world: &Mat4,
    b_half_width: f32,
    tangent_b: f32,
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
    let row1 = row0.map(|p| p + fwd_a * tangent_a);
    let row2 = row3.map(|p| p - fwd_b * tangent_b);
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
    fn flipped_end_node_does_not_bowtie() {
        // Node b rotated 180° around Y: its local X points at −X. Without the
        // uncrossing fix, corner 0 would connect to world +X — a bowtie.
        let b = translation(0.0, 0.0, 10.0)
            * Mat4::from_axis_angle(
                &redlilium_core::math::nalgebra::Vector3::y_axis(),
                std::f32::consts::PI,
            );
        let patch = patch_from_nodes(&translation(0.0, 0.0, 0.0), 3.0, 4.0, &b, 3.0, 4.0);
        // Same-side corners stay on the same side of the X=0 plane.
        assert!(patch[0][0].x < 0.0 && patch[3][0].x < 0.0);
        assert!(patch[0][3].x > 0.0 && patch[3][3].x > 0.0);
    }
}
