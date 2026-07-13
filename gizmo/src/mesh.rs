//! Handle geometry: solid arrows and plane quads, tessellated per frame in
//! world space (position + color vertices, triangle list).

use bytemuck::{Pod, Zeroable};
use redlilium_core::math::Vec3;

use crate::math::{GizmoCamera, perpendicular_basis, screen_scale};
use crate::state::{Handle, TranslateGizmo, plane_axes};

/// One gizmo vertex: world position + RGBA color. Matches the debug-draw
/// shader's input layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Uniform block: column-major view-projection.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GizmoUniforms {
    pub view_proj: [[f32; 4]; 4],
}

const CONE_SEGMENTS: usize = 10;
const SHAFT_SIDES: usize = 6;

fn axis_color(handle: Handle) -> [f32; 4] {
    match handle {
        Handle::AxisX | Handle::PlaneYZ => [0.92, 0.22, 0.18, 1.0],
        Handle::AxisY | Handle::PlaneXZ => [0.32, 0.82, 0.22, 1.0],
        Handle::AxisZ | Handle::PlaneXY => [0.22, 0.42, 0.95, 1.0],
    }
}

/// Hover brightens, active drag goes hot yellow; planes are translucent.
fn handle_color(gizmo: &TranslateGizmo, handle: Handle) -> [f32; 4] {
    let mut c = axis_color(handle);
    if handle.is_plane() {
        c[3] = 0.35;
    }
    if gizmo.active() == Some(handle) {
        return [1.0, 0.85, 0.2, if handle.is_plane() { 0.55 } else { 1.0 }];
    }
    if gizmo.hovered() == Some(handle) {
        for ch in &mut c[..3] {
            *ch = (*ch + 0.45).min(1.0);
        }
        if handle.is_plane() {
            c[3] = 0.55;
        }
    }
    c
}

/// Tessellate the gizmo at its effective target for this frame's camera.
/// Returns an empty list when the gizmo is hidden.
pub fn build_vertices(gizmo: &TranslateGizmo, camera: &GizmoCamera) -> Vec<GizmoVertex> {
    let Some(target) = gizmo.effective_target() else {
        return Vec::new();
    };
    let s = screen_scale(camera, target, gizmo.config().size_factor);
    let mut out = Vec::with_capacity(1024);

    for handle in [Handle::AxisX, Handle::AxisY, Handle::AxisZ] {
        let color = handle_color(gizmo, handle);
        push_arrow(&mut out, target, handle.primary_dir(), s, color);
    }
    for handle in [Handle::PlaneXY, Handle::PlaneXZ, Handle::PlaneYZ] {
        let color = handle_color(gizmo, handle);
        let (u, v) = plane_axes(handle);
        let (min, max) = (gizmo.config().plane_min * s, gizmo.config().plane_max * s);
        push_quad(&mut out, target, u, v, min, max, color);
    }
    out
}

fn vertex(p: Vec3, color: [f32; 4]) -> GizmoVertex {
    GizmoVertex {
        position: [p.x, p.y, p.z],
        color,
    }
}

fn push_tri(out: &mut Vec<GizmoVertex>, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
    out.push(vertex(a, color));
    out.push(vertex(b, color));
    out.push(vertex(c, color));
}

/// Solid arrow: a prism shaft (0.2..0.82 of the length) + a cone tip
/// (0.82..1.05), all in gizmo-local units scaled by `s`.
fn push_arrow(out: &mut Vec<GizmoVertex>, origin: Vec3, axis: Vec3, s: f32, color: [f32; 4]) {
    let (u, v) = perpendicular_basis(axis);
    let shaft_r = 0.022 * s;
    let cone_r = 0.075 * s;
    let shaft_a = origin + axis * (0.2 * s);
    let shaft_b = origin + axis * (0.82 * s);
    let tip = origin + axis * (1.05 * s);

    // Shaft: SHAFT_SIDES quads around the axis.
    for i in 0..SHAFT_SIDES {
        let t0 = i as f32 / SHAFT_SIDES as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / SHAFT_SIDES as f32 * std::f32::consts::TAU;
        let r0 = (u * t0.cos() + v * t0.sin()) * shaft_r;
        let r1 = (u * t1.cos() + v * t1.sin()) * shaft_r;
        push_tri(out, shaft_a + r0, shaft_b + r0, shaft_b + r1, color);
        push_tri(out, shaft_a + r0, shaft_b + r1, shaft_a + r1, color);
    }
    // Cone: side fan + base disc.
    for i in 0..CONE_SEGMENTS {
        let t0 = i as f32 / CONE_SEGMENTS as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / CONE_SEGMENTS as f32 * std::f32::consts::TAU;
        let r0 = (u * t0.cos() + v * t0.sin()) * cone_r;
        let r1 = (u * t1.cos() + v * t1.sin()) * cone_r;
        push_tri(out, shaft_b + r0, tip, shaft_b + r1, color);
        push_tri(out, shaft_b + r1, shaft_b, shaft_b + r0, color);
    }
}

/// Double-sided plane quad spanning `[min, max]²` in the (u, v) basis.
fn push_quad(
    out: &mut Vec<GizmoVertex>,
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    min: f32,
    max: f32,
    color: [f32; 4],
) {
    let p00 = origin + u * min + v * min;
    let p10 = origin + u * max + v * min;
    let p11 = origin + u * max + v * max;
    let p01 = origin + u * min + v * max;
    push_tri(out, p00, p10, p11, color);
    push_tri(out, p00, p11, p01, color);
    // Back face so the quad is visible from either side.
    push_tri(out, p00, p11, p10, color);
    push_tri(out, p00, p01, p11, color);
}

/// Anchor dots: small screen-constant octahedra marking every draggable
/// control point of the focused entity. The active one (carrying the full
/// gizmo) renders hot, the rest neutral.
pub fn build_anchor_dots(
    dots: &[(Vec3, bool)],
    camera: &GizmoCamera,
    size_factor: f32,
) -> Vec<GizmoVertex> {
    let mut out = Vec::with_capacity(dots.len() * 24);
    for &(pos, active) in dots {
        let r = screen_scale(camera, pos, size_factor) * 0.06;
        let color = if active {
            [1.0, 0.85, 0.2, 1.0]
        } else {
            [0.95, 0.95, 0.95, 0.9]
        };
        push_octahedron(&mut out, pos, r, color);
    }
    out
}

fn push_octahedron(out: &mut Vec<GizmoVertex>, c: Vec3, r: f32, color: [f32; 4]) {
    let px = c + Vec3::new(r, 0.0, 0.0);
    let nx = c - Vec3::new(r, 0.0, 0.0);
    let py = c + Vec3::new(0.0, r, 0.0);
    let ny = c - Vec3::new(0.0, r, 0.0);
    let pz = c + Vec3::new(0.0, 0.0, r);
    let nz = c - Vec3::new(0.0, 0.0, r);
    for (a, b) in [(px, pz), (pz, nx), (nx, nz), (nz, px)] {
        push_tri(out, py, a, b, color);
        push_tri(out, ny, b, a, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GizmoConfig;
    use redlilium_core::math::{look_at_rh, perspective_rh};

    fn camera() -> GizmoCamera {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let view = look_at_rh(&eye, &Vec3::zeros(), &Vec3::new(0.0, 1.0, 0.0));
        let proj = perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        GizmoCamera {
            view_proj: proj * view,
            eye,
            viewport: (1600.0, 900.0),
        }
    }

    #[test]
    fn hidden_gizmo_builds_nothing() {
        let gizmo = TranslateGizmo::new(GizmoConfig::default());
        assert!(build_vertices(&gizmo, &camera()).is_empty());
    }

    #[test]
    fn visible_gizmo_builds_triangle_list() {
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::new(1.0, 2.0, 3.0)));
        let verts = build_vertices(&gizmo, &camera());
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 3, 0, "triangle list");
        // All vertices are near the target (within ~2 gizmo scales).
        let s = screen_scale(&camera(), Vec3::new(1.0, 2.0, 3.0), 0.15);
        for v in &verts {
            let p = Vec3::new(v.position[0], v.position[1], v.position[2]);
            assert!((p - Vec3::new(1.0, 2.0, 3.0)).norm() < 2.0 * s);
        }
    }
}
