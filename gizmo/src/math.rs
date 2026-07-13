//! Ray construction, ray–handle intersection, and drag-parameter math.
//!
//! Everything here is pure and unit-tested headlessly — the interaction
//! model's correctness lives in these functions, not in the renderer.

use redlilium_core::math::{Mat4, Vec3, Vec4};

/// A world-space picking ray. `dir` is normalized.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

/// The camera parameters the gizmo needs each frame.
#[derive(Debug, Clone, Copy)]
pub struct GizmoCamera {
    /// Combined view-projection matrix (column-major, as produced by
    /// `perspective_rh * look_at_rh`).
    pub view_proj: Mat4,
    /// The camera's world position (for screen-constant scaling).
    pub eye: Vec3,
    /// Viewport size in pixels.
    pub viewport: (f32, f32),
}

impl GizmoCamera {
    /// The picking ray through a cursor position (pixels, origin top-left).
    ///
    /// Returns `None` if the view-projection matrix is singular.
    pub fn ray_from_screen(&self, cursor: (f32, f32)) -> Option<Ray> {
        let inv = self.view_proj.try_inverse()?;
        let ndc_x = 2.0 * cursor.0 / self.viewport.0 - 1.0;
        let ndc_y = 1.0 - 2.0 * cursor.1 / self.viewport.1;

        let unproject = |z: f32| -> Vec3 {
            let p = inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
            Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
        };
        // wgpu/Vulkan clip space: near plane at z = 0.
        let near = unproject(0.0);
        let far = unproject(1.0);
        let dir = far - near;
        let len = dir.norm();
        if !len.is_finite() || len <= f32::EPSILON {
            return None;
        }
        Some(Ray {
            origin: near,
            dir: dir / len,
        })
    }

    /// Projects a world point to screen pixels. Returns `None` behind the
    /// camera. Used by tests to steer the synthetic cursor.
    pub fn project(&self, world: Vec3) -> Option<(f32, f32)> {
        let clip = self.view_proj * Vec4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        Some((
            (ndc_x + 1.0) * 0.5 * self.viewport.0,
            (1.0 - ndc_y) * 0.5 * self.viewport.1,
        ))
    }
}

/// Closest-point parameters between a ray and an infinite line
/// (Ericson, *Real-Time Collision Detection* §5.1.8, both directions
/// normalized). Returns `(t_ray, t_line)` — `p_ray = ray.origin + t_ray *
/// ray.dir`, `p_line = line_origin + t_line * line_dir` — or `None` when the
/// directions are near-parallel (the drag would be numerically unstable).
pub fn closest_ray_line_params(ray: &Ray, line_origin: Vec3, line_dir: Vec3) -> Option<(f32, f32)> {
    let b = ray.dir.dot(&line_dir);
    let denom = 1.0 - b * b;
    // |b| ~ 1 → ray nearly parallel to the axis: a pixel of cursor motion
    // maps to a huge axis delta. Refuse; the caller disables the handle.
    if denom < 1e-4 {
        return None;
    }
    let d = ray.origin - line_origin;
    let c = ray.dir.dot(&d);
    let f = line_dir.dot(&d);
    let t_ray = (b * f - c) / denom;
    let t_line = (f - b * c) / denom;
    Some((t_ray, t_line))
}

/// Ray–plane intersection point. `None` when the ray is near-parallel to the
/// plane (grazing drags are unstable) or the hit is behind the ray origin.
pub fn ray_plane_intersect(ray: &Ray, plane_point: Vec3, plane_normal: Vec3) -> Option<Vec3> {
    let denom = ray.dir.dot(&plane_normal);
    if denom.abs() < 5e-2 {
        return None;
    }
    let t = (plane_point - ray.origin).dot(&plane_normal) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}

/// Ray-vs-capsule test around the axis segment `[origin + t_min*axis,
/// origin + t_max*axis]`. Returns the ray parameter of the closest approach
/// when the distance is within `radius`.
pub fn ray_capsule_hit(
    ray: &Ray,
    origin: Vec3,
    axis: Vec3,
    t_min: f32,
    t_max: f32,
    radius: f32,
) -> Option<f32> {
    let (_, t_axis) = closest_ray_line_params(ray, origin, axis)?;
    let t_axis = t_axis.clamp(t_min, t_max);
    // Closest ray point to the clamped segment point (never behind the origin).
    let seg_point = origin + axis * t_axis;
    let t_ray = (seg_point - ray.origin).dot(&ray.dir).max(0.0);
    let ray_point = ray.origin + ray.dir * t_ray;
    if (ray_point - seg_point).norm() <= radius {
        Some(t_ray)
    } else {
        None
    }
}

/// Ray-vs-quad test. The quad spans `[min, max]` in the (u, v) basis around
/// `origin` (i.e. corners at `origin + u*min + v*min` … `origin + u*max +
/// v*max`). Returns the ray parameter of the hit.
pub fn ray_quad_hit(ray: &Ray, origin: Vec3, u: Vec3, v: Vec3, min: f32, max: f32) -> Option<f32> {
    let normal = u.cross(&v);
    let denom = ray.dir.dot(&normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray.origin).dot(&normal) / denom;
    if t < 0.0 {
        return None;
    }
    let p = ray.origin + ray.dir * t - origin;
    let pu = p.dot(&u);
    let pv = p.dot(&v);
    if (min..=max).contains(&pu) && (min..=max).contains(&pv) {
        Some(t)
    } else {
        None
    }
}

/// Screen-constant gizmo scale: proportional to the eye–target distance so
/// the gizmo occupies roughly the same screen area at any zoom.
pub fn screen_scale(camera: &GizmoCamera, target: Vec3, size_factor: f32) -> f32 {
    ((camera.eye - target).norm() * size_factor).max(1e-4)
}

/// An orthonormal basis perpendicular to `axis` (assumed normalized).
pub fn perpendicular_basis(axis: Vec3) -> (Vec3, Vec3) {
    let helper = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(&helper).normalize();
    let v = axis.cross(&u);
    (u, v)
}

/// Ray-vs-ring test: a torus-like band around `center` in the plane with
/// `normal`, at `radius` with half-width `band`. Returns the ray parameter
/// of the plane hit when it lands inside the band.
pub fn ray_ring_hit(ray: &Ray, center: Vec3, normal: Vec3, radius: f32, band: f32) -> Option<f32> {
    let denom = ray.dir.dot(&normal);
    if denom.abs() < 5e-2 {
        // Edge-on ring: picking (and dragging) would be unstable.
        return None;
    }
    let t = (center - ray.origin).dot(&normal) / denom;
    if t < 0.0 {
        return None;
    }
    let p = ray.origin + ray.dir * t;
    let r = (p - center).norm();
    ((radius - band)..=(radius + band))
        .contains(&r)
        .then_some(t)
}

/// The direction from `center` to the ray's intersection with the plane
/// (`normal` through `center`), normalized. The rotate drag's anchor and
/// per-frame sample. `None` when the ray misses or grazes the plane, or the
/// hit is (numerically) at the center.
pub fn ring_direction(ray: &Ray, center: Vec3, normal: Vec3) -> Option<Vec3> {
    let p = ray_plane_intersect(ray, center, normal)?;
    let v = p - center;
    let len = v.norm();
    (len > 1e-6).then(|| v / len)
}

/// Signed angle (radians) rotating `from` to `to` around `normal`
/// (right-handed). Both inputs unit-length and in the plane of `normal`.
pub fn signed_angle(from: Vec3, to: Vec3, normal: Vec3) -> f32 {
    let cos = from.dot(&to).clamp(-1.0, 1.0);
    let sin = from.cross(&to).dot(&normal);
    sin.atan2(cos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::{look_at_rh, perspective_rh};

    fn test_camera() -> GizmoCamera {
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
    fn screen_center_ray_points_forward() {
        let cam = test_camera();
        let ray = cam.ray_from_screen((800.0, 450.0)).unwrap();
        // Looking from +Z towards the origin: the center ray points -Z.
        assert!(ray.dir.z < -0.99, "dir {:?}", ray.dir);
    }

    #[test]
    fn project_unproject_roundtrip() {
        let cam = test_camera();
        let world = Vec3::new(0.7, -0.3, 0.2);
        let screen = cam.project(world).unwrap();
        let ray = cam.ray_from_screen(screen).unwrap();
        // The ray must pass through the world point.
        let t = (world - ray.origin).dot(&ray.dir);
        let closest = ray.origin + ray.dir * t;
        assert!(
            (closest - world).norm() < 1e-3,
            "off by {}",
            (closest - world).norm()
        );
    }

    #[test]
    fn closest_params_hit_known_point() {
        // Ray from above pointing straight down at x=2 crosses the X axis at t_line=2.
        let ray = Ray {
            origin: Vec3::new(2.0, 5.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        let (t_ray, t_line) =
            closest_ray_line_params(&ray, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        assert!((t_ray - 5.0).abs() < 1e-5);
        assert!((t_line - 2.0).abs() < 1e-5);
    }

    #[test]
    fn parallel_ray_rejected() {
        let ray = Ray {
            origin: Vec3::new(0.0, 1.0, 0.0),
            dir: Vec3::new(1.0, 0.0, 0.0),
        };
        assert!(closest_ray_line_params(&ray, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn plane_intersect_basic() {
        let ray = Ray {
            origin: Vec3::new(1.0, 2.0, 3.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        let hit = ray_plane_intersect(&ray, Vec3::zeros(), Vec3::new(0.0, 0.0, 1.0)).unwrap();
        assert!((hit - Vec3::new(1.0, 2.0, 0.0)).norm() < 1e-5);
    }

    #[test]
    fn grazing_plane_rejected() {
        let ray = Ray {
            origin: Vec3::new(0.0, 1.0, 0.0),
            dir: Vec3::new(1.0, 0.0, 0.0),
        };
        assert!(ray_plane_intersect(&ray, Vec3::zeros(), Vec3::new(0.0, 1.0, 0.0)).is_none());
    }

    #[test]
    fn capsule_hit_and_miss() {
        let ray = Ray {
            origin: Vec3::new(0.5, 3.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        // Straight down over the X axis segment [0.2, 1.0]: hit.
        assert!(
            ray_capsule_hit(&ray, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 0.2, 1.0, 0.1).is_some()
        );
        // Same ray but the segment ends before x=0.5: miss.
        assert!(
            ray_capsule_hit(&ray, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 0.0, 0.3, 0.1).is_none()
        );
    }

    #[test]
    fn quad_hit_and_miss() {
        let ray = Ray {
            origin: Vec3::new(0.4, 0.4, 3.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        let u = Vec3::new(1.0, 0.0, 0.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        assert!(ray_quad_hit(&ray, Vec3::zeros(), u, v, 0.3, 0.6).is_some());
        assert!(ray_quad_hit(&ray, Vec3::zeros(), u, v, 0.5, 0.9).is_none());
    }

    #[test]
    fn screen_scale_proportional_to_distance() {
        let cam = test_camera();
        let near = screen_scale(&cam, Vec3::new(0.0, 0.0, 4.0), 0.15); // dist 1
        let far = screen_scale(&cam, Vec3::new(0.0, 0.0, -5.0), 0.15); // dist 10
        assert!((far / near - 10.0).abs() < 1e-3);
    }

    #[test]
    fn ring_hit_band_and_miss() {
        let ray = |x: f32| Ray {
            origin: Vec3::new(x, 0.0, 5.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        let n = Vec3::new(0.0, 0.0, 1.0);
        // Ring radius 1.0, band 0.1 in the XY plane.
        assert!(ray_ring_hit(&ray(1.05), Vec3::zeros(), n, 1.0, 0.1).is_some());
        assert!(ray_ring_hit(&ray(0.5), Vec3::zeros(), n, 1.0, 0.1).is_none());
        assert!(ray_ring_hit(&ray(1.3), Vec3::zeros(), n, 1.0, 0.1).is_none());
    }

    #[test]
    fn signed_angle_quarter_turns() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        assert!((signed_angle(x, y, z) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((signed_angle(y, x, z) + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((signed_angle(x, -x, z).abs() - std::f32::consts::PI).abs() < 1e-5);
    }
}
