use redlilium_core::math::Vec3;

/// A directional light (e.g., sunlight).
///
/// Direction comes from the entity's [`GlobalTransform`](crate::GlobalTransform)
/// forward vector.
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct DirectionalLight {
    /// Light color (linear RGB).
    pub color: Vec3,
    /// Light intensity in lux.
    pub intensity: f32,
}

impl DirectionalLight {
    /// Create a new directional light.
    pub fn new(color: Vec3, intensity: f32) -> Self {
        Self { color, intensity }
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }
    }
}

/// A point light that emits in all directions from a position.
///
/// Position comes from the entity's [`GlobalTransform`](crate::GlobalTransform)
/// translation.
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct PointLight {
    /// Light color (linear RGB).
    pub color: Vec3,
    /// Light intensity in candela.
    pub intensity: f32,
    /// Maximum range. Beyond this, attenuation is zero.
    /// Zero means infinite range.
    pub range: f32,
}

impl PointLight {
    /// Create a new point light with infinite range.
    pub fn new(color: Vec3, intensity: f32) -> Self {
        Self {
            color,
            intensity,
            range: 0.0,
        }
    }

    /// Set the maximum range.
    #[must_use]
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
            range: 0.0,
        }
    }
}

/// A spot light that emits in a cone from a position.
///
/// Position and direction come from the entity's
/// [`GlobalTransform`](crate::GlobalTransform).
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct SpotLight {
    /// Light color (linear RGB).
    pub color: Vec3,
    /// Light intensity in candela.
    pub intensity: f32,
    /// Maximum range. Zero means infinite range.
    pub range: f32,
    /// Inner cone angle in radians (full intensity within this cone).
    pub inner_cone_angle: f32,
    /// Outer cone angle in radians (light fades to zero at this angle).
    pub outer_cone_angle: f32,
}

impl SpotLight {
    /// Create a new spot light with the given cone angles and infinite range.
    pub fn new(color: Vec3, intensity: f32, inner_cone_angle: f32, outer_cone_angle: f32) -> Self {
        Self {
            color,
            intensity,
            range: 0.0,
            inner_cone_angle,
            outer_cone_angle,
        }
    }

    /// Set the maximum range.
    #[must_use]
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
            range: 0.0,
            inner_cone_angle: std::f32::consts::FRAC_PI_8,
            outer_cone_angle: std::f32::consts::FRAC_PI_4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_default() {
        let light = DirectionalLight::default();
        assert_eq!(light.color, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(light.intensity, 1.0);
    }

    #[test]
    fn point_with_range() {
        let light = PointLight::new(Vec3::new(1.0, 0.0, 0.0), 100.0).with_range(50.0);
        assert_eq!(light.range, 50.0);
        assert_eq!(light.color, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn spot_default_angles() {
        let light = SpotLight::default();
        assert_eq!(light.inner_cone_angle, std::f32::consts::FRAC_PI_8);
        assert_eq!(light.outer_cone_angle, std::f32::consts::FRAC_PI_4);
    }
}
