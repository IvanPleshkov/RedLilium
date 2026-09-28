//! Authoring of final rigid-body mass properties, independently of collider geometry.
use redlilium_core::math::{Quat, Vec2, Vec3};

/// Principal moments about the centre of mass and their axes in body-local space.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AngularInertia3D {
    pub principal: Vec3,
    pub rotation: Quat,
}
impl AngularInertia3D {
    pub fn diagonal(principal: Vec3) -> Self {
        Self {
            principal,
            rotation: Quat::identity(),
        }
    }
}

/// Explicit final mass. Missing centre/inertia are derived from density-weighted colliders.
/// Centre is body-local; inertia is central. Moving the centre preserves central inertia.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MassSettings2D {
    pub mass: f32,
    pub center_of_mass: Option<Vec2>,
    pub inertia: Option<f32>,
}
impl MassSettings2D {
    pub fn new(mass: f32) -> Self {
        Self {
            mass,
            center_of_mass: None,
            inertia: None,
        }
    }
    pub fn with_center_of_mass(mut self, value: Option<Vec2>) -> Self {
        self.center_of_mass = value;
        self
    }
    pub fn with_inertia(mut self, value: Option<f32>) -> Self {
        self.inertia = value;
        self
    }
    #[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
    pub(super) fn needs_geometry(self) -> bool {
        self.center_of_mass.is_none() || self.inertia.is_none()
    }
}
impl crate::ComponentField for Option<MassSettings2D> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut value = self.unwrap_or_else(|| MassSettings2D::new(1.0));
            if enabled {
                if let Some(mass) = value.mass.inspect_field("Final mass", ui, ctx) {
                    value.mass = mass;
                    changed = true;
                }
                let mut explicit_center = value.center_of_mass.is_some();
                if ui
                    .checkbox(&mut explicit_center, "Explicit centre of mass")
                    .changed()
                {
                    value.center_of_mass = explicit_center.then(Vec2::zeros);
                    changed = true;
                }
                if let Some(center) = value.center_of_mass {
                    if let Some(edited) = center.inspect_field("Local centre", ui, ctx) {
                        value.center_of_mass = Some(edited);
                        changed = true;
                    }
                }
                let mut explicit_inertia = value.inertia.is_some();
                if ui
                    .checkbox(&mut explicit_inertia, "Explicit central inertia")
                    .changed()
                {
                    value.inertia = explicit_inertia.then(|| 1.0);
                    changed = true;
                }
                if let Some(inertia) = value.inertia {
                    if let Some(edited) = inertia.inspect_field("Inertia", ui, ctx) {
                        value.inertia = Some(edited);
                        changed = true;
                    }
                }
            }
            changed.then(|| enabled.then_some(value))
        })
        .inner
    }
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<(), crate::serialize::SerializeError> {
        ctx.write_serde(name, self)
    }
    fn deserialize_field(
        name: &str,
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.read_serde(name)
    }
}

/// Explicit final mass. Missing centre/inertia are derived from density-weighted colliders.
/// Centre is body-local; inertia is central. Moving the centre preserves central inertia.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MassSettings3D {
    pub mass: f32,
    pub center_of_mass: Option<Vec3>,
    pub inertia: Option<AngularInertia3D>,
}
impl MassSettings3D {
    pub fn new(mass: f32) -> Self {
        Self {
            mass,
            center_of_mass: None,
            inertia: None,
        }
    }
    pub fn with_center_of_mass(mut self, value: Option<Vec3>) -> Self {
        self.center_of_mass = value;
        self
    }
    pub fn with_inertia(mut self, value: Option<AngularInertia3D>) -> Self {
        self.inertia = value;
        self
    }
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    pub(super) fn needs_geometry(self) -> bool {
        self.center_of_mass.is_none() || self.inertia.is_none()
    }
}
impl crate::ComponentField for Option<MassSettings3D> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut value = self.unwrap_or_else(|| MassSettings3D::new(1.0));
            if enabled {
                if let Some(mass) = value.mass.inspect_field("Final mass", ui, ctx) {
                    value.mass = mass;
                    changed = true;
                }
                let mut explicit_center = value.center_of_mass.is_some();
                if ui
                    .checkbox(&mut explicit_center, "Explicit centre of mass")
                    .changed()
                {
                    value.center_of_mass = explicit_center.then(Vec3::zeros);
                    changed = true;
                }
                if let Some(center) = value.center_of_mass {
                    if let Some(edited) = center.inspect_field("Local centre", ui, ctx) {
                        value.center_of_mass = Some(edited);
                        changed = true;
                    }
                }
                let mut explicit_inertia = value.inertia.is_some();
                if ui
                    .checkbox(&mut explicit_inertia, "Explicit central inertia")
                    .changed()
                {
                    value.inertia =
                        explicit_inertia.then(|| AngularInertia3D::diagonal(Vec3::repeat(1.0)));
                    changed = true;
                }
                if let Some(inertia) = value.inertia {
                    if let Some(edited) = inertia.inspect_field("Inertia", ui, ctx) {
                        value.inertia = Some(edited);
                        changed = true;
                    }
                }
            }
            changed.then(|| enabled.then_some(value))
        })
        .inner
    }
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<(), crate::serialize::SerializeError> {
        ctx.write_serde(name, self)
    }
    fn deserialize_field(
        name: &str,
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.read_serde(name)
    }
}

impl crate::ComponentField for AngularInertia3D {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut value = *self;
            let mut changed = false;
            if let Some(v) = self.principal.inspect_field("Principal moments", ui, ctx) {
                value.principal = v;
                changed = true;
            }
            // Explicit quaternion editing, normalized by validation/conversion.
            ui.horizontal(|ui| {
                ui.label("Local axes quaternion XYZW");
                for v in value.rotation.coords.iter_mut() {
                    changed |= ui.add(egui::DragValue::new(v).speed(0.01)).changed();
                }
            });
            changed.then_some(value)
        })
        .inner
    }
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<(), crate::serialize::SerializeError> {
        ctx.write_serde(name, self)
    }
    fn deserialize_field(
        name: &str,
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.read_serde(name)
    }
}

fn positive(entity: crate::Entity, name: &str, value: f32) -> Result<(), crate::SystemError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(super::validation::invalid(
            entity,
            &format!("mass_properties.{name} must be finite and positive"),
        ));
    }
    Ok(())
}
impl AngularInertia3D {
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    fn validate(self, entity: crate::Entity) -> Result<(), crate::SystemError> {
        for v in self.principal.iter() {
            positive(entity, "inertia.principal", *v)?;
        }
        let [x, y, z] = [
            self.principal.x as f64,
            self.principal.y as f64,
            self.principal.z as f64,
        ];
        let largest = x.max(y).max(z);
        if largest > (x + y + z - largest) + largest * 1e-6 {
            return Err(super::validation::invalid(
                entity,
                "mass_properties.inertia principal moments must satisfy triangle inequalities",
            ));
        }
        if !self.rotation.coords.iter().all(|v| v.is_finite())
            || !self.rotation.norm_squared().is_finite()
            || self.rotation.norm_squared() < 1e-12
        {
            return Err(super::validation::invalid(
                entity,
                "mass_properties.inertia.rotation must be a finite nonzero quaternion",
            ));
        }
        Ok(())
    }
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::super::rapier2d::prelude::*;
    use super::*;
    impl MassSettings2D {
        pub(in crate::std::physics) fn validate(
            self,
            entity: crate::Entity,
        ) -> Result<(), crate::SystemError> {
            positive(entity, "mass", self.mass)?;
            if self
                .center_of_mass
                .is_some_and(|v| !v.iter().all(|x| x.is_finite()))
            {
                return Err(super::super::validation::invalid(
                    entity,
                    "mass_properties.center_of_mass must be finite",
                ));
            }
            if let Some(inertia) = self.inertia {
                positive(entity, "inertia", inertia)?;
            }
            Ok(())
        }
        pub(in crate::std::physics) fn resolve(
            self,
            entity: crate::Entity,
            mut computed: MassProperties,
        ) -> Result<MassProperties, crate::SystemError> {
            self.validate(entity)?;
            if self.needs_geometry() {
                if !(computed.mass().is_finite() && computed.mass() > 0.0) {
                    return Err(super::super::validation::invalid(
                        entity,
                        "mass_properties needs positive finite collider mass to derive centre/inertia",
                    ));
                }
                computed.set_mass(self.mass as Real, self.inertia.is_none());
            } else {
                computed.set_mass(self.mass as Real, false);
            }
            if let Some(v) = self.center_of_mass {
                computed.local_com = Vector::new(v.x as Real, v.y as Real);
            }
            if let Some(inertia) = self.inertia {
                computed.inv_principal_inertia = 1.0 / inertia as Real;
            }
            let inertia_valid = computed.inv_principal_inertia.is_finite()
                && computed.inv_principal_inertia > 0.0
                && computed.principal_inertia().is_finite()
                && computed.principal_inertia() > 0.0;
            if !computed.local_com.is_finite()
                || !computed.inv_mass.is_finite()
                || computed.inv_mass <= 0.0
                || !inertia_valid
            {
                return Err(super::super::validation::invalid(
                    entity,
                    "mass_properties is not representable as finite positive mass and central inertia in the active physics precision",
                ));
            }
            Ok(computed)
        }
    }
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::super::rapier3d::prelude::*;
    use super::*;
    impl MassSettings3D {
        pub(in crate::std::physics) fn validate(
            self,
            entity: crate::Entity,
        ) -> Result<(), crate::SystemError> {
            positive(entity, "mass", self.mass)?;
            if self
                .center_of_mass
                .is_some_and(|v| !v.iter().all(|x| x.is_finite()))
            {
                return Err(super::super::validation::invalid(
                    entity,
                    "mass_properties.center_of_mass must be finite",
                ));
            }
            if let Some(inertia) = self.inertia {
                inertia.validate(entity)?;
            }
            Ok(())
        }
        pub(in crate::std::physics) fn resolve(
            self,
            entity: crate::Entity,
            mut computed: MassProperties,
        ) -> Result<MassProperties, crate::SystemError> {
            self.validate(entity)?;
            if self.needs_geometry() {
                if !(computed.mass().is_finite() && computed.mass() > 0.0) {
                    return Err(super::super::validation::invalid(
                        entity,
                        "mass_properties needs positive finite collider mass to derive centre/inertia",
                    ));
                }
                computed.set_mass(self.mass as Real, self.inertia.is_none());
            } else {
                computed.set_mass(self.mass as Real, false);
            }
            if let Some(v) = self.center_of_mass {
                computed.local_com = Vector::new(v.x as Real, v.y as Real, v.z as Real);
            }
            if let Some(inertia) = self.inertia {
                let q = inertia.rotation.normalize();
                computed.inv_principal_inertia = Vector::new(
                    1.0 / inertia.principal.x as Real,
                    1.0 / inertia.principal.y as Real,
                    1.0 / inertia.principal.z as Real,
                );
                computed.principal_inertia_local_frame =
                    Rotation::from_xyzw(q.i as Real, q.j as Real, q.k as Real, q.w as Real)
                        .normalize();
            }
            let principal = computed.principal_inertia();
            let max = principal.max_element();
            let inertia_valid = computed.inv_principal_inertia.is_finite()
                && computed.inv_principal_inertia.min_element() > 0.0
                && principal.is_finite()
                && principal.min_element() > 0.0
                && (max as f64)
                    <= (principal.x as f64 + principal.y as f64 + principal.z as f64 - max as f64)
                        + max as f64 * 1e-5
                && computed.principal_inertia_local_frame.is_finite();
            if !computed.local_com.is_finite()
                || !computed.inv_mass.is_finite()
                || computed.inv_mass <= 0.0
                || !inertia_valid
            {
                return Err(super::super::validation::invalid(
                    entity,
                    "mass_properties is not representable as finite positive mass and central inertia in the active physics precision",
                ));
            }
            Ok(computed)
        }
    }
}
