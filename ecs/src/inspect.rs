//! Compile-time field inspector for ECS components.
//!
//! Uses the [`Inspect`] wrapper with Rust's method resolution priority:
//! inherent `show()` methods for known types take precedence over the
//! blanket [`InspectFallback`] trait, which displays unknown types as opaque.
//!
//! The `#[derive(Component)]` macro generates `inspect_ui` by wrapping each
//! field in `Inspect(&mut self.field).show("field", ui)`.
//!
//! # Adding a new inspectable type
//!
//! Add an inherent `show` impl on `Inspect<'_, YourType>` in this file:
//!
//! ```ignore
//! impl Inspect<'_, MyVec2> {
//!     pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
//!         ui.horizontal(|ui| {
//!             ui.label(name);
//!             ui.add(egui::DragValue::new(&mut self.0.x).speed(0.01));
//!             ui.add(egui::DragValue::new(&mut self.0.y).speed(0.01));
//!         });
//!     }
//! }
//! ```

use redlilium_core::math::{Mat4, Quat, Vec2, Vec3, Vec4};

use crate::StringId;

/// Wrapper that enables compile-time field inspector dispatch.
///
/// Inherent `show()` methods for known types take priority over the
/// [`InspectFallback`] blanket trait impl, which shows "(opaque)".
pub struct Inspect<'a, T: ?Sized>(pub &'a mut T);

/// Fallback trait for types without a specific inspector.
///
/// The blanket impl displays the field as "(opaque)". Rust's method
/// resolution ensures this is only used when no inherent `show()` method
/// exists on `Inspect<'_, T>`.
pub trait InspectFallback {
    fn show(&mut self, name: &str, ui: &mut egui::Ui);
}

impl<T: 'static> InspectFallback for Inspect<'_, T> {
    fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.weak(format!("({})", std::any::type_name::<T>()));
        });
    }
}

// ---------------------------------------------------------------------------
// Primitive type inspectors
// ---------------------------------------------------------------------------

impl Inspect<'_, f32> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self.0).speed(0.01));
        });
    }
}

impl Inspect<'_, f64> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self.0).speed(0.01));
        });
    }
}

impl Inspect<'_, bool> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.checkbox(self.0, "");
        });
    }
}

impl Inspect<'_, u8> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self.0 as i32;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=255))
                .changed()
            {
                *self.0 = v as u8;
            }
        });
    }
}

impl Inspect<'_, u32> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self.0 as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=u32::MAX as i64))
                .changed()
            {
                *self.0 = v as u32;
            }
        });
    }
}

impl Inspect<'_, i32> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self.0));
        });
    }
}

impl Inspect<'_, u64> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self.0 as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                .changed()
            {
                *self.0 = v as u64;
            }
        });
    }
}

impl Inspect<'_, usize> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self.0 as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                .changed()
            {
                *self.0 = v as usize;
            }
        });
    }
}

impl Inspect<'_, String> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.text_edit_singleline(self.0);
        });
    }
}

impl Inspect<'_, StringId> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!("StringId({})", self.0.0));
        });
    }
}

// ---------------------------------------------------------------------------
// Math type inspectors (via redlilium_core re-exports of nalgebra)
// ---------------------------------------------------------------------------

impl Inspect<'_, Vec2> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(
                egui::DragValue::new(&mut self.0.x)
                    .speed(0.01)
                    .prefix("x: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.y)
                    .speed(0.01)
                    .prefix("y: "),
            );
        });
    }
}

impl Inspect<'_, Vec3> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(
                egui::DragValue::new(&mut self.0.x)
                    .speed(0.01)
                    .prefix("x: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.y)
                    .speed(0.01)
                    .prefix("y: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.z)
                    .speed(0.01)
                    .prefix("z: "),
            );
        });
    }
}

impl Inspect<'_, Vec4> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(
                egui::DragValue::new(&mut self.0.x)
                    .speed(0.01)
                    .prefix("x: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.y)
                    .speed(0.01)
                    .prefix("y: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.z)
                    .speed(0.01)
                    .prefix("z: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.0.w)
                    .speed(0.01)
                    .prefix("w: "),
            );
        });
    }
}

impl Inspect<'_, Quat> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!(
                "[{:.3}, {:.3}, {:.3}, {:.3}]",
                self.0.coords.x, self.0.coords.y, self.0.coords.z, self.0.coords.w
            ));
        });
    }
}

impl Inspect<'_, Mat4> {
    pub fn show(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label("(matrix)");
        });
    }
}
