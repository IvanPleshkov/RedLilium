//! Compile-time field inspector for ECS components.
//!
//! Uses the [`Inspect`] wrapper with Rust's method resolution priority:
//! the generic inherent `show()` method for [`ComponentField`](crate::ComponentField)
//! types takes precedence over the blanket [`InspectFallback`] trait, which
//! displays unknown types as opaque.
//!
//! The `#[derive(Component)]` macro generates `inspect_ui` by wrapping each
//! field in `Inspect(&self.field).show("field", ui)`.
//!
//! # Adding a new inspectable type
//!
//! Implement [`ComponentField`](crate::ComponentField) for your type to get
//! integrated inspection, serialization, and deserialization support:
//!
//! ```ignore
//! impl ComponentField for MyVec2 {
//!     fn inspect_field(&self, name: &str, ui: &mut egui::Ui) -> Option<Self> {
//!         let mut v = *self;
//!         let changed = ui.horizontal(|ui| {
//!             ui.label(name);
//!             let x = ui.add(egui::DragValue::new(&mut v.0.x).speed(0.01)).changed();
//!             let y = ui.add(egui::DragValue::new(&mut v.0.y).speed(0.01)).changed();
//!             x || y
//!         }).inner;
//!         changed.then_some(v)
//!     }
//!     // ... serialize_field, deserialize_field
//! }
//! ```

/// Wrapper that enables compile-time field inspector dispatch.
///
/// The generic inherent `show()` method for [`ComponentField`](crate::ComponentField)
/// types takes priority over the [`InspectFallback`] blanket trait impl,
/// which shows "(opaque)".
pub struct Inspect<'a, T: ?Sized>(pub &'a T);

/// Fallback trait for types without a [`ComponentField`](crate::ComponentField)
/// implementation.
///
/// The blanket impl displays the field as "(opaque)" and returns `None`
/// (no edit possible). Rust's method resolution ensures this is only used
/// when no inherent `show()` method exists on `Inspect<'_, T>`.
pub trait InspectFallback<T> {
    fn show(&self, name: &str, ui: &mut egui::Ui) -> Option<T>;
}

impl<T: Clone + 'static> InspectFallback<T> for Inspect<'_, T> {
    fn show(&self, name: &str, ui: &mut egui::Ui) -> Option<T> {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.weak(format!("({})", std::any::type_name::<T>()));
        });
        None
    }
}

// ---------------------------------------------------------------------------
// Generic dispatch to ComponentField trait
// ---------------------------------------------------------------------------

impl<T: crate::component_field::ComponentField> Inspect<'_, T> {
    pub fn show(&self, name: &str, ui: &mut egui::Ui) -> Option<T> {
        self.0.inspect_field(name, ui)
    }
}
