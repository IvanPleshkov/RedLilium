//! Runtime reflection for ECS components.
//!
//! The [`Component`] trait provides field-level introspection, enabling editors
//! and debug tools to enumerate, read, and write component fields at runtime
//! without knowing the concrete type at compile time.
//!
//! Use `#[derive(Component)]` from [`ecs_macro`] to auto-implement the trait.

use std::any::{Any, TypeId};

/// Metadata describing a single field of a reflected component.
#[derive(Debug, Clone)]
pub struct FieldInfo {
    /// Field name (`"translation"` for named fields, `"0"` for tuple fields).
    pub name: &'static str,
    /// Human-readable type name (from [`core::any::type_name`]).
    pub type_name: &'static str,
    /// Runtime [`TypeId`] for downcasting via [`Any`].
    pub type_id: TypeId,
}

/// Trait for reflected ECS components.
///
/// Provides runtime access to component fields by name, enabling editor
/// property panels and serialization without compile-time type knowledge.
///
/// # Derive
///
/// ```ignore
/// use redlilium_ecs::Component;
///
/// #[derive(Component)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
/// ```
///
/// # Manual Implementation
///
/// ```ignore
/// use redlilium_ecs::{Component, FieldInfo};
///
/// struct Marker;
///
/// impl Component for Marker {
///     fn component_name(&self) -> &'static str { "Marker" }
///     fn field_infos(&self) -> &'static [FieldInfo] { &[] }
///     fn field(&self, _name: &str) -> Option<&dyn Any> { None }
///     fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Any> { None }
/// }
/// ```
pub trait Component: Send + Sync + 'static {
    /// Returns the struct name (e.g. `"Transform"`).
    fn component_name(&self) -> &'static str;

    /// Returns metadata for all reflected fields.
    fn field_infos(&self) -> &'static [FieldInfo];

    /// Returns a reference to the field with the given name, as `&dyn Any`.
    ///
    /// Use [`Any::downcast_ref`] to get the concrete type.
    fn field(&self, name: &str) -> Option<&dyn Any>;

    /// Returns a mutable reference to the field with the given name, as `&mut dyn Any`.
    ///
    /// Use [`Any::downcast_mut`] to get the concrete type.
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Any>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestComponent {
        value: f32,
        label: String,
    }

    impl Component for TestComponent {
        fn component_name(&self) -> &'static str {
            "TestComponent"
        }

        fn field_infos(&self) -> &'static [FieldInfo] {
            static INFOS: std::sync::LazyLock<Vec<FieldInfo>> = std::sync::LazyLock::new(|| {
                vec![
                    FieldInfo {
                        name: "value",
                        type_name: std::any::type_name::<f32>(),
                        type_id: TypeId::of::<f32>(),
                    },
                    FieldInfo {
                        name: "label",
                        type_name: std::any::type_name::<String>(),
                        type_id: TypeId::of::<String>(),
                    },
                ]
            });
            &INFOS
        }

        fn field(&self, name: &str) -> Option<&dyn Any> {
            match name {
                "value" => Some(&self.value),
                "label" => Some(&self.label),
                _ => None,
            }
        }

        fn field_mut(&mut self, name: &str) -> Option<&mut dyn Any> {
            match name {
                "value" => Some(&mut self.value),
                "label" => Some(&mut self.label),
                _ => None,
            }
        }
    }

    #[test]
    fn component_name() {
        let c = TestComponent {
            value: 42.0,
            label: "hello".into(),
        };
        assert_eq!(c.component_name(), "TestComponent");
    }

    #[test]
    fn field_infos_match() {
        let c = TestComponent {
            value: 0.0,
            label: String::new(),
        };
        let infos = c.field_infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].name, "value");
        assert_eq!(infos[0].type_id, TypeId::of::<f32>());
        assert_eq!(infos[1].name, "label");
        assert_eq!(infos[1].type_id, TypeId::of::<String>());
    }

    #[test]
    fn field_read() {
        let c = TestComponent {
            value: 42.5,
            label: "test".into(),
        };
        let v = c.field("value").unwrap().downcast_ref::<f32>().unwrap();
        assert_eq!(*v, 42.5);
        let l = c.field("label").unwrap().downcast_ref::<String>().unwrap();
        assert_eq!(l.as_str(), "test");
        assert!(c.field("nonexistent").is_none());
    }

    #[test]
    fn field_write() {
        let mut c = TestComponent {
            value: 1.0,
            label: "one".into(),
        };
        *c.field_mut("value").unwrap().downcast_mut::<f32>().unwrap() = 2.0;
        assert_eq!(c.value, 2.0);
        *c.field_mut("label")
            .unwrap()
            .downcast_mut::<String>()
            .unwrap() = "two".into();
        assert_eq!(c.label, "two");
    }
}
