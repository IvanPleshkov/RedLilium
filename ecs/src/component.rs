//! Runtime reflection for ECS components.
//!
//! The [`Component`] trait provides field-level introspection, enabling editors
//! and WASM plugins to enumerate, read, and write component fields at runtime.
//!
//! All `Component` types must be [`bytemuck::Pod`] (plain old data), ensuring
//! they can be trivially serialized to bytes for cross-boundary communication.
//!
//! Use `#[derive(Component)]` from [`ecs_macro`] to auto-implement the trait.

use std::any::Any;

/// Portable type descriptor for a component field.
///
/// Used by editors and WASM plugins to determine the semantic type of a field
/// without relying on Rust's [`TypeId`](std::any::TypeId) (which is
/// compilation-unit-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FieldKind {
    F32 = 0,
    U8 = 1,
    U32 = 2,
    I32 = 3,
    Vec2 = 4,
    Vec3 = 5,
    Vec4 = 6,
    Quat = 7,
    Mat4 = 8,
    StringId = 9,
}

impl FieldKind {
    /// Fixed byte size of a value of this kind.
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I32 => 4,
            Self::Vec2 => 8,
            Self::Vec3 => 12,
            Self::Vec4 => 16,
            Self::Quat => 16,
            Self::Mat4 => 64,
            Self::StringId => 4,
        }
    }
}

/// Metadata describing a single field of a reflected component.
#[derive(Debug, Clone)]
pub struct FieldInfo {
    /// Field name (`"translation"` for named fields, `"0"` for tuple fields).
    pub name: &'static str,
    /// Human-readable type name (from [`core::any::type_name`]).
    pub type_name: &'static str,
    /// Portable type descriptor for WASM and editor integration.
    pub kind: FieldKind,
}

/// Trait for reflected Pod ECS components.
///
/// All implementors must be [`bytemuck::Pod`], enabling whole-component
/// byte serialization via `bytemuck::bytes_of()` / `from_bytes()`.
///
/// # Derive
///
/// ```ignore
/// use redlilium_ecs::Component;
///
/// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
/// #[repr(C)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
/// ```
pub trait Component: bytemuck::Pod + Send + Sync {
    /// Returns the struct name (e.g. `"Transform"`).
    fn component_name(&self) -> &'static str;

    /// Returns metadata for all reflected fields.
    fn field_infos(&self) -> &'static [FieldInfo];

    /// Returns a reference to the field with the given name, as `&dyn Any`.
    fn field(&self, name: &str) -> Option<&dyn Any>;

    /// Returns a mutable reference to the field with the given name, as `&mut dyn Any`.
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Any>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct TestComponent {
        value: f32,
        count: u32,
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
                        kind: FieldKind::F32,
                    },
                    FieldInfo {
                        name: "count",
                        type_name: std::any::type_name::<u32>(),
                        kind: FieldKind::U32,
                    },
                ]
            });
            &INFOS
        }

        fn field(&self, name: &str) -> Option<&dyn Any> {
            match name {
                "value" => Some(&self.value),
                "count" => Some(&self.count),
                _ => None,
            }
        }

        fn field_mut(&mut self, name: &str) -> Option<&mut dyn Any> {
            match name {
                "value" => Some(&mut self.value),
                "count" => Some(&mut self.count),
                _ => None,
            }
        }
    }

    #[test]
    fn component_name() {
        let c = TestComponent {
            value: 42.0,
            count: 7,
        };
        assert_eq!(c.component_name(), "TestComponent");
    }

    #[test]
    fn field_infos_match() {
        let c = TestComponent {
            value: 0.0,
            count: 0,
        };
        let infos = c.field_infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].name, "value");
        assert_eq!(infos[0].kind, FieldKind::F32);
        assert_eq!(infos[1].name, "count");
        assert_eq!(infos[1].kind, FieldKind::U32);
    }

    #[test]
    fn field_read() {
        let c = TestComponent {
            value: 42.5,
            count: 10,
        };
        let v = c.field("value").unwrap().downcast_ref::<f32>().unwrap();
        assert_eq!(*v, 42.5);
        let n = c.field("count").unwrap().downcast_ref::<u32>().unwrap();
        assert_eq!(*n, 10);
        assert!(c.field("nonexistent").is_none());
    }

    #[test]
    fn field_write() {
        let mut c = TestComponent {
            value: 1.0,
            count: 1,
        };
        *c.field_mut("value").unwrap().downcast_mut::<f32>().unwrap() = 2.0;
        assert_eq!(c.value, 2.0);
        *c.field_mut("count").unwrap().downcast_mut::<u32>().unwrap() = 99;
        assert_eq!(c.count, 99);
    }

    #[test]
    fn pod_byte_serialization() {
        let c = TestComponent {
            value: 1.5,
            count: 42,
        };
        let bytes = bytemuck::bytes_of(&c);
        assert_eq!(bytes.len(), 8); // f32 + u32
        let restored: &TestComponent = bytemuck::from_bytes(bytes);
        assert_eq!(restored.value, 1.5);
        assert_eq!(restored.count, 42);
    }

    #[test]
    fn field_kind_byte_sizes() {
        assert_eq!(FieldKind::F32.byte_size(), 4);
        assert_eq!(FieldKind::U8.byte_size(), 1);
        assert_eq!(FieldKind::Vec3.byte_size(), 12);
        assert_eq!(FieldKind::Quat.byte_size(), 16);
        assert_eq!(FieldKind::Mat4.byte_size(), 64);
        assert_eq!(FieldKind::StringId.byte_size(), 4);
    }
}
