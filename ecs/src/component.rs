//! Runtime reflection for ECS components.
//!
//! The [`Component`] trait provides component-level introspection with
//! integrated egui inspector support via [`inspect_ui`](Component::inspect_ui).
//!
//! Components can be any `Send + Sync + 'static` type. Types that also implement
//! [`bytemuck::Pod`] additionally support byte-level serialization, GPU upload,
//! and snapshot/rollback.
//!
//! Use `#[derive(Component)]` from [`ecs_macro`] to auto-implement the trait.

/// Trait for reflected ECS components.
///
/// Components can be any `Send + Sync + 'static` type. The derive macro
/// generates [`inspect_ui`](Self::inspect_ui) using the [`Inspect`](crate::inspect::Inspect)
/// wrapper for compile-time field dispatch.
///
/// # Deriving
///
/// ```ignore
/// #[derive(Component)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
/// ```
///
/// # Manual implementation
///
/// ```ignore
/// impl Component for CustomType {
///     const NAME: &'static str = "CustomType";
///     fn inspect_ui(&mut self, ui: &mut egui::Ui) {
///         // custom inspector layout
///     }
/// }
/// ```
pub trait Component: Send + Sync + 'static {
    /// The struct name as a static string (e.g. `"Transform"`).
    ///
    /// Used by the World's inspector registration to key metadata
    /// without requiring an instance.
    const NAME: &'static str;

    /// Returns the struct name (e.g. `"Transform"`).
    fn component_name(&self) -> &'static str {
        Self::NAME
    }

    /// Render an inspector UI for this component's fields.
    ///
    /// The derive macro generates this by calling
    /// [`Inspect::show`](crate::inspect::Inspect) for each field.
    fn inspect_ui(&mut self, ui: &mut egui::Ui);

    /// Collect all [`Entity`](crate::Entity) references stored in this component.
    ///
    /// The derive macro generates this by wrapping each field in
    /// [`EntityRef`](crate::map_entities::EntityRef). Fields of type `Entity`,
    /// `Vec<Entity>`, and `Option<Entity>` are collected; all others are skipped.
    ///
    /// The default implementation is a no-op (no entity references).
    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    /// Remap all [`Entity`](crate::Entity) references stored in this component.
    ///
    /// The derive macro generates this by wrapping each field in
    /// [`EntityMut`](crate::map_entities::EntityMut). Fields of type `Entity`,
    /// `Vec<Entity>`, and `Option<Entity>` are remapped; all others are skipped.
    ///
    /// The default implementation is a no-op (no entity references).
    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    /// Register required components for this type.
    ///
    /// Called automatically by [`World::register_inspector`](crate::World::register_inspector)
    /// and [`World::register_inspector_default`](crate::World::register_inspector_default).
    ///
    /// The `#[derive(Component)]` macro generates this from `#[require(...)]` attributes:
    ///
    /// ```ignore
    /// #[derive(Component)]
    /// #[require(Transform, GlobalTransform, Visibility)]
    /// struct Camera { /* ... */ }
    /// ```
    ///
    /// The default implementation does nothing.
    fn register_required(_world: &mut crate::World) {}

    /// Serialize this component's fields into a [`Value`](crate::serialize::Value).
    ///
    /// The `#[derive(Component)]` macro generates this automatically using
    /// [`SerializeField`](crate::serialize::SerializeField) wrappers for
    /// each field. Use `#[skip_serialization]` on the component struct to
    /// opt out and use this default (which returns `NotSerializable`).
    ///
    /// For custom serialization (e.g., GPU resources), override this method
    /// manually and use `ctx.world()` to access resources.
    fn serialize_component(
        &self,
        _ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        Err(crate::serialize::SerializeError::NotSerializable {
            component: Self::NAME,
        })
    }

    /// Deserialize a component from a [`DeserializeContext`](crate::serialize::DeserializeContext).
    ///
    /// The `#[derive(Component)]` macro generates this automatically using
    /// [`DeserializeField`](crate::serialize::DeserializeField) wrappers.
    /// Use `#[skip_serialization]` to opt out.
    ///
    /// For custom deserialization, override this method manually and use
    /// `ctx.world()` / `ctx.world_mut()` to access resources.
    fn deserialize_component(
        _ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError>
    where
        Self: Sized,
    {
        Err(crate::serialize::DeserializeError::NotDeserializable {
            component: Self::NAME.to_string(),
        })
    }
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
        const NAME: &'static str = "TestComponent";

        fn inspect_ui(&mut self, ui: &mut egui::Ui) {
            crate::inspect::Inspect(&mut self.value).show("value", ui);
            crate::inspect::Inspect(&mut self.count).show("count", ui);
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

    // --- Non-Pod component test ---

    use std::sync::Arc;

    struct RichComponent {
        label: String,
        _data: Arc<Vec<u8>>,
    }

    impl Component for RichComponent {
        const NAME: &'static str = "RichComponent";

        fn inspect_ui(&mut self, ui: &mut egui::Ui) {
            #[allow(unused_imports)]
            use crate::inspect::InspectFallback as _;
            crate::inspect::Inspect(&mut self.label).show("label", ui);
            crate::inspect::Inspect(&mut self._data).show("data", ui);
        }
    }

    #[test]
    fn non_pod_component_name() {
        let c = RichComponent {
            label: "hello".to_string(),
            _data: Arc::new(vec![1, 2, 3]),
        };
        assert_eq!(c.component_name(), "RichComponent");
    }
}
