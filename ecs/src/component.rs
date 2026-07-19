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
///     fn inspect_ui(&self, ui: &mut egui::Ui, world: &World, entity: Entity) -> InspectResult {
///         // custom inspector layout; return edit actions if changed
///         None
///     }
/// }
/// ```
pub trait Component: Clone + Send + Sync + 'static {
    /// The struct name as a static string (e.g. `"Transform"`).
    ///
    /// Used by the World's inspector registration to key metadata
    /// without requiring an instance.
    const NAME: &'static str;

    /// The schema version of this component type.
    ///
    /// Incremented when the component's serialized fields change, allowing
    /// migrations to transform old schema data to new schema during snapshot
    /// restore. Defaults to 1; derived components can override via
    /// `#[schema_version = N]` attribute.
    const SCHEMA_VERSION: u32 = 1;

    /// Returns the struct name (e.g. `"Transform"`).
    fn component_name(&self) -> &'static str {
        Self::NAME
    }

    /// Render an inspector UI for this component's fields.
    ///
    /// Returns `Some(actions)` if the user edited any field, or `None` if
    /// nothing changed. The `world` parameter provides read access to ECS
    /// resources. Use [`set_component_actions`](crate::set_component_actions)
    /// to create the standard "replace component" action.
    ///
    /// The derive macro generates this by calling
    /// [`Inspect::show`](crate::inspect::Inspect) for each field and
    /// wrapping the result with [`set_component_actions`](crate::set_component_actions).
    ///
    /// Override manually for custom undo/redo actions (e.g., GPU resource rebuild).
    fn inspect_ui(
        &self,
        ui: &mut egui::Ui,
        world: &crate::World,
        entity: crate::Entity,
    ) -> crate::InspectResult;

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

    /// Visit all asset references stored in this component (read-only), each
    /// passed as `&dyn Any` (a concrete `AssetRef<S>`).
    ///
    /// The derive macro generates this by wrapping each field in
    /// [`AssetRefsRef`](crate::asset_refs::AssetRefsRef) — fields implementing
    /// [`ComponentField`](crate::ComponentField) forward to its hook (an
    /// `AssetRef` yields itself), all others are skipped. The generic asset-sync
    /// system uses this to find refs to resolve / hot-reload.
    ///
    /// The default implementation is a no-op (no asset references).
    fn visit_asset_refs(&self, _f: &mut dyn FnMut(&dyn std::any::Any)) {}

    /// Visit all asset references stored in this component (mutably) — the
    /// write half of [`visit_asset_refs`](Self::visit_asset_refs), used by the
    /// sync system to apply re-resolutions.
    ///
    /// The default implementation is a no-op (no asset references).
    fn visit_asset_refs_mut(&mut self, _f: &mut dyn FnMut(&mut dyn std::any::Any)) {}

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

    /// Returns the axis-aligned bounding box for this component, if applicable.
    ///
    /// The default implementation returns `None`. Components with spatial
    /// extent (e.g. meshes, colliders) should override this. The `world`
    /// parameter allows looking up resources (e.g. mesh AABB caches).
    fn aabb(&self, _world: &crate::World) -> Option<redlilium_core::math::Aabb> {
        None
    }

    /// Compute a deterministic schema hash for this component type.
    ///
    /// The name-keyed, cross-image schema identity: `#[derive(Component)]`
    /// generates it from the schema version plus the visible fields' names
    /// and type tokens in declaration order, so two images compiled from the
    /// same source agree and any reshape (field added/removed/retyped/
    /// reordered) shifts it. Consumers: snapshot restore logs schema drift
    /// (tolerated — see `World::deserialize_world_into`), and the editor's
    /// Tier-1 behavior diff compares a static plugin image against a rebuilt
    /// game cdylib (ADR-037).
    ///
    /// The default implementation returns an empty string (no schema
    /// tracking) — that's what `#[skip_serialization]` components keep.
    fn schema_hash() -> String {
        String::new()
    }

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

        fn inspect_ui(
            &self,
            ui: &mut egui::Ui,
            world: &crate::World,
            entity: crate::Entity,
        ) -> crate::InspectResult {
            #[allow(unused_imports)]
            use crate::inspect::InspectFallback as _;
            let ctx = &crate::FieldInspectCtx { world, entity };
            let mut _changed = false;
            let value = match crate::inspect::Inspect(&self.value).show("value", ui, ctx) {
                Some(v) => {
                    _changed = true;
                    v
                }
                None => self.value,
            };
            let count = match crate::inspect::Inspect(&self.count).show("count", ui, ctx) {
                Some(v) => {
                    _changed = true;
                    v
                }
                None => self.count,
            };
            if _changed {
                crate::set_component_actions(entity, *self, Self { value, count })
            } else {
                None
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

    #[derive(Clone)]
    struct RichComponent {
        label: String,
        _data: Arc<Vec<u8>>,
    }

    impl Component for RichComponent {
        const NAME: &'static str = "RichComponent";

        fn inspect_ui(
            &self,
            ui: &mut egui::Ui,
            world: &crate::World,
            entity: crate::Entity,
        ) -> crate::InspectResult {
            #[allow(unused_imports)]
            use crate::inspect::InspectFallback as _;
            let ctx = &crate::FieldInspectCtx { world, entity };
            let mut _changed = false;
            let label = match crate::inspect::Inspect(&self.label).show("label", ui, ctx) {
                Some(v) => {
                    _changed = true;
                    v
                }
                None => self.label.clone(),
            };
            let _data = match crate::inspect::Inspect(&self._data).show("data", ui, ctx) {
                Some(v) => {
                    _changed = true;
                    v
                }
                None => self._data.clone(),
            };
            if _changed {
                crate::set_component_actions(entity, self.clone(), Self { label, _data })
            } else {
                None
            }
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
