use crate::component::Component;
use crate::entity::Entity;
use crate::sparse_set::InspectResult;
use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};

use super::World;

/// Reversible action that replaces a component on an entity.
///
/// Produced by the inspector when the user edits component fields.
/// Stores both old and new values for undo/redo.
struct SetComponentAction<T: Component + Clone> {
    entity: Entity,
    old: T,
    new: T,
}

impl<T: Component + Clone> std::fmt::Debug for SetComponentAction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetComponentAction")
            .field("component", &T::NAME)
            .finish()
    }
}

impl<T: Component + Clone> EditAction<World> for SetComponentAction<T> {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        let _ = world.insert(self.entity, self.new.clone());
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        let _ = world.insert(self.entity, self.old.clone());
        Ok(())
    }

    fn description(&self) -> &str {
        T::NAME
    }

    fn merge(&mut self, other: Box<dyn EditAction<World>>) -> Option<Box<dyn EditAction<World>>> {
        if let Some(other) = other.as_any().downcast_ref::<Self>()
            && self.entity == other.entity
        {
            self.new = other.new.clone();
            return None; // consumed — keep first old, use latest new
        }
        Some(other)
    }
}

/// A boxed set-component action for programmatic edits (the gizmo drag
/// path). Same merge semantics as inspector edits: consecutive actions on
/// the same entity+type collapse (first `old`, latest `new`) — a drag
/// becomes one undo entry.
pub fn set_component_action<T: Component + Clone>(
    entity: Entity,
    old: T,
    new: T,
) -> Box<dyn EditAction<World>> {
    Box::new(SetComponentAction { entity, old, new })
}

/// Creates an [`InspectResult`] that replaces a component value on an entity.
///
/// This is the standard way for [`Component::inspect_ui`] implementations to
/// report an edit. The derive macro calls this automatically. Manual impls
/// should call it when the user edits a field value:
///
/// ```ignore
/// fn inspect_ui(&self, ui: &mut egui::Ui, world: &World, entity: Entity) -> InspectResult {
///     // ... show widgets, compute new_value ...
///     set_component_actions(entity, self.clone(), new_value)
/// }
/// ```
pub fn set_component_actions<T: Component>(entity: Entity, old: T, new: T) -> InspectResult {
    Some(vec![Box::new(SetComponentAction { entity, old, new })])
}
