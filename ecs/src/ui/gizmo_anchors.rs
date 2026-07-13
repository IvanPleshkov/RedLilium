//! Gizmo anchor providers (#85): any component can expose draggable
//! world-space control points, edited through the standard undo machinery.
//!
//! The pattern mirrors the inspector: a component implements
//! [`GizmoAnchors`], registers via
//! [`register_gizmo_anchors`](crate::World::register_gizmo_anchors), and the
//! editor's gizmo orchestrator discovers anchors type-erased through
//! `ComponentMeta` — it never names concrete component types. `Transform` is
//! just the built-in provider (one anchor at its origin); a level-generator
//! component with a dozen control corners plugs in identically.
//!
//! Drags become [`SetComponentAction`]-style undoable edits: consecutive
//! deltas of one drag merge into a single undo entry (see
//! [`set_component_action`](crate::set_component_action) and the history's
//! merge rules); the drag-end barrier is the consumer's job.

use redlilium_core::abstract_editor::EditAction;
use redlilium_core::math::{Mat4, Vec3};

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// One draggable control point exposed by a component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoAnchor {
    /// Provider-stable identifier (e.g. a corner index). Passed back to
    /// [`GizmoAnchors::apply_drag`].
    pub id: u32,
    /// World-space position of the anchor.
    pub position: Vec3,
}

/// Context handed to anchor providers: the entity and its parent's global
/// transform (identity for roots), so providers convert between their local
/// data and the gizmo's world-space positions/deltas themselves.
#[derive(Debug, Clone, Copy)]
pub struct AnchorCtx {
    pub entity: Entity,
    /// The parent's `GlobalTransform` matrix, identity when unparented.
    pub parent_global: Mat4,
}

impl AnchorCtx {
    /// Transform a world-space delta into the provider's parent space
    /// (rotation/scale only — deltas are vectors, not points).
    pub fn world_delta_to_parent(&self, world_delta: Vec3) -> Vec3 {
        self.parent_global
            .try_inverse()
            .map(|inv| inv.transform_vector(&world_delta))
            .unwrap_or(world_delta)
    }
}

/// A component with draggable gizmo control points.
///
/// `anchors` reports world-space points; `apply_drag` mutates the component
/// by a world-space delta on one of them. The editor wraps `apply_drag` into
/// a clone-old/clone-new undoable action, so implementations just move data.
pub trait GizmoAnchors: Component + Clone {
    /// The component's control points, in world space.
    fn anchors(&self, ctx: &AnchorCtx) -> Vec<GizmoAnchor>;

    /// Apply a world-space translation to the anchor `id`.
    fn apply_drag(&mut self, id: u32, world_delta: Vec3, ctx: &AnchorCtx);
}

/// Monomorphized `ComponentMeta::gizmo_anchors_fn` body.
fn anchors_fn<T: GizmoAnchors>(world: &World, entity: Entity, ctx: &AnchorCtx) -> Vec<GizmoAnchor> {
    world
        .get::<T>(entity)
        .map(|c| c.anchors(ctx))
        .unwrap_or_default()
}

/// Monomorphized `ComponentMeta::gizmo_drag_fn` body: clone-old, mutate,
/// wrap into the standard merging set-component action.
fn drag_fn<T: GizmoAnchors>(
    world: &World,
    entity: Entity,
    id: u32,
    world_delta: Vec3,
    ctx: &AnchorCtx,
) -> Option<Box<dyn EditAction<World>>> {
    let old = world.get::<T>(entity)?.clone();
    let mut new = old.clone();
    new.apply_drag(id, world_delta, ctx);
    Some(crate::world::set_component_action::<T>(entity, old, new))
}

impl World {
    /// Register `T`'s gizmo anchors. Call **after**
    /// [`register_inspector`](World::register_inspector) — the provider fns
    /// live in the component's meta (so a game module's providers are purged
    /// with its storage on unload).
    pub fn register_gizmo_anchors<T: GizmoAnchors>(&mut self) {
        let updated = self.with_component_meta_mut::<T>(|meta| {
            meta.gizmo_anchors_fn = Some(anchors_fn::<T>);
            meta.gizmo_drag_fn = Some(drag_fn::<T>);
        });
        debug_assert!(
            updated,
            "register_gizmo_anchors::<{}> requires register_inspector first",
            T::NAME
        );
    }

    /// The anchor context for an entity: its parent's global transform
    /// (identity for roots or when the parent has no `GlobalTransform`).
    pub fn anchor_ctx(&self, entity: Entity) -> AnchorCtx {
        let parent_global = self
            .get::<crate::Parent>(entity)
            .map(|p| p.0)
            .and_then(|p| self.get::<crate::GlobalTransform>(p).map(|g| g.0))
            .unwrap_or_else(Mat4::identity);
        AnchorCtx {
            entity,
            parent_global,
        }
    }

    /// Every gizmo anchor exposed by `entity`'s components, tagged with the
    /// providing component's name. Order: component registration order is
    /// not guaranteed; anchor ids are stable per provider.
    pub fn gizmo_anchors_of(&self, entity: Entity) -> Vec<(&'static str, GizmoAnchor)> {
        if !self.is_alive(entity) {
            return Vec::new();
        }
        let ctx = self.anchor_ctx(entity);
        let mut out = Vec::new();
        for (name, f) in self.gizmo_anchor_providers() {
            for anchor in f(self, entity, &ctx) {
                out.push((name, anchor));
            }
        }
        out
    }

    /// Build the undoable action for dragging `anchor_id` of `component` on
    /// `entity` by `world_delta`. `None` when the entity/component/provider
    /// is gone (a stale drag — the consumer drops it).
    pub fn gizmo_drag_action(
        &self,
        entity: Entity,
        component: &str,
        anchor_id: u32,
        world_delta: Vec3,
    ) -> Option<Box<dyn EditAction<World>>> {
        if !self.is_alive(entity) {
            return None;
        }
        let ctx = self.anchor_ctx(entity);
        let f = self.gizmo_drag_provider(component)?;
        f(self, entity, anchor_id, world_delta, &ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GlobalTransform, Transform};
    use redlilium_core::abstract_editor::EditActionHistory;

    /// A miniature "level generator": two world-space control corners.
    #[derive(Clone, crate::Component)]
    struct Corridor {
        a: Vec3,
        b: Vec3,
    }

    impl GizmoAnchors for Corridor {
        fn anchors(&self, _ctx: &AnchorCtx) -> Vec<GizmoAnchor> {
            vec![
                GizmoAnchor {
                    id: 0,
                    position: self.a,
                },
                GizmoAnchor {
                    id: 1,
                    position: self.b,
                },
            ]
        }
        fn apply_drag(&mut self, id: u32, world_delta: Vec3, _ctx: &AnchorCtx) {
            match id {
                0 => self.a += world_delta,
                _ => self.b += world_delta,
            }
        }
    }

    fn corridor_world() -> (World, Entity) {
        let mut world = World::new();
        world.register_inspector::<Corridor>();
        world.register_gizmo_anchors::<Corridor>();
        let e = world.spawn();
        world
            .insert(
                e,
                Corridor {
                    a: Vec3::new(0.0, 0.0, 0.0),
                    b: Vec3::new(4.0, 0.0, 0.0),
                },
            )
            .unwrap();
        (world, e)
    }

    #[test]
    fn custom_component_exposes_its_anchors() {
        let (world, e) = corridor_world();
        let anchors = world.gizmo_anchors_of(e);
        assert_eq!(anchors.len(), 2);
        assert!(anchors.iter().all(|(name, _)| *name == Corridor::NAME));
        assert_eq!(anchors[1].1.position, Vec3::new(4.0, 0.0, 0.0));
    }

    /// The #85 acceptance core: a whole drag (many deltas) on a custom
    /// provider collapses into ONE undo entry, and undo restores the
    /// pre-drag value exactly.
    #[test]
    fn drag_merges_into_single_undo_entry() {
        let (mut world, e) = corridor_world();
        let mut history: EditActionHistory<World> = EditActionHistory::new(64);

        // Ten per-frame deltas on corner 1, as the gizmo would emit them.
        for _ in 0..10 {
            let action = world
                .gizmo_drag_action(e, Corridor::NAME, 1, Vec3::new(0.1, 0.0, 0.0))
                .expect("provider present");
            history.execute(action, &mut world).unwrap();
        }
        let b = world.get::<Corridor>(e).unwrap().b;
        assert!((b.x - 5.0).abs() < 1e-5, "10 × 0.1 applied, b.x = {}", b.x);

        // One undo reverts the entire drag.
        history.undo(&mut world).unwrap();
        let b = world.get::<Corridor>(e).unwrap().b;
        assert!((b.x - 4.0).abs() < 1e-5, "single undo reverts the drag");
        assert!(!history.can_undo(), "the whole drag was one entry");
    }

    /// Transform is just the built-in provider: one anchor at the (world)
    /// origin of the entity; drags convert into parent space.
    #[test]
    fn transform_provider_handles_rotated_parent() {
        let mut world = World::new();
        world.register_inspector::<Transform>();
        world.register_inspector::<GlobalTransform>();
        world.register_component::<crate::Parent>();
        world.register_component::<crate::Children>();
        world.register_gizmo_anchors::<Transform>();

        // Parent rotated 90° around Y: parent-local +X points to world -Z.
        let parent = world.spawn();
        let parent_tf = Transform::from_rotation(redlilium_core::math::quat_from_rotation_y(
            std::f32::consts::FRAC_PI_2,
        ));
        world.insert(parent, parent_tf).unwrap();
        world
            .insert(parent, GlobalTransform(parent_tf.to_matrix()))
            .unwrap();

        let child = world.spawn();
        world.insert(child, Transform::IDENTITY).unwrap();
        world
            .insert(child, GlobalTransform(Mat4::identity()))
            .unwrap();
        crate::set_parent(&mut world, child, parent);

        // Drag the child by world +X: its LOCAL translation must move along
        // the axis that maps to world +X under the parent's rotation.
        let mut action = world
            .gizmo_drag_action(child, Transform::NAME, 0, Vec3::new(1.0, 0.0, 0.0))
            .expect("provider present");
        action.apply(&mut world).unwrap();

        let local = world.get::<Transform>(child).unwrap().translation;
        // inv(rotY(90°)) * (1,0,0) = (0,0,1)
        assert!(local.x.abs() < 1e-4, "local = {local:?}");
        assert!((local.z - 1.0).abs() < 1e-4, "local = {local:?}");
    }
}
