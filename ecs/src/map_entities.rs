//! Entity reference collection and remapping for components.
//!
//! Uses the same method-resolution trick as [`Inspect`](crate::inspect::Inspect):
//! inherent methods on wrapper types for known entity-holding types take
//! precedence over the blanket fallback trait impls (which are no-ops).
//!
//! The `#[derive(Component)]` macro generates `collect_entities` and
//! `remap_entities` by wrapping each field in [`EntityRef`] / [`EntityMut`].
//!
//! # Supported types
//!
//! - `Entity` — single entity reference
//! - `Vec<Entity>` — list of entity references
//! - `Option<Entity>` — optional entity reference
//!
//! All other types are silently skipped (no-op fallback).
//!
//! # Adding a new entity-holding type
//!
//! Add inherent `collect_entities` / `remap_entities` impls in this file:
//!
//! ```ignore
//! impl EntityRef<'_, MyEntitySet> {
//!     pub fn collect_entities(&self, collector: &mut Vec<Entity>) {
//!         collector.extend(self.0.iter());
//!     }
//! }
//!
//! impl EntityMut<'_, MyEntitySet> {
//!     pub fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {
//!         self.0.remap(map);
//!     }
//! }
//! ```

use crate::Entity;

// ---------------------------------------------------------------------------
// Read-only wrapper — collecting entity references
// ---------------------------------------------------------------------------

/// Read-only wrapper for collecting [`Entity`] references from a field.
///
/// Inherent `collect_entities` methods for known types take priority over
/// the [`EntityRefFallback`] blanket impl (which is a no-op).
pub struct EntityRef<'a, T: ?Sized>(pub &'a T);

/// Fallback trait for types that don't contain entity references.
///
/// The blanket impl is a no-op. Rust's method resolution ensures this is
/// only used when no inherent `collect_entities` method exists.
pub trait EntityRefFallback {
    fn collect_entities(&self, _collector: &mut Vec<Entity>) {}
}

impl<T: 'static> EntityRefFallback for EntityRef<'_, T> {}

impl EntityRef<'_, Entity> {
    pub fn collect_entities(&self, collector: &mut Vec<Entity>) {
        collector.push(*self.0);
    }
}

impl EntityRef<'_, Vec<Entity>> {
    pub fn collect_entities(&self, collector: &mut Vec<Entity>) {
        collector.extend(self.0.iter().copied());
    }
}

impl EntityRef<'_, Option<Entity>> {
    pub fn collect_entities(&self, collector: &mut Vec<Entity>) {
        if let Some(e) = self.0 {
            collector.push(*e);
        }
    }
}

// ---------------------------------------------------------------------------
// Mutable wrapper — remapping entity references
// ---------------------------------------------------------------------------

/// Mutable wrapper for remapping [`Entity`] references in a field.
///
/// Inherent `remap_entities` methods for known types take priority over
/// the [`EntityMutFallback`] blanket impl (which is a no-op).
pub struct EntityMut<'a, T: ?Sized>(pub &'a mut T);

/// Fallback trait for types that don't contain entity references.
///
/// The blanket impl is a no-op. Rust's method resolution ensures this is
/// only used when no inherent `remap_entities` method exists.
pub trait EntityMutFallback {
    fn remap_entities(&mut self, _map: &mut dyn FnMut(Entity) -> Entity) {}
}

impl<T: 'static> EntityMutFallback for EntityMut<'_, T> {}

impl EntityMut<'_, Entity> {
    pub fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {
        *self.0 = map(*self.0);
    }
}

impl EntityMut<'_, Vec<Entity>> {
    pub fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {
        for e in self.0.iter_mut() {
            *e = map(*e);
        }
    }
}

impl EntityMut<'_, Option<Entity>> {
    pub fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {
        if let Some(e) = self.0.as_mut() {
            *e = map(*e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(index: u32) -> Entity {
        // Use the public Entity test helper or construct via World;
        // here we cheat through the crate-visible constructor.
        Entity::new(index, 0)
    }

    // -- EntityRef tests --

    #[test]
    fn collect_single_entity() {
        let e = make_entity(42);
        let mut collector = Vec::new();
        EntityRef(&e).collect_entities(&mut collector);
        assert_eq!(collector, vec![e]);
    }

    #[test]
    fn collect_vec_entities() {
        let entities = vec![make_entity(1), make_entity(2), make_entity(3)];
        let mut collector = Vec::new();
        EntityRef(&entities).collect_entities(&mut collector);
        assert_eq!(collector, entities);
    }

    #[test]
    fn collect_option_some() {
        let e = Some(make_entity(7));
        let mut collector = Vec::new();
        EntityRef(&e).collect_entities(&mut collector);
        assert_eq!(collector, vec![make_entity(7)]);
    }

    #[test]
    fn collect_option_none() {
        let e: Option<Entity> = None;
        let mut collector = Vec::new();
        EntityRef(&e).collect_entities(&mut collector);
        assert!(collector.is_empty());
    }

    #[test]
    fn collect_fallback_noop() {
        let value = 42u32;
        let mut collector = Vec::new();
        use super::EntityRefFallback as _;
        EntityRef(&value).collect_entities(&mut collector);
        assert!(collector.is_empty());
    }

    // -- EntityMut tests --

    #[test]
    fn remap_single_entity() {
        let mut e = make_entity(1);
        EntityMut(&mut e).remap_entities(&mut |_| make_entity(99));
        assert_eq!(e, make_entity(99));
    }

    #[test]
    fn remap_vec_entities() {
        let mut entities = vec![make_entity(1), make_entity(2), make_entity(3)];
        EntityMut(&mut entities)
            .remap_entities(&mut |e| Entity::new(e.index() + 10, e.spawn_tick()));
        assert_eq!(
            entities,
            vec![make_entity(11), make_entity(12), make_entity(13)]
        );
    }

    #[test]
    fn remap_option_some() {
        let mut e = Some(make_entity(5));
        EntityMut(&mut e).remap_entities(&mut |_| make_entity(50));
        assert_eq!(e, Some(make_entity(50)));
    }

    #[test]
    fn remap_option_none() {
        let mut e: Option<Entity> = None;
        EntityMut(&mut e).remap_entities(&mut |_| make_entity(50));
        assert_eq!(e, None);
    }

    #[test]
    fn remap_fallback_noop() {
        let mut value = 42u32;
        use super::EntityMutFallback as _;
        EntityMut(&mut value).remap_entities(&mut |_| make_entity(99));
        assert_eq!(value, 42);
    }
}
