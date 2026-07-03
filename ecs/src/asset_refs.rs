//! Asset-reference visitation for components.
//!
//! Uses the same method-resolution trick as
//! [`EntityRef`](crate::map_entities::EntityRef): inherent methods on wrapper
//! types for fields that implement [`ComponentField`](crate::ComponentField)
//! take precedence over the blanket fallback trait impls (which are no-ops).
//!
//! The `#[derive(Component)]` macro generates
//! [`Component::visit_asset_refs`](crate::Component::visit_asset_refs) /
//! [`visit_asset_refs_mut`](crate::Component::visit_asset_refs_mut) by wrapping
//! each field in [`AssetRefsRef`] / [`AssetRefsMut`]. Fields whose type
//! implements `ComponentField` forward to its
//! [`visit_asset_refs`](crate::ComponentField::visit_asset_refs) hook (a no-op
//! by default; `AssetRef<S>` overrides it to yield itself); all other fields are
//! silently skipped.
//!
//! The visited refs are passed as `&dyn Any` — the generic asset-sync system
//! downcasts them to the concrete `AssetRef<S>` types it has resolvers for.

use std::any::Any;

use crate::ComponentField;

// ---------------------------------------------------------------------------
// Read-only wrapper
// ---------------------------------------------------------------------------

/// Read-only wrapper for visiting asset references in a field.
pub struct AssetRefsRef<'a, T: ?Sized>(pub &'a T);

/// Fallback trait for field types that don't carry asset references.
pub trait AssetRefsRefFallback {
    fn visit_asset_refs(&self, _f: &mut dyn FnMut(&dyn Any)) {}
}

impl<T: 'static> AssetRefsRefFallback for AssetRefsRef<'_, T> {}

impl<T: ComponentField> AssetRefsRef<'_, T> {
    pub fn visit_asset_refs(&self, f: &mut dyn FnMut(&dyn Any)) {
        self.0.visit_asset_refs(f);
    }
}

// ---------------------------------------------------------------------------
// Mutable wrapper
// ---------------------------------------------------------------------------

/// Mutable wrapper for visiting asset references in a field.
pub struct AssetRefsMut<'a, T: ?Sized>(pub &'a mut T);

/// Fallback trait for field types that don't carry asset references.
pub trait AssetRefsMutFallback {
    fn visit_asset_refs_mut(&mut self, _f: &mut dyn FnMut(&mut dyn Any)) {}
}

impl<T: 'static> AssetRefsMutFallback for AssetRefsMut<'_, T> {}

impl<T: ComponentField> AssetRefsMut<'_, T> {
    pub fn visit_asset_refs_mut(&mut self, f: &mut dyn FnMut(&mut dyn Any)) {
        self.0.visit_asset_refs_mut(f);
    }
}
