//! A user component with an `AssetRef` field gets asset-ref visitation (and
//! thus loading + hot reload) purely through `#[derive(Component)]` — the
//! derive chains each field's `ComponentField::visit_asset_refs` hook.

#![cfg(feature = "rendering")]

use redlilium_ecs::{AssetRef, Component, MeshGenerator, MeshSource};

#[derive(Clone, Component)]
struct DecorMesh {
    strength: f32,
    mesh: AssetRef<MeshSource>,
}

fn test_component() -> DecorMesh {
    DecorMesh {
        strength: 0.5,
        mesh: AssetRef::new(MeshSource::Generated(MeshGenerator::cube(0.5))),
    }
}

/// The derived read hook yields exactly the `AssetRef` field (the `f32` is
/// skipped by the fallback), downcastable to its concrete type.
#[test]
fn derive_visits_asset_refs() {
    let comp = test_component();
    let mut seen = 0;
    comp.visit_asset_refs(&mut |any| {
        let r = any
            .downcast_ref::<AssetRef<MeshSource>>()
            .expect("visited ref downcasts to AssetRef<MeshSource>");
        assert!(r.get().is_none(), "starts unresolved");
        seen += 1;
    });
    assert_eq!(seen, 1);
}

/// The mutable hook reaches the same ref for in-place resolution.
#[test]
fn derive_visits_asset_refs_mut() {
    let mut comp = test_component();
    let mut seen = 0;
    comp.visit_asset_refs_mut(&mut |any| {
        assert!(any.downcast_mut::<AssetRef<MeshSource>>().is_some());
        seen += 1;
    });
    assert_eq!(seen, 1);
}
