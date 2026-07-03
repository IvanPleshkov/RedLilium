//! A user component with an `AssetRef` field gets asset-ref visitation (and
//! thus loading + hot reload) purely through `#[derive(Component)]` — the
//! derive chains each field's `ComponentField::visit_asset_refs` hook.

#![cfg(feature = "rendering")]

use redlilium_ecs::{AssetRef, Component, MeshGenerator, MeshSource, World};

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

/// The editor guarantee end-to-end (minus egui): replacing a primitive's
/// material ref — what the inspector's drop target produces via
/// `set_component_actions` — flows through the ActionQueue + history and is
/// fully undoable, restoring both the identity and the cached resolution.
#[test]
fn mesh_renderer_ref_edit_is_undoable() {
    use redlilium_assets::Guid;
    use redlilium_core::abstract_editor::{ActionQueue, EditActionHistory};
    use redlilium_ecs::{MaterialInstanceSource, MeshRenderer, Primitive};

    let mut world = World::new();
    redlilium_ecs::register_std_components(&mut world);
    redlilium_ecs::register_rendering_components(&mut world);
    world.insert_resource(ActionQueue::<World>::new());
    let mut history = EditActionHistory::<World>::new(64);

    let old_mat = MaterialInstanceSource {
        guid: Guid::stable("materials/a.matinst"),
    };
    let new_mat = MaterialInstanceSource {
        guid: Guid::stable("materials/b.matinst"),
    };

    let entity = world.spawn();
    world
        .insert(
            entity,
            MeshRenderer::single(Primitive::new(
                MeshSource::Generated(redlilium_ecs::MeshGenerator::cube(0.5)),
                old_mat.clone(),
            )),
        )
        .unwrap();

    // What the inspector's drop handler does: clone, replace the ref, emit
    // the set-component actions into the queue (the editor's dispatch path).
    {
        let renderer = world.get::<MeshRenderer>(entity).unwrap();
        let mut edited = renderer.clone();
        edited.primitives[0].material = AssetRef::new(new_mat.clone());
        let actions = redlilium_ecs::set_component_actions(entity, renderer.clone(), edited)
            .expect("edit produces actions");
        let queue = world.resource::<ActionQueue<World>>();
        for action in actions {
            queue.push(action);
        }
    }

    // The editor's frame loop: drain the queue through the history.
    let actions = world.resource::<ActionQueue<World>>().drain();
    assert_eq!(actions.len(), 1, "one recorded action for the drop");
    for action in actions {
        history.execute(action, &mut world).unwrap();
    }
    assert_eq!(
        world.get::<MeshRenderer>(entity).unwrap().primitives[0]
            .material
            .source(),
        &new_mat,
        "apply replaced the material identity"
    );

    history.undo(&mut world).unwrap();
    assert_eq!(
        world.get::<MeshRenderer>(entity).unwrap().primitives[0]
            .material
            .source(),
        &old_mat,
        "undo restored the material identity"
    );
}

/// Asset-record edits are undoable too: the settings action applies through
/// the history, feeds hot reload (ChangedAssets) and persistence (DirtyMounts)
/// on BOTH apply and undo, and undo restores the previous record settings.
#[test]
fn asset_settings_edit_is_undoable() {
    use redlilium_assets::{AssetDb, AssetPath};
    use redlilium_core::abstract_editor::EditActionHistory;
    use redlilium_ecs::{ChangedAssets, DirtyMounts, SetAssetSettingsAction};

    let mut world = World::new();
    let mut db = AssetDb::new();
    let guid = db.register_path(AssetPath::new("std", "materials/m.material"), "material", 0);
    db.set_settings(
        &guid,
        Some("(shading_model:\"opaque\",properties:[])".into()),
    );
    world.insert_resource(db);
    world.insert_resource(ChangedAssets::new());
    world.insert_resource(DirtyMounts::new());
    let mut history = EditActionHistory::<World>::new(64);

    let old = world
        .resource::<AssetDb>()
        .record(&guid)
        .unwrap()
        .settings
        .clone();
    let new = Some("(shading_model:\"opaque_textured\",properties:[])".to_owned());
    history
        .execute(
            Box::new(SetAssetSettingsAction {
                guid,
                old: old.clone(),
                new: new.clone(),
            }),
            &mut world,
        )
        .unwrap();
    assert_eq!(
        world.resource::<AssetDb>().record(&guid).unwrap().settings,
        new,
        "apply wrote the new settings"
    );
    assert_eq!(
        world.resource_mut::<ChangedAssets>().drain(),
        vec![guid],
        "apply feeds hot reload"
    );
    assert_eq!(
        world.resource_mut::<DirtyMounts>().drain(),
        vec!["std".to_owned()],
        "apply marks the mount for persistence"
    );

    history.undo(&mut world).unwrap();
    assert_eq!(
        world.resource::<AssetDb>().record(&guid).unwrap().settings,
        old,
        "undo restored the old settings"
    );
    assert_eq!(
        world.resource_mut::<ChangedAssets>().drain(),
        vec![guid],
        "undo feeds hot reload (the revert applies live)"
    );
    assert!(
        !world.resource_mut::<DirtyMounts>().drain().is_empty(),
        "undo marks the mount for persistence"
    );
}
