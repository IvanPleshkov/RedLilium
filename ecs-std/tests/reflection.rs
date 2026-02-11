use redlilium_core::math::{Mat4, Quat, Vec3, mat4_from_translation, quat_from_rotation_y};
use redlilium_ecs::{Component, FieldKind};

use ecs_std::components::*;

// ---------------------------------------------------------------------------
// Named struct reflection (Transform)
// ---------------------------------------------------------------------------

#[test]
fn transform_component_name() {
    let t = Transform::IDENTITY;
    assert_eq!(t.component_name(), "Transform");
}

#[test]
fn transform_field_infos() {
    let t = Transform::IDENTITY;
    let infos = t.field_infos();
    assert_eq!(infos.len(), 3);

    assert_eq!(infos[0].name, "translation");
    assert_eq!(infos[0].kind, FieldKind::Vec3);

    assert_eq!(infos[1].name, "rotation");
    assert_eq!(infos[1].kind, FieldKind::Quat);

    assert_eq!(infos[2].name, "scale");
    assert_eq!(infos[2].kind, FieldKind::Vec3);
}

#[test]
fn transform_field_read() {
    let t = Transform::new(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::identity(),
        Vec3::new(1.0, 1.0, 1.0),
    );

    let translation = t
        .field("translation")
        .unwrap()
        .downcast_ref::<Vec3>()
        .unwrap();
    assert_eq!(*translation, Vec3::new(1.0, 2.0, 3.0));

    let scale = t.field("scale").unwrap().downcast_ref::<Vec3>().unwrap();
    assert_eq!(*scale, Vec3::new(1.0, 1.0, 1.0));

    assert!(t.field("nonexistent").is_none());
}

#[test]
fn transform_field_write() {
    let mut t = Transform::IDENTITY;

    *t.field_mut("translation")
        .unwrap()
        .downcast_mut::<Vec3>()
        .unwrap() = Vec3::new(10.0, 20.0, 30.0);

    assert_eq!(t.translation, Vec3::new(10.0, 20.0, 30.0));
}

// ---------------------------------------------------------------------------
// Tuple struct reflection (GlobalTransform)
// ---------------------------------------------------------------------------

#[test]
fn global_transform_component_name() {
    let gt = GlobalTransform::IDENTITY;
    assert_eq!(gt.component_name(), "GlobalTransform");
}

#[test]
fn global_transform_tuple_field() {
    let gt = GlobalTransform(mat4_from_translation(Vec3::new(5.0, 0.0, 0.0)));
    let infos = gt.field_infos();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].name, "0");
    assert_eq!(infos[0].kind, FieldKind::Mat4);

    let mat = gt.field("0").unwrap().downcast_ref::<Mat4>().unwrap();
    assert_eq!(mat[(0, 3)], 5.0);
}

#[test]
fn global_transform_field_write() {
    let mut gt = GlobalTransform::IDENTITY;
    *gt.field_mut("0").unwrap().downcast_mut::<Mat4>().unwrap() =
        mat4_from_translation(Vec3::new(42.0, 0.0, 0.0));
    assert_eq!(gt.translation(), Vec3::new(42.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Camera reflection (Pod: 2 Mat4 fields)
// ---------------------------------------------------------------------------

#[test]
fn camera_field_infos() {
    let cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    let infos = cam.field_infos();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].name, "view_matrix");
    assert_eq!(infos[0].kind, FieldKind::Mat4);
    assert_eq!(infos[1].name, "projection_matrix");
    assert_eq!(infos[1].kind, FieldKind::Mat4);
}

#[test]
fn camera_read_view_matrix() {
    let cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    let view = cam
        .field("view_matrix")
        .unwrap()
        .downcast_ref::<Mat4>()
        .unwrap();
    assert_eq!(*view, Mat4::identity());
}

#[test]
fn camera_write_view_matrix() {
    let mut cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    let new_view = mat4_from_translation(Vec3::new(1.0, 2.0, 3.0));
    *cam.field_mut("view_matrix")
        .unwrap()
        .downcast_mut::<Mat4>()
        .unwrap() = new_view;
    assert_eq!(cam.view_matrix, new_view);
}

// ---------------------------------------------------------------------------
// Visibility (tuple struct with bool)
// ---------------------------------------------------------------------------

#[test]
fn visibility_reflection() {
    let v = Visibility::VISIBLE;
    assert_eq!(v.component_name(), "Visibility");

    let infos = v.field_infos();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].name, "0");
    assert_eq!(infos[0].kind, FieldKind::Bool);

    let val = v.field("0").unwrap().downcast_ref::<bool>().unwrap();
    assert!(*val);
}

// ---------------------------------------------------------------------------
// Name (tuple struct with String)
// ---------------------------------------------------------------------------

#[test]
fn name_reflection() {
    let n = Name::new("TestEntity");
    assert_eq!(n.component_name(), "Name");

    let infos = n.field_infos();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].name, "0");
    assert_eq!(infos[0].kind, FieldKind::String);

    let val = n.field("0").unwrap().downcast_ref::<String>().unwrap();
    assert_eq!(val, "TestEntity");
}

#[test]
fn name_field_write() {
    let mut n = Name::new("old");
    *n.field_mut("0").unwrap().downcast_mut::<String>().unwrap() = "new".to_string();
    assert_eq!(n.as_str(), "new");
}

// ---------------------------------------------------------------------------
// Light components
// ---------------------------------------------------------------------------

#[test]
fn directional_light_reflection() {
    let light = DirectionalLight::new(Vec3::new(1.0, 0.9, 0.8), 100000.0);
    assert_eq!(light.component_name(), "DirectionalLight");

    let infos = light.field_infos();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].name, "color");
    assert_eq!(infos[0].kind, FieldKind::Vec3);
    assert_eq!(infos[1].name, "intensity");
    assert_eq!(infos[1].kind, FieldKind::F32);

    let intensity = light
        .field("intensity")
        .unwrap()
        .downcast_ref::<f32>()
        .unwrap();
    assert_eq!(*intensity, 100000.0);
}

#[test]
fn point_light_reflection() {
    let light = PointLight::new(Vec3::new(1.0, 1.0, 1.0), 50.0).with_range(25.0);
    assert_eq!(light.component_name(), "PointLight");

    let infos = light.field_infos();
    assert_eq!(infos.len(), 3);
    assert_eq!(infos[0].name, "color");
    assert_eq!(infos[1].name, "intensity");
    assert_eq!(infos[2].name, "range");

    let range = light.field("range").unwrap().downcast_ref::<f32>().unwrap();
    assert_eq!(*range, 25.0);
}

#[test]
fn spot_light_reflection() {
    let light = SpotLight::default();
    assert_eq!(light.component_name(), "SpotLight");

    let infos = light.field_infos();
    assert_eq!(infos.len(), 5);
    assert_eq!(infos[0].name, "color");
    assert_eq!(infos[1].name, "intensity");
    assert_eq!(infos[2].name, "range");
    assert_eq!(infos[3].name, "inner_cone_angle");
    assert_eq!(infos[4].name, "outer_cone_angle");
}

#[test]
fn spot_light_field_write() {
    let mut light = SpotLight::default();
    *light
        .field_mut("intensity")
        .unwrap()
        .downcast_mut::<f32>()
        .unwrap() = 999.0;
    assert_eq!(light.intensity, 999.0);
}

// ---------------------------------------------------------------------------
// Dynamic iteration pattern (editor-like use case)
// ---------------------------------------------------------------------------

#[test]
fn enumerate_all_fields_dynamically() {
    let t = Transform::new(
        Vec3::new(1.0, 2.0, 3.0),
        quat_from_rotation_y(0.5),
        Vec3::new(2.0, 2.0, 2.0),
    );

    // An editor would do this to build a property panel
    let infos = t.field_infos();
    assert_eq!(infos.len(), 3);

    for info in infos {
        let _field_ref = t.field(info.name).expect("field should exist");
        // With FieldKind, the editor knows the type without TypeId
        assert!(
            matches!(info.kind, FieldKind::Vec3 | FieldKind::Quat),
            "Transform fields should be Vec3 or Quat"
        );
    }
}

// ---------------------------------------------------------------------------
// Pod byte serialization (only for types that remain Pod)
// ---------------------------------------------------------------------------

#[test]
fn pod_components_serialization() {
    // Verify Pod component types can still be serialized to/from bytes
    let transform = Transform::IDENTITY;
    let bytes = bytemuck::bytes_of(&transform);
    assert_eq!(bytes.len(), std::mem::size_of::<Transform>());

    let gt = GlobalTransform::IDENTITY;
    let bytes = bytemuck::bytes_of(&gt);
    assert_eq!(bytes.len(), std::mem::size_of::<GlobalTransform>());

    let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0);
    let bytes = bytemuck::bytes_of(&cam);
    assert_eq!(bytes.len(), std::mem::size_of::<Camera>());

    let dl = DirectionalLight::default();
    let bytes = bytemuck::bytes_of(&dl);
    assert_eq!(bytes.len(), std::mem::size_of::<DirectionalLight>());

    let pl = PointLight::default();
    let bytes = bytemuck::bytes_of(&pl);
    assert_eq!(bytes.len(), std::mem::size_of::<PointLight>());

    let sl = SpotLight::default();
    let bytes = bytemuck::bytes_of(&sl);
    assert_eq!(bytes.len(), std::mem::size_of::<SpotLight>());
}

#[test]
fn transform_pod_roundtrip() {
    let original = Transform::new(
        Vec3::new(1.0, 2.0, 3.0),
        quat_from_rotation_y(0.5),
        Vec3::new(2.0, 2.0, 2.0),
    );
    let bytes = bytemuck::bytes_of(&original);
    let restored: &Transform = bytemuck::from_bytes(bytes);
    assert_eq!(*restored, original);
}

// ---------------------------------------------------------------------------
// Non-Pod component verification
// ---------------------------------------------------------------------------

#[test]
fn non_pod_components_have_reflection() {
    // Visibility (now bool-based, non-Pod)
    let v = Visibility::VISIBLE;
    assert_eq!(v.component_name(), "Visibility");

    // Name (now String-based, non-Pod)
    let n = Name::new("test");
    assert_eq!(n.component_name(), "Name");

    // Parent (new: has Component derive)
    // Entity creation requires World::spawn(), so test via world
    let mut world = redlilium_ecs::World::new();
    ecs_std::register_std_components(&mut world);
    let entity = world.spawn();
    let p = Parent(entity);
    assert_eq!(p.component_name(), "Parent");

    // Children (new: has Component derive)
    let c = Children::default();
    assert_eq!(c.component_name(), "Children");
}

// ---------------------------------------------------------------------------
// All component names
// ---------------------------------------------------------------------------

#[test]
fn all_component_names() {
    assert_eq!(Transform::IDENTITY.component_name(), "Transform");
    assert_eq!(
        GlobalTransform::IDENTITY.component_name(),
        "GlobalTransform"
    );
    assert_eq!(
        Camera::perspective(1.0, 1.0, 0.1, 100.0).component_name(),
        "Camera"
    );
    assert_eq!(Visibility::VISIBLE.component_name(), "Visibility");
    assert_eq!(Name::default().component_name(), "Name");
    assert_eq!(
        DirectionalLight::default().component_name(),
        "DirectionalLight"
    );
    assert_eq!(PointLight::default().component_name(), "PointLight");
    assert_eq!(SpotLight::default().component_name(), "SpotLight");
}
