use std::any::TypeId;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use redlilium_ecs::Component;

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
    assert_eq!(infos[0].type_id, TypeId::of::<Vec3>());

    assert_eq!(infos[1].name, "rotation");
    assert_eq!(infos[1].type_id, TypeId::of::<Quat>());

    assert_eq!(infos[2].name, "scale");
    assert_eq!(infos[2].type_id, TypeId::of::<Vec3>());
}

#[test]
fn transform_field_read() {
    let t = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);

    let translation = t
        .field("translation")
        .unwrap()
        .downcast_ref::<Vec3>()
        .unwrap();
    assert_eq!(*translation, Vec3::new(1.0, 2.0, 3.0));

    let scale = t.field("scale").unwrap().downcast_ref::<Vec3>().unwrap();
    assert_eq!(*scale, Vec3::ONE);

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
    let gt = GlobalTransform(Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
    let infos = gt.field_infos();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].name, "0");
    assert_eq!(infos[0].type_id, TypeId::of::<Mat4>());

    let mat = gt.field("0").unwrap().downcast_ref::<Mat4>().unwrap();
    assert_eq!(mat.w_axis.x, 5.0);
}

#[test]
fn global_transform_field_write() {
    let mut gt = GlobalTransform::IDENTITY;
    *gt.field_mut("0").unwrap().downcast_mut::<Mat4>().unwrap() =
        Mat4::from_translation(Vec3::new(42.0, 0.0, 0.0));
    assert_eq!(gt.translation(), Vec3::new(42.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Camera reflection
// ---------------------------------------------------------------------------

#[test]
fn camera_field_infos() {
    let cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    let infos = cam.field_infos();
    assert_eq!(infos.len(), 4);
    assert_eq!(infos[0].name, "projection");
    assert_eq!(infos[1].name, "active");
    assert_eq!(infos[1].type_id, TypeId::of::<bool>());
    assert_eq!(infos[2].name, "view_matrix");
    assert_eq!(infos[3].name, "projection_matrix");
}

#[test]
fn camera_read_active() {
    let cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    let active = cam.field("active").unwrap().downcast_ref::<bool>().unwrap();
    assert!(*active);
}

#[test]
fn camera_write_active() {
    let mut cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
    *cam.field_mut("active")
        .unwrap()
        .downcast_mut::<bool>()
        .unwrap() = false;
    assert!(!cam.active);
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
    assert_eq!(infos[0].type_id, TypeId::of::<bool>());

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

    let val = n.field("0").unwrap().downcast_ref::<String>().unwrap();
    assert_eq!(val.as_str(), "TestEntity");
}

#[test]
fn name_field_write() {
    let mut n = Name::new("Old");
    *n.field_mut("0").unwrap().downcast_mut::<String>().unwrap() = "New".into();
    assert_eq!(n.as_str(), "New");
}

// ---------------------------------------------------------------------------
// MeshRenderer (Arc fields)
// ---------------------------------------------------------------------------

#[test]
fn mesh_renderer_field_infos() {
    use redlilium_core::material::{AlphaMode, CpuMaterial, CpuMaterialInstance};
    use redlilium_core::mesh::{CpuMesh, VertexLayout};

    let layout = VertexLayout::position_only();
    let mesh = Arc::new(CpuMesh::new(layout.clone()));
    let decl = Arc::new(CpuMaterial::pbr_metallic_roughness(
        layout,
        AlphaMode::Opaque,
        false,
        false,
        false,
        false,
        false,
        false,
    ));
    let material = Arc::new(CpuMaterialInstance::new(decl));
    let mr = MeshRenderer::new(mesh, material);

    assert_eq!(mr.component_name(), "MeshRenderer");
    let infos = mr.field_infos();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].name, "mesh");
    assert_eq!(infos[1].name, "material");
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
    assert_eq!(infos[1].name, "intensity");
    assert_eq!(infos[1].type_id, TypeId::of::<f32>());

    let intensity = light
        .field("intensity")
        .unwrap()
        .downcast_ref::<f32>()
        .unwrap();
    assert_eq!(*intensity, 100000.0);
}

#[test]
fn point_light_reflection() {
    let light = PointLight::new(Vec3::ONE, 50.0).with_range(25.0);
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
        Quat::from_rotation_y(0.5),
        Vec3::splat(2.0),
    );

    // An editor would do this to build a property panel
    let infos = t.field_infos();
    assert_eq!(infos.len(), 3);

    for info in infos {
        let field_ref = t.field(info.name).expect("field should exist");
        // Verify the type_id matches what Any reports
        assert_eq!(field_ref.type_id(), info.type_id);
    }
}

#[test]
fn trait_object_dispatch() {
    let components: Vec<Box<dyn Component>> = vec![
        Box::new(Transform::IDENTITY),
        Box::new(GlobalTransform::IDENTITY),
        Box::new(Visibility::VISIBLE),
        Box::new(Name::new("test")),
        Box::new(DirectionalLight::default()),
        Box::new(PointLight::default()),
        Box::new(SpotLight::default()),
    ];

    let names: Vec<&str> = components.iter().map(|c| c.component_name()).collect();
    assert_eq!(
        names,
        vec![
            "Transform",
            "GlobalTransform",
            "Visibility",
            "Name",
            "DirectionalLight",
            "PointLight",
            "SpotLight",
        ]
    );

    // Total field count across all components
    let total_fields: usize = components.iter().map(|c| c.field_infos().len()).sum();
    assert_eq!(total_fields, 3 + 1 + 1 + 1 + 2 + 3 + 5); // = 16
}
