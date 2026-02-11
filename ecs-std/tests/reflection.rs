use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_ecs::Component;

use ecs_std::components::*;

// ---------------------------------------------------------------------------
// Component names
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

// ---------------------------------------------------------------------------
// Pod byte serialization (only for types that remain Pod)
// ---------------------------------------------------------------------------

#[test]
fn pod_components_serialization() {
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
    let v = Visibility::VISIBLE;
    assert_eq!(v.component_name(), "Visibility");

    let n = Name::new("test");
    assert_eq!(n.component_name(), "Name");

    let mut world = redlilium_ecs::World::new();
    ecs_std::register_std_components(&mut world);
    let entity = world.spawn();
    let p = Parent(entity);
    assert_eq!(p.component_name(), "Parent");

    let c = Children::default();
    assert_eq!(c.component_name(), "Children");
}
