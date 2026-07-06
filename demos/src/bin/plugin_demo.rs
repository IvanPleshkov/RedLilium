//! # Plugin Demo
//!
//! Game code authored as a `redlilium-runtime` [`Plugin`] (ADR-020, #44/#45):
//! `build` registers a component and adds systems; `spawn_scene` populates the
//! initial scene. The split lets warm-restart reload re-run registration while
//! restoring the scene from a snapshot. The runtime owns the window, the world,
//! the frame loop, and the render bracket.
//!
//! Controls: right-drag to look, WASD to fly (std `FreeFlyCamera`).
#![recursion_limit = "256"]

use std::f32::consts::{FRAC_PI_4, TAU};

use redlilium_assets::Guid;
use redlilium_core::math::{Quat, Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Camera, Component, FreeFlyCamera, GlobalTransform, MaterialInstanceSource, MeshGenerator,
    MeshRenderer, MeshSource, PostUpdate, Primitive, Res, System, SystemContext, Time, Transform,
    Update, UpdateFreeFlyCamera, UpdateGlobalTransforms, Visibility, World, WriteAll,
};
use redlilium_runtime::{App, GameConfig, Plugin};

/// Rotates the entity around +Y. A plain game component: registered and
/// serialized like any std component, driven by a game system.
#[derive(Clone, Component)]
struct Spin {
    /// Angular velocity, radians per second.
    speed: f32,
    /// Accumulated angle, radians.
    angle: f32,
}

struct SpinSystem;

impl System for SpinSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), redlilium_ecs::SystemError> {
        ctx.lock::<(Res<Time>, WriteAll<Spin>, WriteAll<Transform>)>()
            .execute(|(time, mut spins, mut transforms)| {
                let dt = time.delta() as f32;
                for (idx, mut spin) in spins.iter_mut() {
                    spin.angle += spin.speed * dt;
                    if let Some(mut transform) = transforms.get_mut(idx) {
                        transform.rotation = quat_from_rotation_y(spin.angle);
                    }
                }
            });
        Ok(())
    }
}

struct SpinDemo;

impl Plugin for SpinDemo {
    fn build(&self, app: &mut App) {
        app.register_component::<Spin>();
        app.add_system::<Update, _>(SpinSystem);
        // Viewport navigation: the std free-fly camera, ordered before the
        // transform propagation the runtime installed.
        app.add_system::<PostUpdate, _>(UpdateFreeFlyCamera);
        app.schedule_mut::<PostUpdate>()
            .add_edge::<UpdateFreeFlyCamera, UpdateGlobalTransforms>()
            .expect("no cycle");
    }

    fn spawn_scene(&self, app: &mut App) {
        let aspect = app.initial_aspect();
        let world = app.world_mut();

        // --- Camera ---
        let camera = world.spawn();
        let free_fly = FreeFlyCamera::new(Vec3::new(0.0, 0.5, 0.0), 6.0)
            .with_yaw(0.6)
            .with_pitch(0.3);
        let transform = free_fly.to_transform();
        world
            .insert(camera, Camera::perspective(FRAC_PI_4, aspect, 0.1, 500.0))
            .unwrap();
        world.insert(camera, free_fly).unwrap();
        world.insert(camera, transform).unwrap();
        world
            .insert(camera, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(camera, Visibility::VISIBLE).unwrap();

        // --- Scene ---
        // The std `default` material instance, bound by its stable guid
        // (resolved from the std mount's assets.db).
        let material = MaterialInstanceSource {
            guid: Guid::stable("materials/default.matinst"),
        };

        // Ground plane (scaled flat cube).
        spawn_mesh(
            world,
            MeshSource::Generated(MeshGenerator::cube(0.5)),
            material.clone(),
            Transform::new(
                Vec3::new(0.0, -0.05, 0.0),
                Quat::identity(),
                Vec3::new(10.0, 0.1, 10.0),
            ),
            None,
        );

        // A ring of spinning cubes.
        const CUBES: usize = 6;
        for i in 0..CUBES {
            let angle = i as f32 / CUBES as f32 * TAU;
            spawn_mesh(
                world,
                MeshSource::Generated(MeshGenerator::cube(0.5)),
                material.clone(),
                Transform::from_translation(Vec3::new(angle.cos() * 2.5, 0.5, angle.sin() * 2.5)),
                Some(Spin {
                    speed: 0.4 + i as f32 * 0.25,
                    angle,
                }),
            );
        }

        // A sphere in the middle.
        spawn_mesh(
            world,
            MeshSource::Generated(MeshGenerator::sphere(0.7, 32, 16)),
            material,
            Transform::from_translation(Vec3::new(0.0, 0.9, 0.0)),
            None,
        );
    }
}

fn spawn_mesh(
    world: &mut World,
    mesh: MeshSource,
    material: MaterialInstanceSource,
    transform: Transform,
    spin: Option<Spin>,
) {
    let entity = world.spawn();
    world.insert(entity, transform).unwrap();
    world
        .insert(entity, GlobalTransform(transform.to_matrix()))
        .unwrap();
    world.insert(entity, Visibility::VISIBLE).unwrap();
    world
        .insert(entity, MeshRenderer::single(Primitive::new(mesh, material)))
        .unwrap();
    if let Some(spin) = spin {
        world.insert(entity, spin).unwrap();
    }
}

fn main() {
    redlilium_runtime::run(
        GameConfig {
            title: "RedLilium Plugin Demo".to_string(),
            ..Default::default()
        },
        SpinDemo,
    );
}
