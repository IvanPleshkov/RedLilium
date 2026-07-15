//! # Car Game
//!
//! The arcade-car vertical slice (#103, milestone "Vertical Slice — Arcade
//! Car"): game code authored as a [`redlilium_runtime::Plugin`], built both as
//! a standalone binary (`cargo run -p car-game`) and as a cdylib the editor's
//! `GameHost` can load and warm-restart-reload (#45).
//!
//! The scene is a placeholder until #104/#105: a ground slab, a box standing
//! in for the car, and a free-fly camera. The dev HUD exercises the in-game
//! UI layer (#100): an egui window drawn from a game system through
//! [`GameUi`], with a Quit button driving [`AppControl`].
#![recursion_limit = "256"]

use std::f32::consts::FRAC_PI_4;

use redlilium_assets::Guid;
use redlilium_core::math::{Quat, Vec3};
use redlilium_ecs::{
    Camera, FreeFlyCamera, GlobalTransform, MaterialInstanceSource, MeshGenerator, MeshRenderer,
    MeshSource, PostUpdate, Primitive, System, SystemContext, SystemError, Time, Transform, Update,
    UpdateFreeFlyCamera, UpdateGlobalTransforms, Visibility, World,
};
use redlilium_runtime::{App, AppControl, GameUi, Plugin};

/// Dev HUD (#100 smoke surface): frame time plus a Quit button. Drawn through
/// the [`GameUi`] resource, which only the standalone runtime host provides —
/// hosted in the editor this system is a no-op.
pub struct DevHud;

impl System for DevHud {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<GameUi>() {
            return Ok(());
        }
        let delta = world.resource::<Time>().delta();
        let egui_ctx = world.resource::<GameUi>().ctx().clone();
        redlilium_graphics::egui::egui::Window::new("Car Game — dev")
            .default_pos([10.0, 10.0])
            .resizable(false)
            .show(&egui_ctx, |ui| {
                ui.label(format!("frame: {:.2} ms", delta * 1000.0));
                // A browser tab has no process to exit (#100).
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Quit").clicked() {
                    world.resource_mut::<AppControl>().request_exit();
                }
            });
        Ok(())
    }
}

/// The car game plugin: `build` registers systems, `spawn_scene` populates the
/// placeholder scene (skipped on reload — the scene comes from a snapshot).
pub struct CarGamePlugin;

impl Plugin for CarGamePlugin {
    fn build(&self, app: &mut App) {
        log::info!("CarGamePlugin::build");
        app.add_system::<Update, _>(DevHud);
        // Viewport navigation until the follow camera lands (#104). The
        // editor's schedules already carry this engine system — only the
        // standalone runtime needs it added here.
        let post = app.schedule_mut::<PostUpdate>();
        if !post.contains::<UpdateFreeFlyCamera>() {
            post.add(UpdateFreeFlyCamera);
            post.add_edge::<UpdateFreeFlyCamera, UpdateGlobalTransforms>()
                .expect("no cycle");
        }
    }

    fn spawn_scene(&self, app: &mut App) {
        let aspect = app.initial_aspect();
        let world = app.world_mut();

        // Camera: behind and above the future car, free-fly for inspection.
        let camera = world.spawn();
        let free_fly = FreeFlyCamera::new(Vec3::new(0.0, 0.8, 0.0), 8.0)
            .with_yaw(std::f32::consts::PI)
            .with_pitch(0.35);
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

        let material = MaterialInstanceSource {
            guid: Guid::stable("materials/default.matinst"),
        };

        // Ground slab.
        spawn_box(
            world,
            material.clone(),
            Transform::new(
                Vec3::new(0.0, -0.1, 0.0),
                Quat::identity(),
                Vec3::new(40.0, 0.2, 40.0),
            ),
        );

        // Placeholder car: a box at the origin until #104 gives it physics.
        spawn_box(
            world,
            material,
            Transform::new(
                Vec3::new(0.0, 0.3, 0.0),
                Quat::identity(),
                Vec3::new(1.0, 0.6, 2.0),
            ),
        );
    }
}

fn spawn_box(world: &mut World, material: MaterialInstanceSource, transform: Transform) {
    let entity = world.spawn();
    world.insert(entity, transform).unwrap();
    world
        .insert(entity, GlobalTransform(transform.to_matrix()))
        .unwrap();
    world.insert(entity, Visibility::VISIBLE).unwrap();
    world
        .insert(
            entity,
            MeshRenderer::single(Primitive::new(
                MeshSource::Generated(MeshGenerator::cube(0.5)),
                material,
            )),
        )
        .unwrap();
}

// Export the ADR-020 game symbols so this cdylib is loadable by the editor's
// GameHost (#45). Same caveat as `redlilium-demos`: the macro expands
// `#[no_mangle]` symbols into a crate built as BOTH cdylib and rlib — sound
// while only this package's own bin links the rlib, but a host binary linking
// two such rlibs would fail with duplicate symbols.
redlilium_runtime::redlilium_game_module!(CarGamePlugin);
