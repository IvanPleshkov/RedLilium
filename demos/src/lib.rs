//! # RedLilium Engine Demos
//!
//! Demo scenes showcasing RedLilium Engine capabilities.
//!
//! ## Available Demos
//!
//! - `window_demo` - Basic window creation with winit
//! - `plugin_demo` - Game code as a [`redlilium_runtime::Plugin`] (ADR-020)
//!
//! ## Loadable game module
//!
//! This crate is also built as a `cdylib` and exports the two ADR-020 game
//! symbols via [`redlilium_runtime::redlilium_game_module!`] over [`SpinDemo`],
//! so the editor can load `libredlilium_demos.dylib` as a game module and
//! warm-restart-reload it (#45).
#![recursion_limit = "256"]

pub mod resources;
pub mod systems;

use std::f32::consts::{FRAC_PI_4, TAU};

use redlilium_assets::Guid;
use redlilium_core::math::{Quat, Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Camera, Component, FreeFlyCamera, GlobalTransform, MaterialInstanceSource, MeshGenerator,
    MeshRenderer, MeshSource, PostUpdate, Primitive, Res, System, SystemContext, Time, Transform,
    Update, UpdateFreeFlyCamera, UpdateGlobalTransforms, Visibility, World, WriteAll,
};
use redlilium_runtime::{App, Plugin};

/// Demos library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Rotates the entity around +Y. A plain game component: registered and
/// serialized like any std component, driven by a game system.
#[derive(Clone, Component)]
pub struct Spin {
    /// Angular velocity, radians per second.
    pub speed: f32,
    /// Accumulated angle, radians.
    pub angle: f32,
}

/// Advances every [`Spin`] and writes the rotation into its `Transform`.
pub struct SpinSystem;

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

/// Panics exactly once, on its first run — raised inside the game cdylib's
/// image. Enabled by `REDLILIUM_DEMO_PANIC=1`; the reload harness uses it to
/// verify cross-image panic containment (#57). Subsequent frames run clean.
#[derive(Default)]
pub struct PanicOnce(std::sync::atomic::AtomicBool);

impl System for PanicOnce {
    type Result = ();
    fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), redlilium_ecs::SystemError> {
        if !self.0.swap(true, std::sync::atomic::Ordering::SeqCst) {
            panic!("deliberate game-module panic (#57 containment verification)");
        }
        Ok(())
    }
}

/// The demo game plugin: a ring of spinning cubes, a ground plane, a sphere,
/// and a free-fly camera. `build` registers types/systems; `spawn_scene`
/// populates the initial scene (skipped on reload — the scene comes from a
/// snapshot).
pub struct SpinDemo;

impl Plugin for SpinDemo {
    fn build(&self, app: &mut App) {
        // Routed through the host's logger via the load-time handoff (#56);
        // without it this line would vanish into the cdylib's own
        // never-initialized `log` static. The reload harness asserts it.
        log::info!("SpinDemo::build (game log via host logger handoff)");
        app.register_component::<Spin>();
        app.add_system::<Update, _>(SpinSystem);
        // #57 verification hook: an env-gated system that panics (once)
        // *inside this cdylib's image*, so the reload harness can prove the
        // host's per-system catch_unwind contains a cross-image panic.
        if std::env::var("REDLILIUM_DEMO_PANIC").is_ok() {
            app.add_system::<Update, _>(PanicOnce::default());
        }
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

/// Browser entry point (#33): boot the runtime with the [`SpinDemo`] plugin.
/// `wasm-bindgen(start)` invokes this when the wasm module initialises; it kicks
/// off async device creation and hands control to the winit `spawn_app` loop
/// (never blocking the browser thread). Native binaries use the `window_demo`
/// bin instead.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    redlilium_runtime::run(redlilium_runtime::GameConfig::default(), SpinDemo);
}

// Export the two ADR-020 game symbols so this cdylib can be loaded and
// warm-restart-reloaded by the editor (#45).
//
// NOTE (constraint): this expands to `#[no_mangle]` `redlilium_game_module` /
// `redlilium_abi_fingerprint` in a crate built as BOTH cdylib and rlib. It is
// sound today — only `plugin_demo` links the rlib, so those symbols merely ride
// along in that bin harmlessly, and the dlopened cdylib exports the one pair the
// loader resolves. But a host binary that ever links two macro-expanding crates
// (e.g. the editor plus a second game rlib) would fail with duplicate symbols.
// When the editor grows real game loading (#45 slice C3), the game should live
// in a dedicated cdylib-only crate (or behind a `game-module` feature) so no
// rlib consumer silently becomes "a game module".
redlilium_runtime::redlilium_game_module!(SpinDemo);
