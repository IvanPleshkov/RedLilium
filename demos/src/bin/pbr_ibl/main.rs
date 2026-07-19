//! PBR/IBL demo — a thin consumer of the standard deferred render path
//! (#144, milestone 9).
//!
//! Everything graphics-related comes from the engine: the runtime host wires
//! the Render schedule, and the built-in `deferred` pipeline — selected per
//! camera via [`RenderPath`] — records the G-buffer, skybox, and IBL resolve
//! passes. The demo only supplies content: a free-fly camera and a sphere
//! grid whose materials sweep metallic (rows) × roughness (columns) through
//! programmatic `pbr` material instances (#113).
//!
//! Controls: right-drag to orbit, scroll to zoom (the std free-fly camera).

use redlilium_assets::Guid;
use redlilium_core::math::Vec3;
use redlilium_ecs::rendering::{MaterialInstanceData, PropValue};
use redlilium_ecs::{
    Camera, DEFERRED_PIPELINE, FreeFlyCamera, GlobalTransform, MaterialInstanceManager,
    MaterialInstanceSource, MeshGenerator, MeshRenderer, MeshSource, Primitive, RenderPath,
    Transform, Update, UpdateFreeFlyCamera, Visibility,
};
use redlilium_runtime::{App, GameConfig, Plugin};

use std::f32::consts::FRAC_PI_4;

/// Rows sweep metallic 0→1, columns sweep roughness 0→1.
const GRID: usize = 6;
const SPACING: f32 = 1.3;

struct PbrIblDemo;

impl Plugin for PbrIblDemo {
    fn build(&self, app: &mut App) {
        app.add_system::<Update, _>(UpdateFreeFlyCamera);
    }

    fn spawn_scene(&self, app: &mut App) {
        let aspect = app.initial_aspect();
        let world = app.world_mut();

        // Free-fly camera on the deferred PBR/IBL path.
        let camera = world.spawn();
        world
            .insert(camera, Camera::perspective(FRAC_PI_4, aspect, 0.1, 500.0))
            .unwrap();
        world
            .insert(camera, RenderPath::named(DEFERRED_PIPELINE))
            .unwrap();
        let free_fly = FreeFlyCamera::new(Vec3::new(0.0, 0.0, 0.0), 11.0);
        let transform = free_fly.to_transform();
        world.insert(camera, free_fly).unwrap();
        world.insert(camera, transform).unwrap();
        world
            .insert(camera, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(camera, Visibility::VISIBLE).unwrap();

        // The sphere grid: one programmatic pbr material instance per sphere
        // (#113 publish path — parented on the std pbr material asset).
        let parent = Guid::stable("materials/pbr.material");
        let offset = (GRID as f32 - 1.0) * SPACING * 0.5;
        for row in 0..GRID {
            for col in 0..GRID {
                let metallic = row as f32 / (GRID as f32 - 1.0);
                let roughness = col as f32 / (GRID as f32 - 1.0);
                let guid = Guid::stable(&format!("pbr_ibl_demo/sphere_{row}_{col}"));
                world
                    .resource_mut::<MaterialInstanceManager>()
                    .publish_virtual(
                        guid,
                        MaterialInstanceData {
                            parent,
                            overrides: vec![
                                (
                                    "base_color".to_owned(),
                                    PropValue::Vec4([0.75, 0.12, 0.1, 1.0]),
                                ),
                                (
                                    "pbr_params".to_owned(),
                                    PropValue::Vec4([metallic, roughness, 0.0, 0.0]),
                                ),
                            ],
                        },
                    );

                let entity = world.spawn();
                let transform = Transform::from_translation(Vec3::new(
                    col as f32 * SPACING - offset,
                    row as f32 * SPACING - offset,
                    0.0,
                ));
                world.insert(entity, transform).unwrap();
                world
                    .insert(entity, GlobalTransform(transform.to_matrix()))
                    .unwrap();
                world.insert(entity, Visibility::VISIBLE).unwrap();
                world
                    .insert(
                        entity,
                        MeshRenderer::single(Primitive::new(
                            MeshSource::Generated(MeshGenerator::sphere(0.5, 48, 24)),
                            MaterialInstanceSource { guid },
                        )),
                    )
                    .unwrap();
            }
        }
    }
}

fn main() {
    redlilium_runtime::run(
        GameConfig {
            title: "RedLilium PBR IBL Demo".to_string(),
            mounts: vec![("std", "std-assets")],
            ..GameConfig::default()
        },
        PbrIblDemo,
    );
}
