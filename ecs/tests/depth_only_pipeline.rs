//! The depth-only pipeline specialization (#129) on the default (baked,
//! Slang-off) build: `PipelineCache::get_or_build_depth_only` must produce a
//! vertex-only, zero-color material from the embedded std depth shader — the
//! exact path a shadow/depth-prepass pass takes at runtime — with its
//! camera/model sets rate-classified through the baked reflection table.
#![cfg(feature = "rendering")]

use redlilium_core::mesh::VertexLayout;
use redlilium_ecs::PipelineCache;
use redlilium_ecs::rendering::shaders::{depth_only_guid, depth_only_shader};
use redlilium_graphics::{GraphicsInstance, TextureFormat, UpdateRate};

#[test]
fn depth_only_pipeline_builds_from_baked_shader() {
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("device");

    let mut cache = PipelineCache::new(device);
    let layout = VertexLayout::position_normal();
    let shader = depth_only_shader();

    let material = cache
        .get_or_build_depth_only(
            depth_only_guid(),
            &shader,
            &layout,
            TextureFormat::Depth32Float,
        )
        .expect("depth-only pipeline builds on the baked path");

    assert!(
        material.color_formats().is_empty(),
        "depth-only pipeline must declare zero color formats"
    );
    assert!(material.depth().is_some(), "depth state present");
    assert_eq!(
        material.set_update_rates(),
        &[Some(UpdateRate::External), Some(UpdateRate::Dynamic)],
        "camera/model sets classified through baked reflection"
    );

    // Same key → cache hit, not a second compile.
    let again = cache
        .get_or_build_depth_only(
            depth_only_guid(),
            &shader,
            &layout,
            TextureFormat::Depth32Float,
        )
        .expect("cache hit");
    assert_eq!(cache.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&material, &again));
}
