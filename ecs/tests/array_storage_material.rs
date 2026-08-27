//! End-to-end proof that a `Texture2DArray` and a `StructuredBuffer` bind as
//! material properties (docs/MATERIAL_ASSETS.md) — the two kinds a plain
//! scalar/vector/texture schema couldn't express. Builds the demo
//! `array_storage_demo` pipeline from the baked shader (the default Slang-off
//! path) and binds real resources of exactly those kinds into its static
//! material set, the way the instance manager does at runtime. Before the
//! feature, the reflected layout mismatched (a D2Array reflected as a plain
//! texture) or had no slot for the buffer, so this bind failed.
#![cfg(feature = "rendering")]

use std::sync::Arc;

use redlilium_assets::Guid;
use redlilium_core::mesh::VertexLayout;
use redlilium_ecs::PipelineCache;
use redlilium_ecs::rendering::loaders::Shader;
use redlilium_graphics::{
    BindingGroupDescriptor, BufferDescriptor, BufferUsage, GraphicsInstance, SamplerDescriptor,
    TextureDescriptor, TextureFormat, TextureUsage, UpdateRate, VariantKey,
};

const SHADER: &str = include_str!("../../std-assets/shaders/array_storage_demo.slang");

#[test]
fn array_and_storage_buffer_bind_to_the_material_set() {
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("device");

    // Build the demo pipeline from the baked shader (baked WGSL keyed on the
    // normalized source — same route the runtime takes on a Slang-off build).
    let mut cache = PipelineCache::new(device.clone());
    let shader = Arc::new(Shader {
        source: SHADER.as_bytes().to_vec(),
    });
    let layout = VertexLayout::position_normal_uv();
    let material = cache
        .get_or_build(
            Guid::stable("shaders/array_storage_demo.slang"),
            &shader,
            &VariantKey::default(),
            &layout,
            &[TextureFormat::Rgba8UnormSrgb],
            TextureFormat::Depth32Float,
        )
        .expect("demo pipeline builds on the baked path");

    // The static material set — where the instance manager binds props.
    let static_idx = material
        .set_update_rates()
        .iter()
        .position(|r| *r == Some(UpdateRate::Static))
        .expect("demo shader declares a static material set");
    let set_layout = material.binding_layouts()[static_idx].clone();

    // Real resources of exactly the kinds the two newer property types resolve
    // to: the packed uniform (binding 0), a D2Array texture + sampler
    // (bindings 1–2), and a read-only storage buffer (binding 3).
    let uniform = device
        .create_buffer(&BufferDescriptor::new(
            16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .unwrap();
    let array_tex = device
        .create_texture(&TextureDescriptor::new_2d_array(
            1,
            1,
            1,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        ))
        .unwrap();
    let sampler = device
        .create_sampler(&SamplerDescriptor::default())
        .unwrap();
    let storage = device
        .create_buffer(&BufferDescriptor::new(
            16,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        ))
        .unwrap();

    let descriptor = BindingGroupDescriptor::new()
        .with_buffer(0, uniform)
        .with_texture(1, array_tex)
        .with_sampler(2, sampler)
        .with_buffer(3, storage);

    // The crux: the reflected material-set layout accepts a D2Array texture and
    // a storage buffer bound as material properties.
    device
        .create_binding_group(set_layout, descriptor)
        .expect("array texture + storage buffer bind to the material set");
}
