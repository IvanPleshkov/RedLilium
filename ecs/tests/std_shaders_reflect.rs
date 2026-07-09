//! The committed std shaders' binding contract, pinned through real slang
//! reflection: every std material shader declares its sets as rate-classified
//! `ParameterBlock`s (`[UpdateRate]`, docs/MATERIAL_ASSETS.md Decision 7) in
//! the canonical order camera (external) / model (dynamic) / material
//! (static) — which is exactly how `ForwardRender` assembles the bind groups.

// Needs runtime Slang reflection, so it is gated on `slang-shaders` (default off,
// no SDK required for a normal build). Run:
//   cargo test -p redlilium-ecs --features "rendering slang-shaders"
#![cfg(all(feature = "rendering", feature = "slang-shaders"))]

use redlilium_graphics::UpdateRate;
use redlilium_graphics::shader::{ShaderReflectInput, SlangCompiler};
use redlilium_graphics::{BindingType, ShaderLibrary, ShaderStage};

fn reflect(
    source: &str,
) -> (
    Vec<redlilium_graphics::BindingLayout>,
    Vec<Option<UpdateRate>>,
) {
    let compiler = SlangCompiler::new().expect("slang compiler");
    compiler
        .write_library_modules(&ShaderLibrary::standard_slang())
        .expect("library modules");
    let shaders: Vec<ShaderReflectInput<'_>> = vec![
        (source, "vs_main", ShaderStage::Vertex, &[]),
        (source, "fs_main", ShaderStage::Fragment, &[]),
    ];
    compiler
        .reflect_all_bindings(&shaders)
        .expect("std shader reflects")
}

#[test]
fn opaque_color_declares_rate_classified_sets() {
    let source = std::fs::read_to_string("../std-assets/shaders/opaque_color.slang").unwrap();
    let (layouts, rates) = reflect(&source);
    assert_eq!(
        rates,
        vec![
            Some(UpdateRate::External),
            Some(UpdateRate::Dynamic),
            Some(UpdateRate::Static),
        ]
    );
    // Each set: the block's implicit uniform buffer at binding 0.
    for layout in &layouts {
        assert_eq!(layout.entries[0].binding, 0);
        assert_eq!(layout.entries[0].binding_type, BindingType::UniformBuffer);
    }
}

#[test]
fn opaque_textured_material_set_matches_instance_group_convention() {
    let source = std::fs::read_to_string("../std-assets/shaders/opaque_textured.slang").unwrap();
    let (layouts, rates) = reflect(&source);
    assert_eq!(
        rates,
        vec![
            Some(UpdateRate::External),
            Some(UpdateRate::Dynamic),
            Some(UpdateRate::Static),
        ]
    );
    // The static material set matches the instance manager's group layout:
    // packed props buffer at 0, then texture/sampler pairs in schema order.
    let material = &layouts[2].entries;
    assert_eq!(material.len(), 3, "{material:?}");
    assert_eq!(material[0].binding_type, BindingType::UniformBuffer);
    assert_eq!(material[1].binding, 1);
    assert_eq!(material[1].binding_type, BindingType::Texture);
    assert_eq!(material[2].binding, 2);
    assert_eq!(material[2].binding_type, BindingType::Sampler);
}
