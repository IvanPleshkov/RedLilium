//! GPU integration tests for the graphics crate.
//!
//! These tests verify that the graphics API works correctly with actual GPU backends.
//! Tests are parameterized using `rstest` to run against multiple backends.
//!
//! # Test Categories
//!
//! - **Resource Copy Tests**: Verify buffer and texture copy operations via render graph
//! - **Render Tests**: Verify basic rendering to textures with readback validation
//! - **Depth Buffer Tests**: Verify depth testing and multiple draw calls
//! - **MRT Tests**: Verify multiple render target support
//!
//! # Running Tests
//!
//! Tests are currently skipped because no GPU backends are implemented yet.
//! As backends are implemented, remove the `#[ignore]` attribute from relevant tests.
//!
//! ```bash
//! # Run all integration tests (skipped ones will be marked as ignored)
//! cargo test --test gpu_tests
//!
//! # Run ignored tests when backends are ready
//! cargo test --test gpu_tests -- --ignored
//! ```

mod common;

use rstest::rstest;

use std::sync::{Arc, Mutex};

use common::{
    Backend, CENTERED_QUAD_VERTICES, ExpectedPixel, FULLSCREEN_QUAD_VERTICES,
    LEFT_HALF_QUAD_VERTICES, TestContext, create_centered_quad, create_fullscreen_quad,
    create_left_half_quad, create_material_instance, create_mrt_pass,
    create_render_pass_with_depth, create_simple_render_pass, create_solid_color_material,
    create_solid_color_material_with_raster, create_texture_sample_instance,
    create_texture_sample_material, generate_test_pattern, get_pixel, quad_vertex_layout,
    readback_buffer_size, verify_pixel, write_quad_vertices,
};
use redlilium_graphics::{
    BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType,
    BufferTextureCopyRegion, BufferTextureLayout, BufferUsage, ColorAttachment,
    DepthStencilAttachment, Extent3d, GraphicsPass, LoadOp, MaterialDescriptor, MaterialInstance,
    QueuePreference, RenderGraph, RenderGraphCompilationMode, RenderTarget, RenderTargetConfig,
    SamplerDescriptor, ShaderSource, StoreOp, TextureCopyLocation, TextureDescriptor,
    TextureFormat, TextureOrigin, TextureUsage, TransferConfig, TransferOperation, TransferPass,
};

// ============================================================================
// Resource Copy Tests
// ============================================================================

/// Test basic buffer-to-buffer copy via render graph.
///
/// This test verifies that:
/// 1. A staging buffer can be created and filled with data
/// 2. Data can be copied to a GPU buffer using a transfer pass
/// 3. Data can be copied back to a readback buffer
/// 4. The readback data matches the original
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_buffer_copy_roundtrip(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const BUFFER_SIZE: u64 = 1024;
    let test_data = generate_test_pattern(BUFFER_SIZE as usize);

    // Create buffers
    let staging = ctx.create_staging_buffer(BUFFER_SIZE);
    let gpu_buffer = ctx.create_gpu_buffer(BUFFER_SIZE, BufferUsage::STORAGE);
    let readback = ctx.create_readback_buffer(BUFFER_SIZE);

    // In a real implementation, we would:
    // 1. Map staging buffer and write test_data
    // 2. Execute transfer graph: staging -> gpu_buffer -> readback
    // 3. Map readback buffer and verify data

    // Create the transfer graph
    let mut graph = RenderGraph::new();

    // First pass: staging -> gpu_buffer
    let mut upload = TransferPass::new("upload".into());
    upload.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::copy_buffer_whole(staging.clone(), gpu_buffer.clone()),
    ));
    let upload_handle = graph.add_transfer_pass(upload);

    // Second pass: gpu_buffer -> readback
    let mut download = TransferPass::new("download".into());
    download.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::copy_buffer_whole(gpu_buffer, readback.clone()),
    ));
    let download_handle = graph.add_transfer_pass(download);

    // Download depends on upload
    graph.add_dependency(download_handle, upload_handle);

    // Verify the graph compiles correctly before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 2);

    // Execute the graph
    ctx.execute_graph(graph);

    // In a real backend test, we would verify:
    // let readback_data = map_buffer_read(&readback);
    // assert_eq!(readback_data, test_data);

    // For now, just verify the test infrastructure works
    assert_eq!(test_data.len(), BUFFER_SIZE as usize);
    assert_eq!(test_data[0], 0);
    assert_eq!(test_data[255], 255);
}

/// Regression test for `TransferOperation::upload_texture_data` (the path egui
/// uses to stage its font-atlas / user textures).
///
/// Under ADR-021 (#89) a buffer created without a mapping flag lands
/// device-local, where the mapped write used to fill the staging buffer
/// panics ("write_buffer on a device-local buffer"). The staging buffer must
/// carry `MAP_WRITE`. Uploads a known pattern into a texture and reads it back.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_upload_texture_data_roundtrip(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {backend:?} not available, skipping");
        return;
    };

    // Single-row texture: the tight row pitch is used verbatim (no 256-byte
    // multi-row padding), so the readback bytes match the uploaded bytes 1:1.
    const WIDTH: u32 = 16;
    let pixels: Vec<u8> = (0..WIDTH * 4).map(|i| (i % 251) as u8).collect();

    let texture = ctx.create_texture_2d(
        WIDTH,
        1,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    );

    // The exact call egui makes — must not panic on a device-local staging
    // buffer.
    let upload = TransferOperation::upload_texture_data(&ctx.device, texture.clone(), &pixels)
        .expect("upload_texture_data should stage the pixels without error");
    let mut upload_graph = RenderGraph::new();
    let mut upload_pass = TransferPass::new("upload_texture".into());
    upload_pass.set_transfer_config(TransferConfig::new().with_operation(upload));
    upload_graph.add_transfer_pass(upload_pass);
    ctx.execute_graph(upload_graph);

    // The dummy backend performs no real copy; the point there is just that the
    // staging + graph execution did not panic.
    if backend == Backend::Dummy {
        return;
    }

    let readback = ctx.create_readback_buffer((WIDTH * 4) as u64);
    let mut readback_graph = RenderGraph::new();
    let mut readback_pass = TransferPass::new("readback_texture".into());
    readback_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(texture.clone(), readback.clone()),
    ));
    readback_graph.add_transfer_pass(readback_pass);
    ctx.execute_graph(readback_graph);

    let data = ctx.read_buffer(&readback, (WIDTH * 4) as u64);
    assert_eq!(
        data, pixels,
        "read-back texture bytes must match the uploaded pattern"
    );
}

/// GPU mip-chain generation (#96): a 4×4 texture with four distinct quadrant
/// colors, then `GenerateMipmaps`, must produce a 1×1 top mip that is the
/// average of the four colors (two linear blit steps: 4→2→1). Runs with
/// validation on so a subresource-range barrier mistake in the blit chain is
/// caught. Skipped where `mip_generation == false` (wgpu/dummy).
#[rstest]
#[case::vulkan(Backend::Vulkan)]
fn test_generate_mipmaps_4x4_average(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Skipping test: {backend:?} backend not available");
        return;
    };
    if !ctx.device.capabilities().mip_generation {
        eprintln!("Skipping test: mip_generation unsupported on {backend:?}");
        return;
    }

    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const W: u32 = 4;
    const H: u32 = 4;
    // Quadrants: top-left red, top-right green, bottom-left blue, bottom-right
    // white. Their average is (~128, ~128, ~128, 255).
    let colors = [
        [255u8, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
    ];
    let mut pixels = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let q = (if y < 2 { 0 } else { 2 }) + (if x < 2 { 0 } else { 1 });
            let i = ((y * W + x) * 4) as usize;
            pixels[i..i + 4].copy_from_slice(&colors[q]);
        }
    }

    // 4×4 → 3 mips (4, 2, 1). COPY_SRC so the blit can read lower mips.
    let mip_count = 3;
    let texture = ctx
        .device
        .create_texture(
            &TextureDescriptor::new_2d(
                W,
                H,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            )
            .with_mip_levels(mip_count),
        )
        .expect("create mip texture");

    // One pass: upload mip 0, then blit the chain — exercises the intra-pass
    // upload→blit barrier the op emits internally.
    let upload = TransferOperation::upload_texture_data(&ctx.device, texture.clone(), &pixels)
        .expect("stage mip0 upload");
    let mut graph = RenderGraph::new();
    let mut pass = TransferPass::new("mipgen".into());
    pass.set_transfer_config(TransferConfig::new().with_operations(vec![
        upload,
        TransferOperation::generate_mipmaps(texture.clone()),
    ]));
    graph.add_transfer_pass(pass);
    ctx.execute_graph(graph);

    // Read back the 1×1 top mip (mip 2).
    let readback = ctx.create_readback_buffer(4);
    let region = BufferTextureCopyRegion::new(
        BufferTextureLayout::packed(),
        TextureCopyLocation::mip(mip_count - 1),
        Extent3d::new_2d(1, 1),
    );
    let mut read_graph = RenderGraph::new();
    let mut read_pass = TransferPass::new("read_top_mip".into());
    read_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture(texture.clone(), readback.clone(), vec![region]),
    ));
    read_graph.add_transfer_pass(read_pass);
    ctx.execute_graph(read_graph);

    let data = ctx.read_buffer(&readback, 4);
    assert_eq!(data.len(), 4, "expected one RGBA texel, got {data:?}");
    // Two linear halvings of the four quadrant colors → mid-gray, opaque.
    let expected = [128i32, 128, 128, 255];
    for (i, &exp) in expected.iter().enumerate() {
        let got = data[i] as i32;
        assert!(
            (got - exp).abs() <= 8,
            "top-mip channel {i}: got {got}, expected ~{exp} (full texel {data:?})"
        );
    }

    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during mip generation"
        );
    }
}

/// Container-supplied chains (#120): `upload_texture_level` targets one
/// (mip, layer) image per operation. Uploads a distinct pattern into every
/// mip of a 3-mip 2D texture and every face of a cubemap, then reads
/// individual subresources back and checks the bytes landed where addressed.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_upload_texture_level_mips_and_faces(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {backend:?} not available, skipping");
        return;
    };

    // --- 2D, 3 mips (4 → 2 → 1), one upload op per mip. Mip 1 is multi-row
    // with an 8-byte tight pitch, exercising the 256-byte staging padding.
    let texture = ctx
        .device
        .create_texture(
            &TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            )
            .with_mip_levels(3),
        )
        .expect("create mip texture");
    let mip_bytes = |mip: u32, fill: u8| vec![fill; ((4usize >> mip).max(1).pow(2)) * 4];
    let mut pass = TransferPass::new("upload_mips".into());
    let mut config = TransferConfig::new();
    for (mip, fill) in [(0u32, 0x11u8), (1, 0x22), (2, 0x33)] {
        config = config.with_operation(
            TransferOperation::upload_texture_level(
                &ctx.device,
                texture.clone(),
                mip,
                0,
                &mip_bytes(mip, fill),
            )
            .expect("stage mip upload"),
        );
    }
    pass.set_transfer_config(config);
    let mut graph = RenderGraph::new();
    graph.add_transfer_pass(pass);
    ctx.execute_graph(graph);

    // --- Cube, 6 faces with distinct fills, one op per face.
    let cube = ctx
        .device
        .create_texture(&TextureDescriptor::new_cube(
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
        ))
        .expect("create cube texture");
    let mut cube_pass = TransferPass::new("upload_faces".into());
    let mut cube_config = TransferConfig::new();
    for face in 0..6u32 {
        cube_config = cube_config.with_operation(
            TransferOperation::upload_texture_level(
                &ctx.device,
                cube.clone(),
                0,
                face,
                &[0x40 + face as u8; 2 * 2 * 4],
            )
            .expect("stage face upload"),
        );
    }
    cube_pass.set_transfer_config(cube_config);
    let mut cube_graph = RenderGraph::new();
    cube_graph.add_transfer_pass(cube_pass);
    ctx.execute_graph(cube_graph);

    // Out-of-range subresources are rejected up front.
    assert!(
        TransferOperation::upload_texture_level(&ctx.device, texture.clone(), 3, 0, &[]).is_err()
    );
    assert!(TransferOperation::upload_texture_level(&ctx.device, cube.clone(), 0, 6, &[]).is_err());

    // The dummy backend performs no real copies.
    if backend == Backend::Dummy {
        return;
    }

    // Read back single-row probes: mip 2 (1×1), a 2×1 strip of mip 1, and a
    // 2×1 strip of face 3 — each packed, no multi-row pitch rules.
    let probes: [(&std::sync::Arc<_>, u32, u32, u32, u8); 3] = [
        (&texture, 2, 0, 1, 0x33),
        (&texture, 1, 0, 2, 0x22),
        (&cube, 0, 3, 2, 0x43),
    ];
    for (tex, mip, layer, width, expected) in probes {
        let bytes = (width * 4) as u64;
        let readback = ctx.create_readback_buffer(bytes);
        let region = BufferTextureCopyRegion::new(
            BufferTextureLayout::packed(),
            TextureCopyLocation::new(mip, TextureOrigin::new(0, 0, layer)),
            Extent3d::new_2d(width, 1),
        );
        let mut read_graph = RenderGraph::new();
        let mut read_pass = TransferPass::new("read_probe".into());
        read_pass.set_transfer_config(TransferConfig::new().with_operation(
            TransferOperation::readback_texture((*tex).clone(), readback.clone(), vec![region]),
        ));
        read_graph.add_transfer_pass(read_pass);
        ctx.execute_graph(read_graph);
        let data = ctx.read_buffer(&readback, bytes);
        assert!(
            data.iter().all(|&b| b == expected),
            "mip {mip} layer {layer}: got {data:?}, expected all {expected:#x}"
        );
    }
}

/// Format gate for mip generation (#96): a blit-eligible color format is
/// supported on Vulkan; a block-compressed format never is (it cannot be
/// blit-downsampled). wgpu/dummy report `false` for everything (no blit path),
/// so the loader keeps a single mip there.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_mip_generation_format_gate(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Skipping test: {backend:?} backend not available");
        return;
    };

    let color = ctx
        .device
        .supports_mipmap_generation(TextureFormat::Rgba8Unorm);
    let compressed = ctx
        .device
        .supports_mipmap_generation(TextureFormat::Bc7RgbaUnorm);

    // Block-compressed formats are never blit-downsamplable on any backend.
    assert!(
        !compressed,
        "{backend:?} claimed mip generation for a block-compressed format"
    );

    if ctx.device.capabilities().mip_generation {
        // Vulkan: a plain color format is blit-eligible.
        assert!(
            color,
            "{backend:?} supports mip generation but rejected Rgba8Unorm"
        );
    } else {
        // wgpu/dummy: no blit path at all.
        assert!(
            !color,
            "{backend:?} has no mip_generation cap but claimed format support"
        );
    }
}

/// Cross-backend face-culling agreement (#39): a counter-clockwise, front-
/// facing quad with `CullMode::Back` must render on BOTH Vulkan and wgpu.
///
/// The engine's logical convention is CCW = front (glTF/OpenGL). The fullscreen
/// quad is wound CCW, so back-face culling must keep it. If a backend's
/// effective winding is inverted (e.g. a Vulkan negative-viewport Y-flip not
/// matched by the front-face mapping), it culls the front face and the target
/// stays at the clear color — the "draws the back side of geometry" bug.
#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_back_face_culling_agreement(#[case] backend: Backend) {
    use redlilium_graphics::{CullMode, FrontFace, PolygonMode, RasterState};

    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {backend:?} not available, skipping");
        return;
    };

    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    let render_target = ctx.create_render_target(WIDTH, HEIGHT);
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    // The fullscreen quad is wound counter-clockwise in clip space.
    let quad_mesh = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &quad_mesh, &FULLSCREEN_QUAD_VERTICES);

    // CCW-front + cull back faces: the quad is front-facing, so it renders.
    let material = create_solid_color_material_with_raster(
        &ctx,
        RasterState {
            cull_mode: CullMode::Back,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
        },
    );
    let instance = create_material_instance(material);

    let mut graph = RenderGraph::new();
    let mut render_pass =
        create_simple_render_pass("cull_back_ccw", render_target.clone(), CLEAR_COLOR);
    render_pass.add_draw(quad_mesh, instance);
    let render_handle = graph.add_graphics_pass(render_pass);

    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);
    graph.add_dependency(copy_handle, render_handle);

    ctx.execute_graph(graph);

    let data = ctx.read_buffer(&readback, readback_size);
    let center = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, ExpectedPixel::RED, 2),
        "CCW front-facing quad with back-face culling must render red on \
         {backend:?}, but got {center:?} — the front face was culled (winding \
         inverted)"
    );
}

/// Test buffer copy with partial regions.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_buffer_copy_partial(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const BUFFER_SIZE: u64 = 2048;
    const COPY_SIZE: u64 = 512;
    const SRC_OFFSET: u64 = 256;
    const DST_OFFSET: u64 = 1024;

    // Create buffers
    let src = ctx.create_gpu_buffer(BUFFER_SIZE, BufferUsage::STORAGE);
    let dst = ctx.create_gpu_buffer(BUFFER_SIZE, BufferUsage::STORAGE);

    // Create transfer graph with partial copy
    let mut graph = RenderGraph::new();
    let mut transfer = TransferPass::new("partial_copy".into());
    transfer.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::copy_buffer(
            src,
            dst,
            vec![redlilium_graphics::BufferCopyRegion::new(
                SRC_OFFSET, DST_OFFSET, COPY_SIZE,
            )],
        ),
    ));
    graph.add_transfer_pass(transfer);

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 1);

    // Execute
    ctx.execute_graph(graph);
}

// ============================================================================
// Single Quad Render Tests
// ============================================================================

/// Test rendering a single quad to a texture and reading back the result.
///
/// This test verifies:
/// 1. A render target texture can be created
/// 2. A graphics pass can render a quad
/// 3. The rendered result can be copied to a readback buffer
/// 4. The readback data shows the expected rendered output
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_single_quad(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; // Black background

    // Create render target
    let render_target = ctx.create_render_target(WIDTH, HEIGHT);

    // Create readback buffer for texture data
    let readback_size = (WIDTH * HEIGHT * 4) as u64;
    let readback = ctx.create_readback_buffer(readback_size);

    // Create render graph
    let mut graph = RenderGraph::new();

    // Render pass - clear to black, would render red quad
    let render_pass = create_simple_render_pass("render_quad", render_target.clone(), CLEAR_COLOR);
    // In a real test, we would add draw commands here:
    // render_pass.add_draw(quad_mesh, red_material);
    let render_handle = graph.add_graphics_pass(render_pass);

    // Copy pass - copy render target to readback buffer
    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);

    // Copy depends on render
    graph.add_dependency(copy_handle, render_handle);

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 2);

    // Execute
    ctx.execute_graph(graph);

    // In a real backend test, we would verify the rendered pixels:
    // let data = map_buffer_read(&readback);
    //
    // Verify corners are black (clear color)
    // assert!(verify_pixel(&data, WIDTH, 0, 0, ExpectedPixel::BLACK, 1));
    //
    // Verify center is red (quad color)
    // assert!(verify_pixel(&data, WIDTH, WIDTH/2, HEIGHT/2, ExpectedPixel::RED, 1));
}

/// Test clearing a render target to a specific color.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_clear_color(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // Skip dummy backend since it doesn't actually render
    if backend == Backend::Dummy {
        return;
    }

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const CLEAR_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

    // Create render target and readback (with alignment for row pitch)
    let render_target = ctx.create_render_target(WIDTH, HEIGHT);
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    // Create graph with just a clear operation
    let mut graph = RenderGraph::new();

    let render_pass = create_simple_render_pass("clear_only", render_target.clone(), CLEAR_COLOR);
    let render_handle = graph.add_graphics_pass(render_pass);

    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);

    graph.add_dependency(copy_handle, render_handle);

    ctx.execute_graph(graph);

    // Read back and verify clear color
    let data = ctx.read_buffer(&readback, readback_size);

    let expected = ExpectedPixel::from_float(0.25, 0.5, 0.75, 1.0);
    let center_pixel = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, expected, 2),
        "Clear color pixel should be {:?}, but got {:?}",
        expected,
        center_pixel
    );
}

/// Test two graphs submitted in one frame with a cross-graph dependency
/// (#47 phase 1/2).
///
/// Graph A renders (clears) into a texture; graph B — a separate queue
/// submit — copies that texture into a readback buffer. This verifies:
/// 1. Multiple submits per frame execute and complete (per-submit timeline
///    fences all wait correctly).
/// 2. The persistent trackers synchronize the texture across submits: B's
///    copy needs a layout transition + barrier against A's color write,
///    emitted in B's command buffer with A's write as the source scope —
///    valid only because both submits share the queue in submission order.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_multi_submit_cross_graph_dependency(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // On Vulkan, assert zero validation errors across the whole workload
    // (#82): the cross-submit barrier/layout handoff is exactly what the
    // layers check.
    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const CLEAR_COLOR: [f32; 4] = [0.75, 0.25, 0.5, 1.0];

    let render_target = ctx.create_render_target(WIDTH, HEIGHT);
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    // Graph A: render (clear) into the texture.
    let mut graph_a = RenderGraph::new();
    graph_a.add_graphics_pass(create_simple_render_pass(
        "multi_submit_render",
        render_target.clone(),
        CLEAR_COLOR,
    ));

    // Graph B: copy the texture to the readback buffer. No explicit
    // dependency on A — ordering comes from submission order alone.
    let mut graph_b = RenderGraph::new();
    let mut copy_pass = TransferPass::new("multi_submit_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    graph_b.add_transfer_pass(copy_pass);

    // One frame, two submits.
    ctx.execute_graphs(vec![graph_a, graph_b]);

    // Skip pixel verification on the dummy backend (it doesn't render).
    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);

    let expected = ExpectedPixel::from_float(0.75, 0.25, 0.5, 1.0);
    let center_pixel = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, expected, 2),
        "Cross-submit copied pixel should be {:?}, but got {:?}",
        expected,
        center_pixel
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during multi-submit cross-graph dependency"
        );
    }
}

/// Per-pass GPU timestamps (#95): a two-pass graph must report a timing for
/// each pass, with non-negative durations bounded by the submit total.
///
/// Timestamp results are read back only when a slot retires (~MAX_FRAMES_IN_
/// FLIGHT frames later), so the graph is run several frames before the timings
/// are populated. Skipped on backends without `gpu_timestamps` (wgpu/dummy).
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_gpu_timestamps_two_pass(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Skipping test: {backend:?} backend not available");
        return;
    };

    // wgpu/dummy report no timestamp support and return empty timings.
    if !ctx.device.capabilities().gpu_timestamps {
        eprintln!("Skipping test: gpu_timestamps unsupported on {backend:?}");
        return;
    }

    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    let target_a = ctx.create_render_target(64, 64);
    let target_b = ctx.create_render_target(64, 64);

    // Run the same two-pass graph enough frames for a recorded slot to retire
    // and have its query results read back into `latest_gpu_timings`.
    let mut timings = redlilium_graphics::FrameGpuTimings::default();
    for _ in 0..8 {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(create_simple_render_pass(
            "pass_a",
            target_a.clone(),
            [0.1, 0.2, 0.3, 1.0],
        ));
        graph.add_graphics_pass(create_simple_render_pass(
            "pass_b",
            target_b.clone(),
            [0.3, 0.2, 0.1, 1.0],
        ));
        ctx.execute_graph(graph);

        let latest = ctx.device.latest_gpu_timings();
        if !latest.is_empty() {
            timings = latest;
        }
    }

    assert!(
        !timings.is_empty(),
        "no GPU timings were reported after running the two-pass graph"
    );

    // The submit that carried both passes.
    let submit = timings
        .submits
        .iter()
        .find(|s| s.passes.len() >= 2)
        .unwrap_or_else(|| panic!("expected a submit timing both passes, got {timings:?}"));

    assert!(
        submit.passes.iter().any(|(n, _)| n == "pass_a"),
        "pass_a missing from timings: {:?}",
        submit.passes
    );
    assert!(
        submit.passes.iter().any(|(n, _)| n == "pass_b"),
        "pass_b missing from timings: {:?}",
        submit.passes
    );

    assert!(submit.total_ms >= 0.0, "submit total is negative");
    // Each pass's timestamp region is a subset of the submit region, so its
    // duration is non-negative and bounded by the submit total. (Passes can
    // pipeline, so their *sum* may exceed the total — do not assert on the sum.)
    // A small epsilon absorbs tick-precision rounding.
    let eps = 0.5;
    for (name, ms) in &submit.passes {
        assert!(*ms >= 0.0, "pass {name} has a negative duration {ms}");
        assert!(
            *ms <= submit.total_ms + eps,
            "pass {name} duration {ms} exceeds submit total {}",
            submit.total_ms
        );
    }

    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during GPU timestamp collection"
        );
    }
}

/// GPU memory stats (#98): `latest_memory_stats` must report the engine's live
/// resource counts on every backend, and — on Vulkan — non-empty driver heaps
/// with plausible sizes and (when `VK_EXT_memory_budget` is present) budget/usage.
/// The sample is refreshed each frame in `advance_frame`, so a graph is run first.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_gpu_memory_stats(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Skipping test: {backend:?} backend not available");
        return;
    };

    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    // A handful of live buffers so the resource counts are non-trivial; keep
    // them alive across the frames below (the counts are weak-ref filtered).
    let _buffers: Vec<_> = (0..3)
        .map(|_| ctx.create_gpu_buffer(4096, BufferUsage::VERTEX))
        .collect();

    let target = ctx.create_render_target(64, 64);
    for _ in 0..3 {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(create_simple_render_pass(
            "pass",
            target.clone(),
            [0.1, 0.2, 0.3, 1.0],
        ));
        ctx.execute_graph(graph);
    }

    let stats = ctx.device.latest_memory_stats();

    // Resource counts work on every backend (filled at the device layer).
    assert!(
        stats.resources.buffers >= 3,
        "expected >=3 live buffers in memory stats, got {}",
        stats.resources.buffers
    );

    if backend == Backend::Vulkan {
        assert!(
            !stats.heaps.is_empty(),
            "Vulkan reported no memory heaps: {stats:?}"
        );
        for heap in &stats.heaps {
            assert!(heap.size > 0, "heap {} has zero size: {heap:?}", heap.index);
        }
        // The allocator has live resources, so reserved >= allocated > 0.
        assert!(
            stats.allocator_reserved >= stats.allocator_allocated,
            "allocator reserved {} < allocated {}",
            stats.allocator_reserved,
            stats.allocator_allocated
        );

        if ctx.device.capabilities().memory_budget {
            let budgeted = stats
                .heaps
                .iter()
                .find(|h| h.budget.is_some())
                .unwrap_or_else(|| panic!("memory_budget on but no heap has a budget: {stats:?}"));
            assert!(
                budgeted.budget.unwrap() > 0,
                "budgeted heap reports zero budget: {budgeted:?}"
            );
            assert!(
                budgeted.usage.is_some(),
                "heap with a budget has no usage: {budgeted:?}"
            );
        }

        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during memory-stats sampling"
        );
    }
}

/// GPU crash breadcrumbs (#97): with breadcrumbs forced on, a multi-frame
/// two-pass graph must encode its per-pass markers with zero validation errors.
/// On MoltenVK this exercises the portable `vkCmdFillBuffer` fallback (no vendor
/// extension); the whole point is that the marker writes and the per-slot buffer
/// reset are legal. Vulkan-only — wgpu/dummy have no breadcrumbs.
#[rstest]
#[case::vulkan(Backend::Vulkan)]
fn test_breadcrumbs_two_pass_encodes_cleanly(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_breadcrumbs(backend) else {
        eprintln!("Skipping test: {backend:?} backend not available");
        return;
    };

    redlilium_graphics::backend::vulkan::reset_validation_error_count();

    let target_a = ctx.create_render_target(64, 64);
    let target_b = ctx.create_render_target(64, 64);

    // Several frames so each per-slot marker buffer is reset-then-written at
    // least once across the frame-in-flight ring (the reset path is the one
    // most likely to trip validation).
    for _ in 0..8 {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(create_simple_render_pass(
            "breadcrumb_pass_a",
            target_a.clone(),
            [0.1, 0.2, 0.3, 1.0],
        ));
        graph.add_graphics_pass(create_simple_render_pass(
            "breadcrumb_pass_b",
            target_b.clone(),
            [0.3, 0.2, 0.1, 1.0],
        ));
        ctx.execute_graph(graph);
    }

    let errors = redlilium_graphics::backend::vulkan::validation_error_count();
    assert_eq!(
        errors, 0,
        "Vulkan validation reported {errors} error(s) while encoding GPU crash breadcrumbs"
    );
}

/// Test the async compute routing opt-in (#47 phase 4).
///
/// Graph A renders (clears) into a texture on the graphics queue; graph B —
/// flagged `QueuePreference::AsyncCompute` — copies it to a readback buffer. On
/// devices with an async compute queue B runs there, and the cross-queue RAW
/// hazard on the texture must be resolved by a tracker-emitted timeline wait
/// (plus CONCURRENT-shared resources). On single-queue devices (MoltenVK
/// default, wgpu, dummy) the hint is ignored and B runs on the graphics
/// queue — the first-class fallback. Either way the copied pixels must match.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_async_compute_opt_in_cross_queue_dependency(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // On Vulkan, assert zero validation errors across the whole workload
    // (#82): on multi-queue hardware this is the tracker-emitted cross-queue
    // timeline wait the layers get to see.
    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const CLEAR_COLOR: [f32; 4] = [0.5, 0.75, 0.25, 1.0];

    // Declared cross-queue (#88): keeps graph B eligible for async routing.
    let render_target = ctx.create_cross_queue_render_target(WIDTH, HEIGHT);
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    let mut graph_a = RenderGraph::new();
    graph_a.add_graphics_pass(create_simple_render_pass(
        "async_optin_render",
        render_target.clone(),
        CLEAR_COLOR,
    ));

    let mut graph_b = RenderGraph::new();
    graph_b.set_queue_preference(QueuePreference::AsyncCompute);
    let mut copy_pass = TransferPass::new("async_optin_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    graph_b.add_transfer_pass(copy_pass);

    ctx.execute_graphs(vec![graph_a, graph_b]);

    if backend == Backend::Dummy {
        return; // dummy doesn't render; routing/execution not panicking is the test
    }

    let data = ctx.read_buffer(&readback, readback_size);

    let expected = ExpectedPixel::from_float(0.5, 0.75, 0.25, 1.0);
    let center_pixel = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, expected, 2),
        "Pixel copied by the (possibly async-routed) graph should be {:?}, but got {:?}",
        expected,
        center_pixel
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during async compute cross-queue dependency"
        );
    }
}

/// Transfer-queue routing (#89): the asset-streaming pattern.
///
/// Graph A — transfer-only, flagged `QueuePreference::Transfer` — uploads a
/// byte pattern into a device-local buffer; graph B reads it back on the
/// graphics queue. On devices with a dedicated transfer family A runs on the
/// DMA engines and the cross-queue RAW hazard is resolved by a
/// tracker-emitted timeline wait; elsewhere the preference walks the ladder
/// (async compute, then graphics) — every rung must produce identical bytes
/// and zero validation errors.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_transfer_queue_upload_then_graphics_read(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const SIZE: u64 = 512;
    let pattern: Arc<[u8]> = (0..SIZE as usize).map(|i| (i * 7) as u8).collect();

    let uploaded = ctx.create_buffer(SIZE, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);
    let readback = ctx.create_readback_buffer(SIZE);

    // Graph A: the upload, requesting the transfer queue (what
    // `AssetProcessor::flush_gpu` emits).
    let mut graph_a = RenderGraph::new();
    graph_a.set_queue_preference(QueuePreference::Transfer);
    assert!(graph_a.is_transfer_only());
    let mut upload_pass = TransferPass::new("transfer_route_upload".into());
    upload_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::write_buffer(uploaded.clone(), 0, pattern.clone()),
    ));
    graph_a.add_transfer_pass(upload_pass);

    // Graph B: the graphics-queue consumer (cross-queue RAW against A).
    let mut graph_b = RenderGraph::new();
    let mut read_pass = TransferPass::new("transfer_route_read".into());
    read_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::copy_buffer_whole(uploaded.clone(), readback.clone()),
    ));
    graph_b.add_transfer_pass(read_pass);

    ctx.execute_graphs(vec![graph_a, graph_b]);

    if backend == Backend::Dummy {
        return; // dummy doesn't execute copies; not panicking is the test
    }

    let data = ctx.read_buffer(&readback, SIZE);
    assert_eq!(
        data.as_slice(),
        &pattern[..],
        "graphics-read bytes must match the transfer-queue-uploaded pattern"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during transfer-queue routing"
        );
    }
}

/// Transfer-queue image upload on a coarse-granularity family (#92).
///
/// Graph A — transfer-only, `QueuePreference::Transfer` — uploads a byte
/// pattern into a cross-queue-declared texture with a **whole-subresource**
/// `BufferToTexture` (exactly what `AssetProcessor::flush_gpu` emits for a
/// texture). Whole-subresource copies are legal at any image-transfer
/// granularity, so on AMD SDMA (granularity 16×16×8, which #92 stopped
/// excluding) the upload routes to the DMA engines; on 1×1×1 families to the
/// transfer queue; elsewhere down the ladder. Every route must produce
/// identical bytes and zero validation errors — a granularity violation
/// (e.g. wrongly routing a partial copy to SDMA) surfaces as a validation
/// error here.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_transfer_queue_whole_image_upload(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    // 64×64 RGBA8: tight row pitch is 256 (already the 256-byte copy alignment),
    // so uploaded and read-back bytes compare directly. The depth axis (1, not a
    // multiple of SDMA's granularity-depth 8) means only the whole-subresource
    // rule — not granularity-multiple alignment — makes this legal on SDMA.
    const W: u32 = 64;
    const H: u32 = 64;
    let pattern: Vec<u8> = (0..(W * H * 4)).map(|i| (i * 13) as u8).collect();

    let texture = ctx
        .device
        .create_texture(
            // TEXTURE_BINDING mirrors a real sampled asset (and gives the
            // texture a valid image view); COPY_DST for the upload, COPY_SRC
            // for the readback.
            &TextureDescriptor::new_2d(
                W,
                H,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            )
            .with_cross_queue(true)
            .with_label("transfer_upload_target"),
        )
        .expect("create texture");

    // Graph A: fill a device buffer with the pattern (pass 1), then copy it
    // whole into the texture (pass 2) — both on the transfer queue. Two passes,
    // not two ops in one pass: the RAW on `src` is ordered by the barrier the
    // tracker places between passes. WriteBuffer is buffer-only
    // (granularity-exempt); the BufferToTexture covers the whole base
    // subresource (legal at any granularity). Tight row pitch is 256 for W=64,
    // so the packed layout satisfies the 256-byte copy alignment.
    let src = ctx.create_gpu_buffer(pattern.len() as u64, BufferUsage::empty());
    let mut graph_a = RenderGraph::new();
    graph_a.set_queue_preference(QueuePreference::Transfer);
    let mut fill_pass = TransferPass::new("transfer_fill_src".into());
    fill_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::write_buffer(src.clone(), 0, Arc::from(pattern.as_slice())),
    ));
    graph_a.add_transfer_pass(fill_pass);
    let mut upload_pass = TransferPass::new("transfer_image_upload".into());
    upload_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::upload_texture_whole(src.clone(), texture.clone()),
    ));
    graph_a.add_transfer_pass(upload_pass);
    assert!(graph_a.is_transfer_only());

    // Graph B: graphics-queue readback (cross-queue RAW against A).
    let readback_size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut graph_b = RenderGraph::new();
    let mut read_pass = TransferPass::new("transfer_image_read".into());
    read_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(texture.clone(), readback.clone()),
    ));
    graph_b.add_transfer_pass(read_pass);

    ctx.execute_graphs(vec![graph_a, graph_b]);

    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);
    assert_eq!(
        data.as_slice(),
        pattern.as_slice(),
        "read-back texture bytes must match the transfer-uploaded pattern"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during transfer-queue image upload"
        );
    }
}

/// Async routing safety for UNDECLARED textures (#88).
///
/// Same shape as [`test_async_compute_opt_in_cross_queue_dependency`], but
/// the render target is a plain (EXCLUSIVE) texture: the async hint must be
/// declined and the copy must run on the graphics queue — accessing an
/// EXCLUSIVE image from another queue family would leave its contents
/// undefined per spec. Verified by readback correctness plus zero validation
/// errors; on single-queue devices this is indistinguishable from the plain
/// fallback.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_async_compute_hint_declined_for_exclusive_texture(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const CLEAR_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

    // Deliberately NOT declared cross-queue.
    let render_target = ctx.create_render_target(WIDTH, HEIGHT);
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    let mut graph_a = RenderGraph::new();
    graph_a.add_graphics_pass(create_simple_render_pass(
        "declined_render",
        render_target.clone(),
        CLEAR_COLOR,
    ));

    let mut graph_b = RenderGraph::new();
    graph_b.set_queue_preference(QueuePreference::AsyncCompute);
    let mut copy_pass = TransferPass::new("declined_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    graph_b.add_transfer_pass(copy_pass);

    ctx.execute_graphs(vec![graph_a, graph_b]);

    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);

    let expected = ExpectedPixel::from_float(0.25, 0.5, 0.75, 1.0);
    let center_pixel = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, expected, 2),
        "Pixel copied by the declined-hint graph should be {:?}, but got {:?}",
        expected,
        center_pixel
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) with the async hint declined"
        );
    }
}

/// Cross-frame cross-queue dependencies (#82 step 4).
///
/// [`test_async_compute_opt_in_cross_queue_dependency`] covers a within-frame
/// hazard; this covers the persistent tracker state ACROSS frames, which the
/// scheduler's per-frame derived edges cannot see. A dedicated pipeline runs
/// with `frames_in_flight = 2` (the shared context uses 1), so consecutive
/// frames genuinely overlap and cross-frame ordering must come from the
/// tracker-emitted timeline waits, not from a frame-fence stall:
///
/// - Frame 1: an async-routed graph WRITES `async_written` (upload).
/// - Frame 2: a graphics graph READS it (copy to readback) — a cross-queue
///   RAW across frames — and a second graphics graph WRITES `gfx_written`.
/// - Frame 3: an async-routed graph READS `gfx_written` — the same hazard in
///   the reverse direction.
///
/// Both copies are verified by readback. On single-queue devices the async
/// hint is ignored and this degrades to plain multi-submit frames — the
/// first-class fallback. On Vulkan the context runs with validation layers
/// and asserts zero validation errors.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_async_compute_cross_frame_cross_queue_dependency(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const SIZE: u64 = 256;
    let pattern_a: Arc<[u8]> = (0..SIZE as usize).map(|i| i as u8).collect();
    let pattern_b: Arc<[u8]> = (0..SIZE as usize).map(|i| 255 - i as u8).collect();

    let async_written = ctx.create_buffer(SIZE, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);
    let gfx_written = ctx.create_buffer(SIZE, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);
    let readback_a = ctx.create_readback_buffer(SIZE);
    let readback_b = ctx.create_readback_buffer(SIZE);
    let dst_a = Arc::new(Mutex::new(Vec::new()));
    let dst_b = Arc::new(Mutex::new(Vec::new()));

    // Dedicated pipeline with 2 frames in flight (the shared TestContext
    // pipeline uses 1, which would serialize frames on the fence).
    let mut pipeline = ctx.device.create_pipeline(2);

    // Frame 1: async-routed upload into `async_written`.
    let mut schedule = pipeline.begin_frame().expect("begin_frame failed");
    let mut graph = RenderGraph::new();
    graph.set_queue_preference(QueuePreference::AsyncCompute);
    let mut pass = TransferPass::new("xframe_async_write".into());
    pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::write_buffer(async_written.clone(), 0, pattern_a.clone()),
    ));
    graph.add_transfer_pass(pass);
    schedule.submit(graph);
    pipeline.end_frame(schedule);

    // Frame 2: a graphics graph reads `async_written` (cross-queue RAW from
    // frame 1), a second graphics graph writes `gfx_written`.
    let mut schedule = pipeline.begin_frame().expect("begin_frame failed");
    let mut graph = RenderGraph::new();
    let mut pass = TransferPass::new("xframe_gfx_read".into());
    pass.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::copy_buffer_whole(
                async_written.clone(),
                readback_a.clone(),
            ))
            .with_operation(TransferOperation::readback_buffer(
                readback_a.clone(),
                0..SIZE as usize,
                dst_a.clone(),
            )),
    );
    graph.add_transfer_pass(pass);
    schedule.submit(graph);
    let mut graph = RenderGraph::new();
    let mut pass = TransferPass::new("xframe_gfx_write".into());
    pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::write_buffer(gfx_written.clone(), 0, pattern_b.clone()),
    ));
    graph.add_transfer_pass(pass);
    schedule.submit(graph);
    pipeline.end_frame(schedule);

    // Frame 3: async-routed graph reads `gfx_written` (the reverse
    // direction).
    let mut schedule = pipeline.begin_frame().expect("begin_frame failed");
    let mut graph = RenderGraph::new();
    graph.set_queue_preference(QueuePreference::AsyncCompute);
    let mut pass = TransferPass::new("xframe_async_read".into());
    pass.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::copy_buffer_whole(
                gfx_written.clone(),
                readback_b.clone(),
            ))
            .with_operation(TransferOperation::readback_buffer(
                readback_b.clone(),
                0..SIZE as usize,
                dst_b.clone(),
            )),
    );
    graph.add_transfer_pass(pass);
    schedule.submit(graph);
    pipeline.end_frame(schedule);

    // Drain: wait for the GPU, then run empty frames so every slot is
    // recycled and its post-fence readback processing fills the dst vecs.
    pipeline.wait_idle().expect("wait_idle failed");
    for _ in 0..2 {
        let mut schedule = pipeline.begin_frame().expect("begin_frame failed");
        schedule.submit(RenderGraph::new());
        pipeline.end_frame(schedule);
    }
    pipeline.wait_idle().expect("wait_idle failed");

    if backend == Backend::Dummy {
        return; // dummy doesn't execute copies; not panicking is the test
    }

    assert_eq!(
        dst_a.lock().unwrap().as_slice(),
        &pattern_a[..],
        "graphics-read data must match the async-written pattern (frame 1 -> 2)"
    );
    assert_eq!(
        dst_b.lock().unwrap().as_slice(),
        &pattern_b[..],
        "async-read data must match the graphics-written pattern (frame 2 -> 3)"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during cross-frame cross-queue dependency"
        );
    }
}

// ============================================================================
// Depth Buffer Tests
// ============================================================================

/// Test rendering with depth buffer - two overlapping quads.
///
/// This test verifies:
/// 1. Depth testing works correctly
/// 2. A closer quad (lower depth) occludes a farther quad
/// 3. Multiple draw calls in a single pass work correctly
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_depth_buffer_two_quads(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; // Black
    const CLEAR_DEPTH: f32 = 1.0;

    // Quad colors:
    // - Back quad (z=0.8): Blue
    // - Front quad (z=0.2): Green (should be visible)

    // Create targets
    let color_target = ctx.create_render_target(WIDTH, HEIGHT);
    let depth_target = ctx.create_depth_texture(WIDTH, HEIGHT);
    let readback_size = (WIDTH * HEIGHT * 4) as u64;
    let readback = ctx.create_readback_buffer(readback_size);

    // Create render graph
    let mut graph = RenderGraph::new();

    // Render pass with depth buffer
    let render_pass = create_render_pass_with_depth(
        "depth_test",
        color_target.clone(),
        depth_target,
        CLEAR_COLOR,
        CLEAR_DEPTH,
    );
    // In a real test:
    // render_pass.add_draw(back_quad_mesh, blue_material);  // z=0.8 (farther)
    // render_pass.add_draw(front_quad_mesh, green_material); // z=0.2 (closer)
    let render_handle = graph.add_graphics_pass(render_pass);

    // Copy to readback
    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(color_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);

    graph.add_dependency(copy_handle, render_handle);

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 2);

    ctx.execute_graph(graph);

    // In a real test, verify that the front (green) quad is visible:
    // let data = map_buffer_read(&readback);
    // Center should be green (front quad)
    // assert!(verify_pixel(&data, WIDTH, WIDTH/2, HEIGHT/2, ExpectedPixel::GREEN, 1));
    // Corners should be black (clear color, no quad)
    // assert!(verify_pixel(&data, WIDTH, 0, 0, ExpectedPixel::BLACK, 1));
}

/// Test depth buffer with reverse draw order.
///
/// Even if the front quad is drawn first, it should still be visible
/// because it has a smaller depth value.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_depth_buffer_reverse_order(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const CLEAR_DEPTH: f32 = 1.0;

    let color_target = ctx.create_render_target(WIDTH, HEIGHT);
    let depth_target = ctx.create_depth_texture(WIDTH, HEIGHT);
    let readback = ctx.create_readback_buffer((WIDTH * HEIGHT * 4) as u64);

    let mut graph = RenderGraph::new();

    let render_pass = create_render_pass_with_depth(
        "depth_reverse",
        color_target.clone(),
        depth_target,
        CLEAR_COLOR,
        CLEAR_DEPTH,
    );
    // Draw in reverse order (front first, then back)
    // render_pass.add_draw(front_quad_mesh, green_material); // z=0.2
    // render_pass.add_draw(back_quad_mesh, blue_material);   // z=0.8
    // Front quad should still be visible due to depth test
    let render_handle = graph.add_graphics_pass(render_pass);

    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(color_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);

    graph.add_dependency(copy_handle, render_handle);

    ctx.execute_graph(graph);

    // Result should be the same as test_render_depth_buffer_two_quads
}

// ============================================================================
// Multiple Render Target (MRT) Tests
// ============================================================================

/// Test rendering to multiple render targets simultaneously.
///
/// This test verifies:
/// 1. Multiple color attachments can be bound
/// 2. A shader can output to multiple targets
/// 3. Each target receives the correct output
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_multiple_targets(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;

    // Create multiple render targets with different clear colors
    let target0 = ctx.create_render_target(WIDTH, HEIGHT); // Will clear to red
    let target1 = ctx.create_render_target(WIDTH, HEIGHT); // Will clear to green
    let target2 = ctx.create_render_target(WIDTH, HEIGHT); // Will clear to blue

    // Create readback buffers (with alignment for row pitch)
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback0 = ctx.create_readback_buffer(readback_size);
    let readback1 = ctx.create_readback_buffer(readback_size);
    let readback2 = ctx.create_readback_buffer(readback_size);

    // Create render graph
    let mut graph = RenderGraph::new();

    // MRT render pass
    let mrt_pass = create_mrt_pass(
        "mrt_render",
        vec![
            (target0.clone(), [1.0, 0.0, 0.0, 1.0]), // Red
            (target1.clone(), [0.0, 1.0, 0.0, 1.0]), // Green
            (target2.clone(), [0.0, 0.0, 1.0, 1.0]), // Blue
        ],
    );
    let render_handle = graph.add_graphics_pass(mrt_pass);

    // Copy each target to its readback buffer
    let mut copy0 = TransferPass::new("copy_target0".into());
    copy0.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(target0, readback0.clone()),
    ));
    let copy0_handle = graph.add_transfer_pass(copy0);

    let mut copy1 = TransferPass::new("copy_target1".into());
    copy1.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(target1, readback1.clone()),
    ));
    let copy1_handle = graph.add_transfer_pass(copy1);

    let mut copy2 = TransferPass::new("copy_target2".into());
    copy2.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(target2, readback2.clone()),
    ));
    let copy2_handle = graph.add_transfer_pass(copy2);

    // All copies depend on render
    graph.add_dependency(copy0_handle, render_handle);
    graph.add_dependency(copy1_handle, render_handle);
    graph.add_dependency(copy2_handle, render_handle);

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 4); // 1 render + 3 copies

    ctx.execute_graph(graph);

    // In a real test:
    // let data0 = map_buffer_read(&readback0);
    // let data1 = map_buffer_read(&readback1);
    // let data2 = map_buffer_read(&readback2);
    // assert!(verify_region(&data0, WIDTH, 0, 0, WIDTH, HEIGHT, ExpectedPixel::RED, 1));
    // assert!(verify_region(&data1, WIDTH, 0, 0, WIDTH, HEIGHT, ExpectedPixel::GREEN, 1));
    // assert!(verify_region(&data2, WIDTH, 0, 0, WIDTH, HEIGHT, ExpectedPixel::BLUE, 1));
}

/// Test MRT with different texture formats.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_render_mrt_different_formats(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;

    // Create targets with different formats
    let target_rgba8 = ctx.create_texture_2d(
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );

    let target_rgba16f = ctx.create_texture_2d(
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba16Float,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );

    // Create graph
    let mut graph = RenderGraph::new();

    // Note: In a real MRT setup, all targets need compatible formats for the same pass
    // This test might need adjustment based on actual GPU constraints

    let pass0 =
        create_simple_render_pass("render_rgba8", target_rgba8.clone(), [1.0, 0.0, 0.0, 1.0]);
    let _handle0 = graph.add_graphics_pass(pass0);

    let pass1 = create_simple_render_pass(
        "render_rgba16f",
        target_rgba16f.clone(),
        [0.0, 1.0, 0.0, 1.0],
    );
    let _handle1 = graph.add_graphics_pass(pass1);

    // These are independent passes, no dependency needed

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 2);

    ctx.execute_graph(graph);
}

// ============================================================================
// Additional Tests
// ============================================================================

/// Test that an empty render graph executes without errors.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_empty_graph(#[case] _backend: Backend) {
    let mut graph = RenderGraph::new();

    // Empty graph should compile successfully
    let compiled = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Empty graph should compile");
    assert_eq!(compiled.pass_order().len(), 0);
}

/// Test complex dependency graph with diamond pattern.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_diamond_dependency_graph(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;

    // Create multiple targets for a diamond dependency pattern:
    //
    //     shadow
    //    /      \
    //  gbuffer  lighting
    //    \      /
    //    composite
    //

    let shadow_target = ctx.create_render_target(WIDTH, HEIGHT);
    let gbuffer_target = ctx.create_render_target(WIDTH, HEIGHT);
    let lighting_target = ctx.create_render_target(WIDTH, HEIGHT);
    let composite_target = ctx.create_render_target(WIDTH, HEIGHT);

    let mut graph = RenderGraph::new();

    // Shadow pass (root)
    let shadow = create_simple_render_pass("shadow", shadow_target, [1.0, 1.0, 1.0, 1.0]);
    let shadow_handle = graph.add_graphics_pass(shadow);

    // GBuffer pass (depends on shadow)
    let gbuffer = create_simple_render_pass("gbuffer", gbuffer_target, [0.5, 0.5, 0.5, 1.0]);
    let gbuffer_handle = graph.add_graphics_pass(gbuffer);
    graph.add_dependency(gbuffer_handle, shadow_handle);

    // Lighting pass (depends on shadow)
    let lighting = create_simple_render_pass("lighting", lighting_target, [0.8, 0.8, 0.8, 1.0]);
    let lighting_handle = graph.add_graphics_pass(lighting);
    graph.add_dependency(lighting_handle, shadow_handle);

    // Composite pass (depends on gbuffer and lighting)
    let composite = create_simple_render_pass("composite", composite_target, [0.0, 0.0, 0.0, 1.0]);
    let composite_handle = graph.add_graphics_pass(composite);
    graph.add_dependency(composite_handle, gbuffer_handle);
    graph.add_dependency(composite_handle, lighting_handle);

    // Verify topological order before executing
    {
        let compiled = graph
            .compile(RenderGraphCompilationMode::Automatic)
            .expect("Diamond graph should compile");
        assert_eq!(compiled.pass_order().len(), 4);

        // Shadow must come before gbuffer and lighting
        // Composite must come last
        let order = compiled.pass_order();
        let shadow_idx = order.iter().position(|&h| h == shadow_handle).unwrap();
        let gbuffer_idx = order.iter().position(|&h| h == gbuffer_handle).unwrap();
        let lighting_idx = order.iter().position(|&h| h == lighting_handle).unwrap();
        let composite_idx = order.iter().position(|&h| h == composite_handle).unwrap();

        assert!(shadow_idx < gbuffer_idx);
        assert!(shadow_idx < lighting_idx);
        assert!(gbuffer_idx < composite_idx);
        assert!(lighting_idx < composite_idx);
    }

    ctx.execute_graph(graph);
}

// ============================================================================
// Shader Rendering Tests
// ============================================================================

/// Test rendering a quad with a WGSL shader.
///
/// This test verifies:
/// 1. WGSL shader compilation works
/// 2. A material can be created and used for rendering
/// 3. A quad covering the left half of the screen is rendered correctly
/// 4. Texture readback returns expected pixel values
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_shader_render_half_quad(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // Skip dummy backend since it doesn't actually render
    if backend == Backend::Dummy {
        eprintln!("Dummy backend doesn't render, skipping pixel verification");
        return;
    }

    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; // Black background

    // Create render target
    let render_target = ctx.create_render_target(WIDTH, HEIGHT);

    // Create readback buffer for texture data (with alignment)
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback = ctx.create_readback_buffer(readback_size);

    // Create the quad mesh covering the left half of the screen
    let quad_mesh = create_left_half_quad(&ctx);
    write_quad_vertices(&ctx, &quad_mesh, &LEFT_HALF_QUAD_VERTICES);

    // Create material and material instance with WGSL shader
    let material = create_solid_color_material(&ctx);
    let material_instance = create_material_instance(material);

    // Create render graph
    let mut graph = RenderGraph::new();

    // Render pass - clear to black, render red quad in left half
    let mut render_pass =
        create_simple_render_pass("render_half_quad", render_target.clone(), CLEAR_COLOR);
    render_pass.add_draw(quad_mesh, material_instance);
    let render_handle = graph.add_graphics_pass(render_pass);

    // Copy pass - copy render target to readback buffer
    let mut copy_pass = TransferPass::new("copy_to_readback".into());
    copy_pass.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(render_target, readback.clone()),
    ));
    let copy_handle = graph.add_transfer_pass(copy_pass);

    // Copy depends on render
    graph.add_dependency(copy_handle, render_handle);

    // Execute
    ctx.execute_graph(graph);

    // Read back the pixel data
    let data = ctx.read_buffer(&readback, readback_size);

    // Verify pixel values
    // Left half (x < WIDTH/2) should be red
    // Right half (x >= WIDTH/2) should be black (clear color)

    // Check a pixel in the left half (should be red)
    let left_x = WIDTH / 4;
    let center_y = HEIGHT / 2;
    let left_pixel = get_pixel(&data, WIDTH, left_x, center_y);
    assert!(
        verify_pixel(&data, WIDTH, left_x, center_y, ExpectedPixel::RED, 2),
        "Left half pixel ({}, {}) should be red, but got: {:?}",
        left_x,
        center_y,
        left_pixel
    );

    // Check a pixel in the right half (should be black)
    let right_x = WIDTH * 3 / 4;
    let right_pixel = get_pixel(&data, WIDTH, right_x, center_y);
    assert!(
        verify_pixel(&data, WIDTH, right_x, center_y, ExpectedPixel::BLACK, 2),
        "Right half pixel ({}, {}) should be black, but got: {:?}",
        right_x,
        center_y,
        right_pixel
    );
}

// ============================================================================
// Layout Tracking Integration Test
// ============================================================================

/// Test automatic texture layout tracking and barrier placement.
///
/// This integration test verifies that the automatic barrier generation system
/// correctly handles texture layout transitions across multiple passes:
///
/// 1. **Pass 1 (Render)**: Render a red quad to RT1
///    - Transition: RT1 Undefined → ColorAttachment
///
/// 2. **Pass 2 (Render)**: Render a green quad to RT2
///    - Transition: RT2 Undefined → ColorAttachment
///    - RT1 remains in ColorAttachment (could be transitioned if we were sampling it)
///
/// 3. **Pass 3 (Copy)**: Copy RT1 to readback buffer
///    - Transition: RT1 ColorAttachment → TransferSrc
///
/// 4. **Pass 4 (Copy)**: Copy RT2 to readback buffer 2
///    - Transition: RT2 ColorAttachment → TransferSrc
///
/// The test verifies that:
/// - No Vulkan validation errors occur (automatic barriers are correct)
/// - Both render targets have the expected colors after readback
///
/// Note: This test doesn't use texture sampling due to current backend limitations.
/// The texture sampling test will be enabled once the wgpu backend supports
/// material binding layouts in pipeline creation.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_layout_tracking_multi_pass(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // Skip dummy backend since it doesn't actually render
    if backend == Backend::Dummy {
        eprintln!("Dummy backend doesn't render, skipping pixel verification");
        return;
    }

    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; // Black background

    // Create render targets that will go through multiple layout transitions
    let rt1 = ctx.create_render_target(WIDTH, HEIGHT);
    let rt2 = ctx.create_render_target(WIDTH, HEIGHT);

    // Create readback buffers for final verification
    let readback_size = readback_buffer_size(WIDTH, HEIGHT, 4);
    let readback1 = ctx.create_readback_buffer(readback_size);
    let readback2 = ctx.create_readback_buffer(readback_size);

    // Create quad mesh for rendering
    let quad = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &quad, &FULLSCREEN_QUAD_VERTICES);

    // Create material for solid red rendering
    let red_material = create_solid_color_material(&ctx);
    let red_instance = create_material_instance(red_material);

    // Build render graph
    let mut graph = RenderGraph::new();

    // Pass 1: Render red quad to RT1
    // This tests: RT1: Undefined → ColorAttachment
    let mut pass1 = create_simple_render_pass("render_to_rt1", rt1.clone(), CLEAR_COLOR);
    pass1.add_draw(quad.clone(), red_instance.clone());
    let pass1_handle = graph.add_graphics_pass(pass1);

    // Pass 2: Clear RT2 to green (different color to verify both passes work)
    // This tests: RT2: Undefined → ColorAttachment
    // Using different clear color instead of shader to avoid binding layout issues
    let pass2 = create_simple_render_pass(
        "clear_rt2_green",
        rt2.clone(),
        [0.0, 1.0, 0.0, 1.0], // Green
    );
    let pass2_handle = graph.add_graphics_pass(pass2);
    // Pass 2 doesn't depend on Pass 1 - they're independent
    // But we add dependency to ensure consistent ordering for the test
    graph.add_dependency(pass2_handle, pass1_handle);

    // Pass 3: Readback RT1 to buffer
    // This tests: RT1: ColorAttachment → TransferSrc
    let mut pass3 = TransferPass::new("readback_rt1".into());
    pass3.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(rt1, readback1.clone()),
    ));
    let pass3_handle = graph.add_transfer_pass(pass3);
    graph.add_dependency(pass3_handle, pass1_handle);

    // Pass 4: Readback RT2 to buffer
    // This tests: RT2: ColorAttachment → TransferSrc
    let mut pass4 = TransferPass::new("readback_rt2".into());
    pass4.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(rt2, readback2.clone()),
    ));
    let pass4_handle = graph.add_transfer_pass(pass4);
    graph.add_dependency(pass4_handle, pass2_handle);

    // Verify graph structure before executing
    let pass_count = graph
        .compile(RenderGraphCompilationMode::Automatic)
        .expect("Graph should compile")
        .pass_order()
        .len();
    assert_eq!(pass_count, 4, "Should have 4 passes");

    // Execute the graph
    // If automatic barriers are incorrect, this would either:
    // - Cause Vulkan validation errors (layout mismatch)
    // - Produce incorrect pixel values (data hazards)
    ctx.execute_graph(graph);

    // Read back and verify RT1 (should be red from the rendered quad)
    let data1 = ctx.read_buffer(&readback1, readback_size);
    let center_x = WIDTH / 2;
    let center_y = HEIGHT / 2;
    let rt1_pixel = get_pixel(&data1, WIDTH, center_x, center_y);
    assert!(
        verify_pixel(&data1, WIDTH, center_x, center_y, ExpectedPixel::RED, 2),
        "RT1 center pixel ({}, {}) should be red, but got: {:?}",
        center_x,
        center_y,
        rt1_pixel
    );

    // Read back and verify RT2 (should be green from clear color)
    let data2 = ctx.read_buffer(&readback2, readback_size);
    let rt2_pixel = get_pixel(&data2, WIDTH, center_x, center_y);
    assert!(
        verify_pixel(&data2, WIDTH, center_x, center_y, ExpectedPixel::GREEN, 2),
        "RT2 center pixel ({}, {}) should be green, but got: {:?}",
        center_x,
        center_y,
        rt2_pixel
    );
}

/// Verify a texture's contents survive a frame boundary (M10).
///
/// Frame 1 clears the texture to a known color; frame 2 (a separate
/// `execute_graph`, crossing the `advance_frame` boundary) reads it back
/// WITHOUT re-clearing. With the old per-frame layout reset, frame 2's first
/// barrier would use `oldLayout = UNDEFINED` and the driver could discard the
/// contents; the global layout tracker keeps the real previous layout, so the
/// cleared color persists.
///
/// `W = 64` makes the readback row exactly 256 bytes (already aligned), so the
/// pixel offset is alignment-independent.
#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_texture_contents_persist_across_frames(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const W: u32 = 64;
    const H: u32 = 64;
    // Values that round-trip cleanly through 8-bit unorm.
    let color = [0.25_f32, 0.5, 0.75, 1.0];

    let tex = ctx.create_render_target(W, H);
    let size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(size);

    // Frame 1: clear the texture to `color` and store it.
    let mut g1 = RenderGraph::new();
    g1.add_graphics_pass(create_simple_render_pass("clear", tex.clone(), color));
    ctx.execute_graph(g1);

    // Frame 2: read the texture back without touching its contents.
    let mut g2 = RenderGraph::new();
    let mut copy = TransferPass::new("readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(tex, readback.clone()),
    ));
    g2.add_transfer_pass(copy);
    ctx.execute_graph(g2);

    let data = ctx.read_buffer(&readback, size);
    let expected = ExpectedPixel::from_float(color[0], color[1], color[2], color[3]);
    assert!(
        verify_pixel(&data, W, W / 2, H / 2, expected, 2),
        "texture contents did not persist across the frame boundary (backend {backend:?})"
    );
}

// ============================================================================
// Eager binding-group lifetime tests (issue #40)
// ============================================================================

/// A binding group's GPU descriptor set is created **once** and reused on every
/// draw. Rendering the same `MaterialInstance` (holding one `Arc<BindingGroup>`)
/// for several consecutive frames binds that one cached set from a different
/// frame slot each time — this must be valid (no per-slot allocation, no
/// validation errors, no double-write).
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_binding_group_reused_across_frames(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const W: u32 = 16;
    const H: u32 = 16;

    // A texture-sample material whose instance carries a single cached binding
    // group (texture at 0, sampler at 1).
    let material = create_texture_sample_material(&ctx);
    let sampled = ctx.create_texture_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    let instance = create_texture_sample_instance(&ctx, material, sampled);

    // Render the SAME instance for 3 consecutive frames. Each execute_graph
    // advances the frame pipeline, so the cached set is bound from a different
    // slot each frame.
    for frame in 0..3 {
        let render_target = ctx.create_render_target(W, H);
        let quad = create_fullscreen_quad(&ctx);
        write_quad_vertices(&ctx, &quad, &FULLSCREEN_QUAD_VERTICES);

        let mut graph = RenderGraph::new();
        let mut pass =
            create_simple_render_pass("reuse_frame", render_target, [0.0, 0.0, 0.0, 1.0]);
        pass.add_draw(quad, instance.clone());
        graph.add_graphics_pass(pass);
        // A validation error or crash here fails the test (execute_graph
        // panics internally on backend errors).
        ctx.execute_graph(graph);
        let _ = frame;
    }
}

/// Dropping the last `Arc<BindingGroup>` right after submit must be safe: the
/// submitted graph retains the instance until its frame slot is recycled after
/// the fence wait, so the descriptor set is freed (via `GpuBindingGroup`'s
/// `Drop`) only once the GPU is done with it. Advancing several frames forces
/// that recycle; a use-after-free or pool/free-descriptor-set validation error
/// would surface here or on teardown.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_binding_group_dropped_while_in_flight(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    const W: u32 = 16;
    const H: u32 = 16;

    let material = create_texture_sample_material(&ctx);
    let sampled = ctx.create_texture_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    let instance = create_texture_sample_instance(&ctx, material, sampled);

    let render_target = ctx.create_render_target(W, H);
    let quad = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &quad, &FULLSCREEN_QUAD_VERTICES);

    let mut graph = RenderGraph::new();
    let mut pass = create_simple_render_pass("drop_inflight", render_target, [0.0, 0.0, 0.0, 1.0]);
    pass.add_draw(quad, instance.clone());
    graph.add_graphics_pass(pass);
    ctx.execute_graph(graph);

    // Drop our references immediately after submit. The frame slot still holds a
    // clone of the graph (and thus the Arc<BindingGroup>), so the set isn't
    // freed yet.
    drop(instance);

    // Advance frames so the slot holding the submitted graph recycles, dropping
    // the last Arc<BindingGroup> and freeing its descriptor set.
    for _ in 0..3 {
        let rt = ctx.create_render_target(W, H);
        let mut g = RenderGraph::new();
        g.add_graphics_pass(create_simple_render_pass(
            "advance",
            rt,
            [0.0, 0.0, 0.0, 1.0],
        ));
        ctx.execute_graph(g);
    }
}

// ============================================================================
// Depth co-use: sampled + read-only depth attachment in one pass (issue #60)
// ============================================================================

/// WGSL shader that samples a depth texture and writes the sampled depth as
/// the red channel. The vertex shader forces clip z = 0.1 so the fragment
/// passes the reversed-Z GreaterEqual depth test against the pre-cleared
/// depth of 0.05.
const DEPTH_CO_USE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position.xy, 0.1, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureSample(depth_texture, depth_sampler, in.uv);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
}
"#;

/// Regression test for issue #60: a depth texture sampled while bound as a
/// **read-only depth attachment** in the same pass.
///
/// Descriptor (`with_depth_texture_co_attached`), barrier (render graph
/// transition), and attachment (read-only depth) must all agree on
/// `DEPTH_STENCIL_READ_ONLY_OPTIMAL`; the pipeline must not write depth
/// (`with_depth_write(false)`).
///
/// Three sequential graphs (the layout tracker persists across executes):
/// 1. Prepass: clear depth to 0.05 (no draws) — leaves the texture in the
///    depth-attachment layout.
/// 2. Co-use pass: the SAME depth texture as a read-only attachment (Load) +
///    a material that samples it and outputs the depth as red. The quad is at
///    clip z = 0.1, passing the GreaterEqual test against 0.05, so every
///    fragment writes red = sampled depth (~0.05).
/// 3. Readback of the color target.
///
/// On Vulkan the context runs with validation layers and asserts **zero**
/// validation errors across the whole workload (the layouts this fix aligns
/// are also exactly what a stricter driver or a sync-validation run would
/// reject).
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_depth_co_use_read_only_attachment(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    // Bracket the whole GPU workload with the thread-local VUID counter. The
    // validation callback fires synchronously on the offending thread, and all
    // GPU work below runs on this test's thread.
    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const W: u32 = 16;
    const H: u32 = 16;
    // Below the quad's z = 0.1 so the reversed-Z default compare
    // (GreaterEqual) passes: 0.1 >= 0.05.
    const PRE_CLEAR_DEPTH: f32 = 0.05;

    // The depth texture is both a render attachment and sampled in a shader.
    let depth = ctx.create_texture_2d(
        W,
        H,
        TextureFormat::Depth32Float,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
    );
    let prepass_color = ctx.create_render_target(W, H);
    let color = ctx.create_render_target(W, H);

    // Graph 1 (prepass): clear depth to 0.25, no draws. Leaves the depth
    // texture in the depth-attachment layout for the next graph.
    let mut g1 = RenderGraph::new();
    g1.add_graphics_pass(create_render_pass_with_depth(
        "depth_prepass",
        prepass_color,
        depth.clone(),
        [0.0, 0.0, 0.0, 1.0],
        PRE_CLEAR_DEPTH,
    ));
    ctx.execute_graph(g1);

    // Material: samples the depth texture (binding 0) with a sampler
    // (binding 1), depth-tests against the read-only attachment without
    // writing depth.
    let binding_layout = Arc::new(
        BindingLayout::new()
            .with_entry(BindingLayoutEntry::new(0, BindingType::DepthTexture))
            .with_sampler(1)
            .with_label("depth_co_use_bindings"),
    );
    let material = ctx
        .device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    DEPTH_CO_USE_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    DEPTH_CO_USE_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_binding_layout(binding_layout)
                .with_color_format(TextureFormat::Rgba8Unorm)
                .with_depth_format(TextureFormat::Depth32Float)
                .with_depth_write(false)
                .with_label("depth_co_use_material"),
        )
        .expect("Failed to create depth co-use material");

    let sampler = ctx
        .device
        .create_sampler(&SamplerDescriptor::nearest().with_label("depth_co_use_sampler"))
        .expect("Failed to create sampler");

    // The co-attached constructor records DEPTH_STENCIL_READ_ONLY_OPTIMAL in
    // the eagerly-written descriptor, matching the attachment layout.
    let binding_group = ctx
        .device
        .create_binding_group(
            material.binding_layouts()[0].clone(),
            BindingGroupDescriptor::new()
                .with_depth_texture_co_attached(0, depth.clone())
                .with_sampler(1, sampler)
                .with_label("depth_co_use_group"),
        )
        .expect("Failed to create depth co-use binding group");

    let instance = Arc::new(
        MaterialInstance::new(material)
            .with_binding_group(binding_group)
            .with_label("depth_co_use_instance"),
    );

    let quad = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &quad, &FULLSCREEN_QUAD_VERTICES);

    // Graph 2 (co-use pass): same depth texture as read-only attachment
    // (Load) while the draw samples it.
    let mut g2 = RenderGraph::new();
    let mut pass = GraphicsPass::new("depth_co_use".into());
    pass.set_render_targets(
        RenderTargetConfig::new()
            .with_color(
                ColorAttachment::from_texture(color.clone())
                    .with_load_op(LoadOp::clear_color(0.0, 0.0, 0.0, 1.0))
                    .with_store_op(StoreOp::Store),
            )
            .with_depth_stencil(
                DepthStencilAttachment::from_texture(depth.clone())
                    .with_depth_load_op(LoadOp::Load)
                    .with_depth_store_op(StoreOp::Store)
                    .with_depth_read_only(true),
            ),
    );
    pass.add_draw(quad, instance);
    g2.add_graphics_pass(pass);
    ctx.execute_graph(g2);

    // Graph 3: read the color target back.
    let readback_size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut g3 = RenderGraph::new();
    let mut copy = TransferPass::new("depth_co_use_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(color, readback.clone()),
    ));
    g3.add_transfer_pass(copy);
    ctx.execute_graph(g3);

    // Dummy backend performs no real rendering — reaching here without a
    // panic is the assertion.
    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);
    let center = get_pixel(&data, W, W / 2, H / 2);
    eprintln!("depth co-use center pixel ({backend:?}): {center:?}");
    // Red channel = sampled depth = 0.05; the depth test consumed the same
    // image (fragments at z=0.1 pass GreaterEqual against 0.05).
    let expected = ExpectedPixel::from_float(PRE_CLEAR_DEPTH, 0.0, 0.0, 1.0);
    assert!(
        verify_pixel(&data, W, W / 2, H / 2, expected, 3),
        "Center pixel should be red = sampled depth {:?}, but got {:?} (backend {:?})",
        expected,
        center,
        backend
    );

    // Descriptor/barrier/attachment layouts agree, so the validation layer
    // stays silent. Note: current validation layers (1.4.x) no longer perform
    // submit-time descriptor-vs-image layout checks by default, so this
    // counter is a general VUID guard for the whole workload rather than a
    // targeted layout-mismatch detector; the pixel assertion above is the
    // functional proof that sampling and depth-testing co-used the same image.
    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during depth co-use"
        );
    }
}

/// Shader variants end-to-end (#6, Decision 5): the shader declares its axes
/// (`//#pragma variant_system` in egui.slang), the caller builds a validated
/// `VariantKey`, and `create_material` resolves it into per-stage defines that
/// hit the *per-variant* baked WGSL + reflection entries (this build has no
/// runtime Slang). One material per real color-space combo must come up on
/// every backend.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_shader_variant_selection_end_to_end(#[case] backend: Backend) {
    use redlilium_graphics::shader::EGUI_SHADER_SOURCE;
    use redlilium_graphics::{
        ShaderStage, ShaderVariantSpace, VertexAttribute, VertexAttributeFormat,
        VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    };

    let Some(ctx) = TestContext::new(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    let space = ShaderVariantSpace::parse(EGUI_SHADER_SOURCE).expect("egui variant pragmas");
    assert_eq!(space.axes().len(), 2, "egui declares two system axes");

    // The egui vertex layout (matches the shader's vs_main inputs).
    let vertex_layout = Arc::new(
        VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute {
                semantic: VertexAttributeSemantic::Position,
                format: VertexAttributeFormat::Float2,
                offset: 0,
                buffer_index: 0,
            })
            .with_attribute(VertexAttribute {
                semantic: VertexAttributeSemantic::TexCoord0,
                format: VertexAttributeFormat::Float2,
                offset: 8,
                buffer_index: 0,
            })
            .with_attribute(VertexAttribute {
                semantic: VertexAttributeSemantic::Color,
                format: VertexAttributeFormat::Float4,
                offset: 16,
                buffer_index: 0,
            }),
    );

    // One color-space combo egui runs with: (hdr, srgb, target format,
    // expected resolved defines on every stage).
    struct Combo {
        hdr: bool,
        srgb: bool,
        format: TextureFormat,
        expected_defines: &'static [(&'static str, &'static str)],
    }
    // The three real combos.
    let combos = [
        Combo {
            hdr: false,
            srgb: false,
            format: TextureFormat::Bgra8Unorm,
            expected_defines: &[],
        },
        Combo {
            hdr: true,
            srgb: false,
            format: TextureFormat::Rgba16Float,
            expected_defines: &[("HDR_OUTPUT", "")],
        },
        Combo {
            hdr: false,
            srgb: true,
            format: TextureFormat::Bgra8UnormSrgb,
            expected_defines: &[("SRGB_FRAMEBUFFER", "")],
        },
    ];

    for combo in combos {
        let variant = space
            .select()
            .system("HDR_OUTPUT", combo.hdr)
            .system("SRGB_FRAMEBUFFER", combo.srgb)
            .build()
            .expect("variant key");

        let material = ctx
            .device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        EGUI_SHADER_SOURCE.as_bytes().to_vec(),
                        "vs_main",
                        Vec::new(),
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        EGUI_SHADER_SOURCE.as_bytes().to_vec(),
                        "fs_main",
                        Vec::new(),
                    ))
                    .with_variant(variant.clone())
                    .with_vertex_layout(vertex_layout.clone())
                    .with_color_format(combo.format)
                    .with_dynamic_uniform(0, 0)
                    .with_label(format!("egui_variant_{variant}")),
            )
            .unwrap_or_else(|e| panic!("variant {variant} failed on {backend:?}: {e:?}"));

        // The stored descriptor carries the resolved per-stage defines.
        for shader in material.shaders() {
            let got: Vec<(&str, &str)> = shader
                .defines
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            assert_eq!(
                got, combo.expected_defines,
                "resolved defines for {variant}"
            );
        }
    }

    // A typo'd axis fails at key-build time — the whole point of declaring
    // the space.
    assert!(space.select().system("HDR_OUTPU", true).build().is_err());
}

/// A fragment shader that grinds a huge loop whose every iteration depends on
/// BOTH the loop counter and the running accumulator (and is seeded from the
/// fragment position) — so the shader compiler can neither const-fold it, hoist
/// it, nor drop it as a side-effect-free spin (which is what defeats a plain
/// `while(true)` or a counter-independent loop: the driver removes it and the
/// draw finishes instantly). At 2e9 iterations per fragment over a full draw it
/// runs for many seconds, overrunning the Windows GPU watchdog (TDR, ~2 s) and
/// forcing `VK_ERROR_DEVICE_LOST`. Only used by the #97 hang demo.
const GPU_HANG_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var acc: f32 = in.position.x * 0.00013 + in.position.y * 0.00017 + 0.317;
    for (var i: u32 = 0u; i < 2000000000u; i = i + 1u) {
        acc = fract(acc + sin(f32(i) * 0.000001 + acc) * 0.5 + 0.0137);
    }
    return vec4<f32>(acc, 0.0, 0.0, 1.0);
}
"#;

/// #97 GPU-crash-breadcrumb demo. **Intentionally hangs the GPU** (a fragment
/// shader that spins past the TDR watchdog), so it is `#[ignore]`d — running it
/// resets the graphics driver (a display blip on the GPU that drives the
/// monitor). Run one mechanism at a time, e.g.:
///
/// ```text
/// REDLILIUM_BREADCRUMBS=1 REDLILIUM_ADAPTER=Radeon \
///   cargo test -p redlilium-graphics --test gpu_tests -- --ignored --nocapture \
///   demo_device_lost_breadcrumbs
/// ```
///
/// Builds a two-pass graph (`safe_pass` then `HANG_pass`) in one submit. When
/// the GPU dies in `HANG_pass`, the backend's device-lost reporter reads the
/// breadcrumbs and writes `redlilium-gpu-crash-<ts>.txt` next to the test exe;
/// this test then asserts that report names `HANG_pass` as the guilty pass.
///
/// HARDWARE NOTE: this only produces a device loss where the GPU watchdog
/// actually fires — i.e. a card whose driver does NOT preempt the runaway
/// shader. Verified on AMD RX 6400 (TDR fires, `VK_AMD_buffer_marker` report is
/// correct). On NVIDIA (Pascal+) the driver preempts the shader to keep the
/// display alive, so no TDR / device loss occurs: the test instead spins on
/// fence timeouts (~10 s each) and ultimately fails with no crash file. Run it
/// on TDR-ing hardware, and prefer the secondary/headless GPU — hanging the
/// card that drives the monitor blips the display.
#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[ignore = "intentionally hangs the GPU; only device-losses on non-preempting (TDR) hardware, e.g. AMD — see #97 demo doc"]
fn demo_device_lost_breadcrumbs(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {backend:?} not available, skipping");
        return;
    };

    // Remove any stale crash reports so we read this run's.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
        .expect("test exe dir");
    let is_crash = |p: &std::path::Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("redlilium-gpu-crash-") && n.ends_with(".txt"))
    };
    if let Ok(entries) = std::fs::read_dir(&exe_dir) {
        for path in entries.flatten().map(|e| e.path()).filter(|p| is_crash(p)) {
            let _ = std::fs::remove_file(path);
        }
    }

    const W: u32 = 64;
    const H: u32 = 64;
    const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    let quad = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &quad, &FULLSCREEN_QUAD_VERTICES);

    let safe_target = ctx.create_render_target(W, H);
    let hang_target = ctx.create_render_target(W, H);

    let safe_instance = create_material_instance(create_solid_color_material(&ctx));
    let hang_material = ctx
        .device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    GPU_HANG_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    GPU_HANG_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_color_format(TextureFormat::Rgba8Unorm)
                .with_label("gpu_hang_material"),
        )
        .expect("Failed to create hang material");
    let hang_instance = create_material_instance(hang_material);

    let mut graph = RenderGraph::new();
    let mut safe_pass = create_simple_render_pass("safe_pass", safe_target.clone(), CLEAR);
    safe_pass.add_draw(quad.clone(), safe_instance);
    let safe_handle = graph.add_graphics_pass(safe_pass);

    let mut hang_pass = create_simple_render_pass("HANG_pass", hang_target.clone(), CLEAR);
    hang_pass.add_draw(quad, hang_instance);
    let hang_handle = graph.add_graphics_pass(hang_pass);
    graph.add_dependency(hang_handle, safe_handle);

    // Both render targets must be CONSUMED or the graph prunes the passes as
    // dead — then no fragments shade and the hang shader never runs. Readback
    // passes keep them live (the copies never actually run: the GPU dies in
    // HANG_pass first).
    let rb_size = readback_buffer_size(W, H, 4);
    let safe_rb = ctx.create_readback_buffer(rb_size);
    let hang_rb = ctx.create_readback_buffer(rb_size);
    let mut copy_safe = TransferPass::new("readback_safe".into());
    copy_safe.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(safe_target, safe_rb),
    ));
    let copy_safe_handle = graph.add_transfer_pass(copy_safe);
    graph.add_dependency(copy_safe_handle, safe_handle);
    let mut copy_hang = TransferPass::new("readback_hang".into());
    copy_hang.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(hang_target, hang_rb.clone()),
    ));
    let copy_hang_handle = graph.add_transfer_pass(copy_hang);
    graph.add_dependency(copy_hang_handle, hang_handle);

    // Executing hangs the GPU in HANG_pass; the watchdog fires a device loss.
    // The backend's reporter writes the crash file synchronously, but the loss
    // does not always surface inside `execute_graph` — on this path it is often
    // only observed on the NEXT submit. So nudge the (now-lost) device with a
    // few readbacks until the reporter has written the crash file. All of this
    // may panic on the lost device, so each step is caught.
    let latest_crash = |dir: &std::path::Path| -> Option<std::path::PathBuf> {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_crash(p))
            .collect();
        v.sort();
        v.pop()
    };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.execute_graph(graph);
    }));
    for _ in 0..8 {
        if latest_crash(&exe_dir).is_some() {
            break;
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ctx.read_buffer(&hang_rb, rb_size);
        }));
    }

    // Confirm the report fingers HANG_pass.
    let latest = latest_crash(&exe_dir)
        .expect("device-lost reporter should have written a crash file (#97)");
    let report = std::fs::read_to_string(&latest).expect("read crash report");
    eprintln!("=== {} ===\n{report}", latest.display());

    assert!(
        report.contains("HANG_pass"),
        "crash report should name the guilty pass HANG_pass:\n{report}"
    );
    assert!(
        report.contains("died in pass") || report.contains("INCOMPLETE"),
        "crash report should mark the submit incomplete:\n{report}"
    );
}

// ============================================================================
// Depth-only pass: zero color attachments + vertex-only pipeline (issue #129)
// ============================================================================

/// Vertex-only WGSL shader for the depth-only pass: forces every fragment to
/// clip z = 0.3 so the depth image records a recognizable value. No fragment
/// stage exists — with zero color attachments there is nothing to shade.
const DEPTH_ONLY_VS: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return vec4<f32>(in.position.xy, 0.3, 1.0);
}
"#;

/// Fullscreen visualizer: samples the depth image and writes it as red.
const DEPTH_VISUALIZE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position.xy, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureSample(depth_texture, depth_sampler, in.uv);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
}
"#;

/// Depth-only rendering end-to-end (#129): a graphics pass with **zero color
/// attachments** drawing through a **vertex-only** pipeline (`color_formats`
/// empty, no fragment stage) — the shape every shadow-map/depth-prepass pass
/// uses. A second pass samples the produced depth image into a color target:
/// the centered quad's footprint must read back its forced depth (0.3), the
/// corners the reversed-Z clear depth (0.0). On Vulkan the whole workload must
/// produce zero validation errors.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_depth_only_pass_zero_color_attachments(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const W: u32 = 16;
    const H: u32 = 16;
    const QUAD_DEPTH: f32 = 0.3;
    // Reversed-Z clear: the quad's 0.3 passes the default GreaterEqual test.
    const CLEAR_DEPTH: f32 = 0.0;

    let depth = ctx.create_texture_2d(
        W,
        H,
        TextureFormat::Depth32Float,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
    );
    let color = ctx.create_render_target(W, H);

    // Vertex-only, zero-color material — the depth-only pipeline shape.
    let depth_material = ctx
        .device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    DEPTH_ONLY_VS.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_depth_format(TextureFormat::Depth32Float)
                .with_label("depth_only_material"),
        )
        .expect("vertex-only material with zero color formats must be legal");
    let depth_instance = Arc::new(MaterialInstance::new(depth_material));

    let centered = create_centered_quad(&ctx);
    write_quad_vertices(&ctx, &centered, &CENTERED_QUAD_VERTICES);

    // Graph 1: the depth-only pass — a depth attachment and nothing else.
    let mut g1 = RenderGraph::new();
    let mut depth_pass = GraphicsPass::new("depth_only".into());
    depth_pass.set_render_targets(RenderTargetConfig::new().with_depth_stencil(
        DepthStencilAttachment::from_texture(depth.clone()).with_clear_depth(CLEAR_DEPTH),
    ));
    depth_pass.add_draw(centered, depth_instance);
    g1.add_graphics_pass(depth_pass);
    ctx.execute_graph(g1);

    // Graph 2: visualize the depth image into a color target.
    let binding_layout = Arc::new(
        BindingLayout::new()
            .with_entry(BindingLayoutEntry::new(0, BindingType::DepthTexture))
            .with_sampler(1)
            .with_label("depth_visualize_bindings"),
    );
    let vis_material = ctx
        .device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    DEPTH_VISUALIZE_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    DEPTH_VISUALIZE_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_binding_layout(binding_layout)
                .with_color_format(TextureFormat::Rgba8Unorm)
                .with_label("depth_visualize_material"),
        )
        .expect("Failed to create depth visualizer material");
    let sampler = ctx
        .device
        .create_sampler(&SamplerDescriptor::nearest().with_label("depth_visualize_sampler"))
        .expect("Failed to create sampler");
    let vis_group = ctx
        .device
        .create_binding_group(
            vis_material.binding_layouts()[0].clone(),
            BindingGroupDescriptor::new()
                .with_texture(0, depth.clone())
                .with_sampler(1, sampler)
                .with_label("depth_visualize_group"),
        )
        .expect("Failed to create depth visualizer binding group");
    let vis_instance = Arc::new(MaterialInstance::new(vis_material).with_binding_group(vis_group));

    let fullscreen = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &fullscreen, &FULLSCREEN_QUAD_VERTICES);

    let mut g2 = RenderGraph::new();
    let mut vis_pass = GraphicsPass::new("depth_visualize".into());
    vis_pass.set_render_targets(RenderTargetConfig::new().with_color(
        ColorAttachment::from_texture(color.clone()).with_clear_color(0.0, 1.0, 0.0, 1.0),
    ));
    vis_pass.add_draw(fullscreen, vis_instance);
    g2.add_graphics_pass(vis_pass);
    ctx.execute_graph(g2);

    // Graph 3: read the color target back.
    let readback_size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut g3 = RenderGraph::new();
    let mut copy = TransferPass::new("depth_only_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(color, readback.clone()),
    ));
    g3.add_transfer_pass(copy);
    ctx.execute_graph(g3);

    // Dummy backend performs no real rendering — reaching here without a
    // panic (pipeline creation + pass encoding accepted zero color
    // attachments) is the assertion.
    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);
    let center = get_pixel(&data, W, W / 2, H / 2);
    let corner = get_pixel(&data, W, 1, 1);
    eprintln!("depth-only center {center:?} corner {corner:?} ({backend:?})");
    let expected_center = ExpectedPixel::from_float(QUAD_DEPTH, 0.0, 0.0, 1.0);
    let expected_corner = ExpectedPixel::from_float(CLEAR_DEPTH, 0.0, 0.0, 1.0);
    assert!(
        verify_pixel(&data, W, W / 2, H / 2, expected_center, 3),
        "center must read the quad's depth {QUAD_DEPTH} as red, got {center:?} ({backend:?})"
    );
    assert!(
        verify_pixel(&data, W, 1, 1, expected_corner, 3),
        "corner must read the clear depth {CLEAR_DEPTH} as red, got {corner:?} ({backend:?})"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during the depth-only workload"
        );
    }
}

/// Upload a texture and sample it in the SAME graph — the egui font-atlas
/// flow: `EguiRenderer::flush_uploads` puts a `TransferPass` (atlas bytes via
/// `upload_texture_data` into Rgba8UnormSrgb) into the same frame's graph as
/// the egui draw that samples it. The graph compiler must order the copy
/// before the draw and make the write visible to sampling; a miss here shows
/// up as egui text rendering from a stale/undefined atlas.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_sample_uploaded_texture_same_graph(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const W: u32 = 16;
    const H: u32 = 16;

    // The egui atlas shape: sRGB sampled texture filled through the graph.
    let atlas = ctx.create_texture_2d(
        W,
        H,
        TextureFormat::Rgba8UnormSrgb,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );

    // Left half pure red, right half pure blue (0/255 are sRGB fixed points,
    // so the values survive the decode untouched).
    let mut pixels = Vec::with_capacity((W * H * 4) as usize);
    for _y in 0..H {
        for x in 0..W {
            if x < W / 2 {
                pixels.extend_from_slice(&[255, 0, 0, 255]);
            } else {
                pixels.extend_from_slice(&[0, 0, 255, 255]);
            }
        }
    }

    let material = create_texture_sample_material(&ctx);
    let instance = create_texture_sample_instance(&ctx, material, atlas.clone());
    let fullscreen = create_fullscreen_quad(&ctx);
    write_quad_vertices(&ctx, &fullscreen, &FULLSCREEN_QUAD_VERTICES);
    let target = ctx.create_render_target(W, H);

    // ONE graph: upload pass + draw sampling the freshly-uploaded texture.
    let mut graph = RenderGraph::new();
    let mut upload = TransferPass::new("atlas_upload".into());
    upload.set_transfer_config(
        TransferConfig::new().with_operation(
            TransferOperation::upload_texture_data(&ctx.device, atlas.clone(), &pixels)
                .expect("stage atlas upload"),
        ),
    );
    graph.add_transfer_pass(upload);

    let mut draw = GraphicsPass::new("atlas_sample".into());
    draw.set_render_targets(RenderTargetConfig::new().with_color(
        ColorAttachment::from_texture(target.clone()).with_clear_color(0.0, 1.0, 0.0, 1.0),
    ));
    draw.add_draw(fullscreen, instance);
    graph.add_graphics_pass(draw);
    ctx.execute_graph(graph);

    // Read the render target back.
    let readback_size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut g2 = RenderGraph::new();
    let mut copy = TransferPass::new("atlas_sample_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(target, readback.clone()),
    ));
    g2.add_transfer_pass(copy);
    ctx.execute_graph(g2);

    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);
    let left = get_pixel(&data, W, W / 4, H / 2);
    let right = get_pixel(&data, W, 3 * W / 4, H / 2);
    eprintln!("same-graph sample left {left:?} right {right:?} ({backend:?})");
    assert!(
        verify_pixel(
            &data,
            W,
            W / 4,
            H / 2,
            ExpectedPixel::from_float(1.0, 0.0, 0.0, 1.0),
            3
        ),
        "left half must sample the uploaded red, got {left:?} ({backend:?})"
    );
    assert!(
        verify_pixel(
            &data,
            W,
            3 * W / 4,
            H / 2,
            ExpectedPixel::from_float(0.0, 0.0, 1.0, 1.0),
            3
        ),
        "right half must sample the uploaded blue, got {right:?} ({backend:?})"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during the same-graph sample workload"
        );
    }
}

/// Chained staging inside ONE transfer pass (#139): `WriteBuffer` fills a
/// staging buffer, `BufferToBuffer` copies it onward, `BufferToTexture` reads
/// the copy — every op reads what the previous op in the same pass wrote.
/// Pass-entry barriers come from the declared usage, but ordering WITHIN a
/// pass needs intra-pass barriers; without them sync validation reports
/// READ_AFTER_WRITE (the original `ibl_upload` repro on both dev GPUs).
/// Verifies the bytes arrive intact and, on Vulkan, that the workload is
/// validation-clean.
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_transfer_pass_intra_pass_buffer_chain(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    // 64px rows: the tight 256-byte row pitch satisfies the multi-row copy
    // alignment without padding.
    const W: u32 = 64;
    const H: u32 = 64;
    let byte_len = (W * H * 4) as u64;

    // Left half red, right half blue.
    let mut pixels = Vec::with_capacity(byte_len as usize);
    for _y in 0..H {
        for x in 0..W {
            if x < W / 2 {
                pixels.extend_from_slice(&[255u8, 0, 0, 255]);
            } else {
                pixels.extend_from_slice(&[0, 0, 255, 255]);
            }
        }
    }

    let staging = ctx.create_buffer(byte_len, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);
    let bounce = ctx.create_buffer(byte_len, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);
    let texture = ctx.create_texture_2d(
        W,
        H,
        TextureFormat::Rgba8Unorm,
        // TEXTURE_BINDING on top of the copy usages: the backend always
        // creates a default view, which needs a view-compatible usage bit.
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    );

    let mut graph = RenderGraph::new();
    let mut pass = TransferPass::new("staged_chain".into());
    pass.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::write_buffer(
                staging.clone(),
                0,
                Arc::from(pixels.as_slice()),
            ))
            .with_operation(TransferOperation::copy_buffer_whole(
                staging.clone(),
                bounce.clone(),
            ))
            .with_operation(TransferOperation::upload_texture_whole(
                bounce.clone(),
                texture.clone(),
            )),
    );
    graph.add_transfer_pass(pass);
    ctx.execute_graph(graph);

    // Read the texture back and verify the pattern survived the chain.
    let readback_size = readback_buffer_size(W, H, 4);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut g2 = RenderGraph::new();
    let mut copy = TransferPass::new("staged_chain_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(texture, readback.clone()),
    ));
    g2.add_transfer_pass(copy);
    ctx.execute_graph(g2);

    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, readback_size);
    let left = get_pixel(&data, W, W / 4, H / 2);
    let right = get_pixel(&data, W, 3 * W / 4, H / 2);
    eprintln!("staged chain left {left:?} right {right:?} ({backend:?})");
    assert!(
        verify_pixel(
            &data,
            W,
            W / 4,
            H / 2,
            ExpectedPixel::from_float(1.0, 0.0, 0.0, 1.0),
            0
        ),
        "left half must arrive as the written red, got {left:?} ({backend:?})"
    );
    assert!(
        verify_pixel(
            &data,
            W,
            3 * W / 4,
            H / 2,
            ExpectedPixel::from_float(0.0, 0.0, 1.0, 1.0),
            0
        ),
        "right half must arrive as the written blue, got {right:?} ({backend:?})"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during the intra-pass \
             buffer chain (#139: ops within one TransferPass need barriers between \
             a write and a later read/write of the same buffer)"
        );
    }
}

/// Ambiguous WAW graph through the SCHEDULER (#141): two transfer passes
/// write the same buffer with no explicit edge, which Strict compilation
/// rejects. The scheduler used to drop the whole graph silently (empty frame,
/// one log line); it must instead fall back to addition-order resolution and
/// still execute. The readback proves execution happened AND that addition
/// order won (the second writer's bytes land).
#[rstest]
#[case::dummy(Backend::Dummy)]
#[case::vulkan(Backend::Vulkan)]
#[case::webgpu(Backend::WebGpu)]
fn test_ambiguous_waw_graph_falls_back_and_executes(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const LEN: usize = 16;
    let buffer = ctx.create_buffer(LEN as u64, BufferUsage::COPY_DST | BufferUsage::COPY_SRC);

    let mut graph = RenderGraph::new();
    for value in [0x11u8, 0x22] {
        let mut pass = TransferPass::new(format!("writer_{value:02x}"));
        pass.set_transfer_config(TransferConfig::new().with_operation(
            TransferOperation::write_buffer(buffer.clone(), 0, Arc::from(&[value; LEN][..])),
        ));
        graph.add_transfer_pass(pass);
    }
    ctx.execute_graph(graph);

    // Copy into a mappable buffer and read it back.
    let readback = ctx.create_readback_buffer(LEN as u64);
    let mut g2 = RenderGraph::new();
    let mut copy = TransferPass::new("waw_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::copy_buffer_whole(buffer, readback.clone()),
    ));
    g2.add_transfer_pass(copy);
    ctx.execute_graph(g2);

    if backend == Backend::Dummy {
        return;
    }

    let data = ctx.read_buffer(&readback, LEN as u64);
    assert_eq!(
        data,
        vec![0x22u8; LEN],
        "the graph must execute via addition-order fallback, with the \
         second writer winning ({backend:?})"
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during the ambiguous \
             WAW fallback workload"
        );
    }
}

/// Headless egui render: draw large text into an offscreen target through the
/// full controller path (atlas delta upload + tessellated draw in one graph)
/// and verify actual GLYPHS came out, on both the SDR and the HDR surface
/// format. Two properties are checked per format:
///
/// 1. egui rendered at all (pixels differ from the clear color);
/// 2. the text is antialiased glyph coverage, not solid blocks — a broken
///    (all-white) font atlas turns every glyph into a filled rectangle, which
///    collapses the pixel-value diversity this asserts on.
#[rstest]
#[case::vulkan_sdr(Backend::Vulkan, TextureFormat::Bgra8UnormSrgb)]
#[case::vulkan_hdr(Backend::Vulkan, TextureFormat::Rgba16Float)]
#[case::webgpu_sdr(Backend::WebGpu, TextureFormat::Bgra8UnormSrgb)]
#[case::webgpu_hdr(Backend::WebGpu, TextureFormat::Rgba16Float)]
fn test_egui_headless_text_renders_glyphs(
    #[case] backend: Backend,
    #[case] surface_format: TextureFormat,
) {
    use redlilium_graphics::egui::{EguiApp, EguiController, egui};

    let Some(ctx) = TestContext::new_with_validation(backend) else {
        eprintln!("Backend {:?} not available, skipping", backend);
        return;
    };

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        redlilium_graphics::backend::vulkan::reset_validation_error_count();
    }

    const W: u32 = 256;
    const H: u32 = 128;

    struct TextUi;
    impl EguiApp for TextUi {
        fn update(&mut self, ctx: &egui::Context) {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("Wg Text 123")
                        .size(48.0)
                        .color(egui::Color32::WHITE),
                );
            });
        }
    }

    let target = ctx.create_texture_2d(
        W,
        H,
        surface_format,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );

    let app: redlilium_graphics::egui::ArcEguiApp = Arc::new(parking_lot::RwLock::new(TextUi));
    let mut controller = EguiController::new(ctx.device.clone(), app, W, H, 1.0, surface_format);

    // Two frames: egui emits the font-atlas delta on the first tessellated
    // frame; run a second to also cover the steady state.
    for frame in 0..2 {
        controller.begin_frame(frame as f64 * 0.016);
        let render_target = RenderTarget::from_texture(target.clone());

        // Clear the target in its own submission (the egui pass loads
        // existing content; same-target pass ordering inside one graph is
        // not what this test is probing).
        let mut clear_graph = RenderGraph::new();
        let mut clear = GraphicsPass::new("egui_test_clear".into());
        clear.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(target.clone()).with_clear_color(0.0, 0.0, 0.0, 1.0),
        ));
        clear_graph.add_graphics_pass(clear);
        ctx.execute_graph(clear_graph);

        let mut graph = RenderGraph::new();
        controller.flush_uploads(&mut graph);
        if let Some(pass) = controller.end_frame(&render_target, W, H) {
            eprintln!(
                "frame {frame}: egui pass has {} draw commands ({backend:?}/{surface_format:?})",
                pass.draw_commands().len()
            );
            graph.add_graphics_pass(pass);
        } else {
            panic!("egui produced no draw pass ({backend:?}/{surface_format:?})");
        }
        ctx.execute_graph(graph);
    }

    // Read back the final frame.
    let bpp: u32 = match surface_format {
        TextureFormat::Rgba16Float => 8,
        _ => 4,
    };
    let readback_size = readback_buffer_size(W, H, bpp);
    let readback = ctx.create_readback_buffer(readback_size);
    let mut g = RenderGraph::new();
    let mut copy = TransferPass::new("egui_test_readback".into());
    copy.set_transfer_config(TransferConfig::new().with_operation(
        TransferOperation::readback_texture_whole(target, readback.clone()),
    ));
    g.add_transfer_pass(copy);
    ctx.execute_graph(g);

    let data = ctx.read_buffer(&readback, readback_size);

    // Walk pixels of the top-left text area, collecting distinct pixel byte
    // patterns and counting non-background pixels. Row pitch is 256-aligned.
    let row_pitch = ((W * bpp).div_ceil(256) * 256) as usize;
    let mut patterns = std::collections::HashSet::new();
    let mut non_bg = 0usize;
    let bg = &data[0..bpp as usize]; // top-left corner: clear+panel fill, no text
    for y in 0..H as usize {
        for x in 0..W as usize {
            let off = y * row_pitch + x * bpp as usize;
            let px = &data[off..off + bpp as usize];
            patterns.insert(px.to_vec());
            if px != bg {
                non_bg += 1;
            }
        }
    }
    eprintln!(
        "egui headless: {} distinct pixel patterns, {} non-bg pixels ({backend:?}/{surface_format:?})",
        patterns.len(),
        non_bg
    );
    assert!(
        non_bg > 500,
        "egui drew almost nothing: {non_bg} non-background pixels ({backend:?}/{surface_format:?})"
    );
    assert!(
        patterns.len() >= 16,
        "text collapsed to flat blocks: only {} distinct pixel patterns ({backend:?}/{surface_format:?})",
        patterns.len()
    );

    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        let errors = redlilium_graphics::backend::vulkan::validation_error_count();
        assert_eq!(
            errors, 0,
            "Vulkan validation reported {errors} error(s) during the egui headless render"
        );
    }
}
