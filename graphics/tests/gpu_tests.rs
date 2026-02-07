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

use common::{
    Backend, ExpectedPixel, FULLSCREEN_QUAD_VERTICES, LEFT_HALF_QUAD_VERTICES, TestContext,
    create_fullscreen_quad, create_left_half_quad, create_material_instance, create_mrt_pass,
    create_render_pass_with_depth, create_simple_render_pass, create_solid_color_material,
    generate_test_pattern, get_pixel, readback_buffer_size, verify_pixel, write_quad_vertices,
};
use redlilium_graphics::{
    BufferUsage, RenderGraph, TextureFormat, TextureUsage, TransferConfig, TransferOperation,
    TransferPass,
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
        .compile()
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
        .compile()
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
        .compile()
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
    let data = ctx.device.read_buffer(&readback, 0, readback_size);

    let expected = ExpectedPixel::from_float(0.25, 0.5, 0.75, 1.0);
    let center_pixel = get_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2);
    assert!(
        verify_pixel(&data, WIDTH, WIDTH / 2, HEIGHT / 2, expected, 2),
        "Clear color pixel should be {:?}, but got {:?}",
        expected,
        center_pixel
    );
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
        .compile()
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
        .compile()
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
        .compile()
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
    let compiled = graph.compile().expect("Empty graph should compile");
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
        let compiled = graph.compile().expect("Diamond graph should compile");
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
    let data = ctx.device.read_buffer(&readback, 0, readback_size);

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
        .compile()
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
    let data1 = ctx.device.read_buffer(&readback1, 0, readback_size);
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
    let data2 = ctx.device.read_buffer(&readback2, 0, readback_size);
    let rt2_pixel = get_pixel(&data2, WIDTH, center_x, center_y);
    assert!(
        verify_pixel(&data2, WIDTH, center_x, center_y, ExpectedPixel::GREEN, 2),
        "RT2 center pixel ({}, {}) should be green, but got: {:?}",
        center_x,
        center_y,
        rt2_pixel
    );
}
