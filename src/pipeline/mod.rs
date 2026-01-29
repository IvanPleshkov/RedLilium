//! Forward+ rendering pipeline
//!
//! This module implements a tiled forward (Forward+) rendering pipeline:
//! 1. Depth pre-pass - Renders depth for early-z and light culling
//! 2. Light culling - Compute pass to assign lights to screen tiles
//! 3. Forward pass - Renders geometry with per-tile light lists
//! 4. Post-processing - Bloom, tonemapping, etc.

pub mod depth_prepass;
pub mod forward_pass;
pub mod light_culling;
pub mod postprocess;

pub use depth_prepass::DepthPrepass;
pub use forward_pass::ForwardPlusPass;
pub use light_culling::LightCullingPass;

use crate::render_graph::{PassType, RenderGraph, RenderGraphBuilder, ResourceId};
use crate::scene::DEFAULT_TILE_SIZE;

/// Configuration for the Forward+ pipeline
#[derive(Debug, Clone)]
pub struct ForwardPlusConfig {
    /// Size of screen tiles for light culling (pixels)
    pub tile_size: u32,
    /// Maximum number of lights
    pub max_lights: u32,
    /// Enable bloom post-processing
    pub enable_bloom: bool,
    /// Enable FXAA
    pub enable_fxaa: bool,
}

impl Default for ForwardPlusConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            max_lights: 1024,
            enable_bloom: true,
            enable_fxaa: false,
        }
    }
}

/// Resources created by the Forward+ pipeline
pub struct ForwardPlusResources {
    pub depth_buffer: ResourceId,
    pub hdr_color: ResourceId,
    pub light_buffer: ResourceId,
    pub tile_light_buffer: ResourceId,
}

/// Build the Forward+ render graph
pub fn build_forward_plus_graph(
    width: u32,
    height: u32,
    config: &ForwardPlusConfig,
) -> (RenderGraph, ForwardPlusResources) {
    let mut graph = RenderGraph::new();

    // Register swapchain as external resource
    let swapchain = graph.register_external("swapchain");

    // Add depth prepass
    let depth_prepass = DepthPrepass::new();
    let _depth_pass_id = graph.add_pass(depth_prepass, PassType::Graphics, width, height);

    // Add light culling compute pass
    let light_culling = LightCullingPass::new(config.tile_size, config.max_lights);
    let _culling_pass_id = graph.add_pass(light_culling, PassType::Compute, width, height);

    // Add forward+ pass
    let forward_pass = ForwardPlusPass::new(config.tile_size);
    let _forward_pass_id = graph.add_pass(forward_pass, PassType::Graphics, width, height);

    // Add post-processing passes
    if config.enable_bloom {
        let bloom_pass = postprocess::BloomPass::new();
        let _bloom_pass_id = graph.add_pass(bloom_pass, PassType::Graphics, width, height);
    }

    // Add tonemapping pass (always needed for HDR -> LDR)
    let tonemap_pass = postprocess::TonemappingPass::new(swapchain);
    let _tonemap_pass_id = graph.add_pass(tonemap_pass, PassType::Graphics, width, height);

    // Get resource IDs (these are created during pass setup)
    // For now, use placeholder IDs since resources are created during setup
    let resources = ForwardPlusResources {
        depth_buffer: ResourceId(0),
        hdr_color: ResourceId(1),
        light_buffer: ResourceId(2),
        tile_light_buffer: ResourceId(3),
    };

    (graph, resources)
}
