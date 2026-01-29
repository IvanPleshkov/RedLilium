//! Deferred rendering pipeline
//!
//! This module implements a deferred rendering pipeline:
//! 1. G-Buffer pass - Renders geometry to multiple render targets
//! 2. Lighting pass - Fullscreen pass computing lighting from G-buffer
//! 3. Post-processing - Bloom, tonemapping, etc.

pub mod gbuffer_pass;
pub mod lighting_pass;
pub mod postprocess;

pub use gbuffer_pass::GBufferPass;
pub use lighting_pass::LightingPass;

use crate::render_graph::{PassType, RenderGraph, ResourceId};

/// Configuration for the Deferred pipeline
#[derive(Debug, Clone)]
pub struct DeferredConfig {
    /// Maximum number of lights
    pub max_lights: u32,
    /// Enable bloom post-processing
    pub enable_bloom: bool,
    /// Enable FXAA
    pub enable_fxaa: bool,
}

impl Default for DeferredConfig {
    fn default() -> Self {
        Self {
            max_lights: 1024,
            enable_bloom: true,
            enable_fxaa: false,
        }
    }
}

/// Resources created by the Deferred pipeline
pub struct DeferredResources {
    pub gbuffer_albedo: ResourceId,
    pub gbuffer_normal: ResourceId,
    pub gbuffer_material: ResourceId,
    pub gbuffer_depth: ResourceId,
    pub hdr_color: ResourceId,
    pub light_buffer: ResourceId,
}

/// Build the Deferred render graph
pub fn build_deferred_graph(
    width: u32,
    height: u32,
    config: &DeferredConfig,
) -> (RenderGraph, DeferredResources) {
    let mut graph = RenderGraph::new();

    // Register swapchain as external resource
    let swapchain = graph.register_external("swapchain");

    // Add G-Buffer pass
    let gbuffer_pass = GBufferPass::new();
    let _gbuffer_pass_id = graph.add_pass(gbuffer_pass, PassType::Graphics, width, height);

    // Get G-buffer resource IDs (these are created during setup)
    // We need to create a separate pass instance to get the resource IDs
    // For now, use sequential IDs based on creation order
    let gbuffer_albedo = ResourceId(1);
    let gbuffer_normal = ResourceId(2);
    let gbuffer_material = ResourceId(3);
    let gbuffer_depth = ResourceId(4);

    // Add Lighting pass
    let mut lighting_pass = LightingPass::new(config.max_lights);
    lighting_pass.set_gbuffer_resources(
        gbuffer_albedo,
        gbuffer_normal,
        gbuffer_material,
        gbuffer_depth,
    );
    let _lighting_pass_id = graph.add_pass(lighting_pass, PassType::Graphics, width, height);

    // Add post-processing passes
    if config.enable_bloom {
        let bloom_pass = postprocess::BloomPass::new();
        let _bloom_pass_id = graph.add_pass(bloom_pass, PassType::Graphics, width, height);
    }

    // Add tonemapping pass (always needed for HDR -> LDR)
    let tonemap_pass = postprocess::TonemappingPass::new(swapchain);
    let _tonemap_pass_id = graph.add_pass(tonemap_pass, PassType::Graphics, width, height);

    // Resource IDs
    let resources = DeferredResources {
        gbuffer_albedo,
        gbuffer_normal,
        gbuffer_material,
        gbuffer_depth,
        hdr_color: ResourceId(5),
        light_buffer: ResourceId(6),
    };

    (graph, resources)
}

// Keep the old Forward+ types for backwards compatibility during migration
// These can be removed once the engine is fully migrated

/// Configuration for the Forward+ pipeline (deprecated, use DeferredConfig)
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
            tile_size: 16,
            max_lights: 1024,
            enable_bloom: true,
            enable_fxaa: false,
        }
    }
}

/// Resources created by the Forward+ pipeline (deprecated)
pub struct ForwardPlusResources {
    pub depth_buffer: ResourceId,
    pub hdr_color: ResourceId,
    pub light_buffer: ResourceId,
    pub tile_light_buffer: ResourceId,
}

/// Build the Forward+ render graph (deprecated, use build_deferred_graph)
/// This now builds a deferred graph internally for backwards compatibility
pub fn build_forward_plus_graph(
    width: u32,
    height: u32,
    config: &ForwardPlusConfig,
) -> (RenderGraph, ForwardPlusResources) {
    let deferred_config = DeferredConfig {
        max_lights: config.max_lights,
        enable_bloom: config.enable_bloom,
        enable_fxaa: config.enable_fxaa,
    };

    let (graph, deferred_resources) = build_deferred_graph(width, height, &deferred_config);

    // Map deferred resources to forward+ resource struct
    let resources = ForwardPlusResources {
        depth_buffer: deferred_resources.gbuffer_depth,
        hdr_color: deferred_resources.hdr_color,
        light_buffer: deferred_resources.light_buffer,
        tile_light_buffer: ResourceId(0), // Not used in deferred
    };

    (graph, resources)
}
