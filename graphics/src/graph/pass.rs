//! Render pass types and handles.

use std::sync::atomic::{AtomicU32, Ordering};

use super::target::RenderTargetConfig;

/// Handle to a render pass in the graph.
///
/// Handles are lightweight and can be copied freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassHandle {
    index: u32,
    generation: u32,
}

impl PassHandle {
    /// Create a new pass handle.
    pub(crate) fn new(index: u32) -> Self {
        static GENERATION: AtomicU32 = AtomicU32::new(0);
        Self {
            index,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Get the index of this pass.
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Type of render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PassType {
    /// Graphics pass (vertex/fragment shaders, rasterization).
    #[default]
    Graphics,
    /// Compute pass (compute shaders).
    Compute,
    /// Transfer pass (copy operations).
    Transfer,
}

/// A render pass in the graph.
///
/// Render passes describe a unit of GPU work with its resource dependencies.
/// For graphics passes, render targets can be configured to specify where
/// the pass renders to.
#[derive(Debug)]
pub struct RenderPass {
    /// Debug name for the pass.
    name: String,
    /// Type of pass.
    pass_type: PassType,
    /// Passes that must execute before this one.
    dependencies: Vec<PassHandle>,
    /// Render targets for graphics passes.
    render_targets: Option<RenderTargetConfig>,
}

impl RenderPass {
    /// Create a new render pass.
    pub fn new(name: String, pass_type: PassType) -> Self {
        Self {
            name,
            pass_type,
            dependencies: Vec::new(),
            render_targets: None,
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass type.
    pub fn pass_type(&self) -> PassType {
        self.pass_type
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> &[PassHandle] {
        &self.dependencies
    }

    /// Add a dependency on another pass.
    pub fn add_dependency(&mut self, pass: PassHandle) {
        if !self.dependencies.contains(&pass) {
            self.dependencies.push(pass);
        }
    }

    /// Get the render target configuration.
    pub fn render_targets(&self) -> Option<&RenderTargetConfig> {
        self.render_targets.as_ref()
    }

    /// Set the render target configuration.
    ///
    /// This is only meaningful for graphics passes.
    pub fn set_render_targets(&mut self, config: RenderTargetConfig) {
        self.render_targets = Some(config);
    }

    /// Check if this pass has render targets configured.
    pub fn has_render_targets(&self) -> bool {
        self.render_targets
            .as_ref()
            .map(|c| c.has_attachments())
            .unwrap_or(false)
    }
}
