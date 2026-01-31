//! Render pass types.

use super::PassHandle;
use super::target::RenderTargetConfig;
use super::transfer::TransferConfig;

/// A pass in the render graph.
///
/// Passes describe units of GPU work with their resource dependencies.
/// Each variant has its own configuration specific to that pass type.
#[derive(Debug)]
pub enum Pass {
    /// Graphics pass (vertex/fragment shaders, rasterization).
    Graphics(GraphicsPass),
    /// Transfer pass (copy operations).
    Transfer(TransferPass),
    /// Compute pass (compute shaders).
    Compute(ComputePass),
}

impl Pass {
    /// Get the pass name.
    pub fn name(&self) -> &str {
        match self {
            Pass::Graphics(p) => p.name(),
            Pass::Transfer(p) => p.name(),
            Pass::Compute(p) => p.name(),
        }
    }

    /// Get the pass dependencies as handles.
    pub fn dependencies(&self) -> &[PassHandle] {
        match self {
            Pass::Graphics(p) => p.dependencies(),
            Pass::Transfer(p) => p.dependencies(),
            Pass::Compute(p) => p.dependencies(),
        }
    }

    /// Add a dependency on another pass (internal use).
    pub(crate) fn add_dependency(&mut self, handle: PassHandle) {
        match self {
            Pass::Graphics(p) => p.add_dependency(handle),
            Pass::Transfer(p) => p.add_dependency(handle),
            Pass::Compute(p) => p.add_dependency(handle),
        }
    }

    /// Get this pass as a graphics pass, if it is one.
    pub fn as_graphics(&self) -> Option<&GraphicsPass> {
        if let Pass::Graphics(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Get this pass as a mutable graphics pass, if it is one.
    pub fn as_graphics_mut(&mut self) -> Option<&mut GraphicsPass> {
        if let Pass::Graphics(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Get this pass as a transfer pass, if it is one.
    pub fn as_transfer(&self) -> Option<&TransferPass> {
        if let Pass::Transfer(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Get this pass as a mutable transfer pass, if it is one.
    pub fn as_transfer_mut(&mut self) -> Option<&mut TransferPass> {
        if let Pass::Transfer(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Get this pass as a compute pass, if it is one.
    pub fn as_compute(&self) -> Option<&ComputePass> {
        if let Pass::Compute(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Get this pass as a mutable compute pass, if it is one.
    pub fn as_compute_mut(&mut self) -> Option<&mut ComputePass> {
        if let Pass::Compute(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Check if this is a graphics pass.
    pub fn is_graphics(&self) -> bool {
        matches!(self, Pass::Graphics(_))
    }

    /// Check if this is a transfer pass.
    pub fn is_transfer(&self) -> bool {
        matches!(self, Pass::Transfer(_))
    }

    /// Check if this is a compute pass.
    pub fn is_compute(&self) -> bool {
        matches!(self, Pass::Compute(_))
    }
}

// ============================================================================
// Graphics Pass
// ============================================================================

/// A graphics pass for rasterization work.
///
/// Graphics passes execute vertex and fragment shaders to render geometry.
/// They can have render targets configured to specify where they render to.
#[derive(Debug)]
pub struct GraphicsPass {
    name: String,
    dependencies: Vec<PassHandle>,
    render_targets: Option<RenderTargetConfig>,
}

impl GraphicsPass {
    /// Create a new graphics pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            dependencies: Vec::new(),
            render_targets: None,
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> &[PassHandle] {
        &self.dependencies
    }

    /// Add a dependency on another pass.
    pub(crate) fn add_dependency(&mut self, handle: PassHandle) {
        if !self.dependencies.contains(&handle) {
            self.dependencies.push(handle);
        }
    }

    /// Get the render target configuration.
    pub fn render_targets(&self) -> Option<&RenderTargetConfig> {
        self.render_targets.as_ref()
    }

    /// Set the render target configuration.
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

// ============================================================================
// Transfer Pass
// ============================================================================

/// A transfer pass for data copy operations.
///
/// Transfer passes execute buffer and texture copy commands.
#[derive(Debug)]
pub struct TransferPass {
    name: String,
    dependencies: Vec<PassHandle>,
    transfer_config: Option<TransferConfig>,
}

impl TransferPass {
    /// Create a new transfer pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            dependencies: Vec::new(),
            transfer_config: None,
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> &[PassHandle] {
        &self.dependencies
    }

    /// Add a dependency on another pass.
    pub(crate) fn add_dependency(&mut self, handle: PassHandle) {
        if !self.dependencies.contains(&handle) {
            self.dependencies.push(handle);
        }
    }

    /// Get the transfer configuration.
    pub fn transfer_config(&self) -> Option<&TransferConfig> {
        self.transfer_config.as_ref()
    }

    /// Set the transfer configuration.
    pub fn set_transfer_config(&mut self, config: TransferConfig) {
        self.transfer_config = Some(config);
    }

    /// Check if this pass has transfer operations configured.
    pub fn has_transfers(&self) -> bool {
        self.transfer_config
            .as_ref()
            .map(|c| c.has_operations())
            .unwrap_or(false)
    }
}

// ============================================================================
// Compute Pass
// ============================================================================

/// A compute pass for compute shader work.
///
/// Compute passes execute compute shaders for general-purpose GPU computation.
#[derive(Debug)]
pub struct ComputePass {
    name: String,
    dependencies: Vec<PassHandle>,
    // Future: compute-specific configuration (dispatch size, etc.)
}

impl ComputePass {
    /// Create a new compute pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            dependencies: Vec::new(),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> &[PassHandle] {
        &self.dependencies
    }

    /// Add a dependency on another pass.
    pub(crate) fn add_dependency(&mut self, handle: PassHandle) {
        if !self.dependencies.contains(&handle) {
            self.dependencies.push(handle);
        }
    }
}
