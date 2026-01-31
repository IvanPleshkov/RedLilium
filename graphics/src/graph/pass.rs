//! Render pass types.

use std::sync::{Arc, RwLock, Weak};

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

    /// Get the pass dependencies.
    ///
    /// Returns only valid (still-alive) dependencies.
    pub fn dependencies(&self) -> Vec<Arc<Pass>> {
        match self {
            Pass::Graphics(p) => p.dependencies(),
            Pass::Transfer(p) => p.dependencies(),
            Pass::Compute(p) => p.dependencies(),
        }
    }

    /// Add a dependency on another pass.
    ///
    /// The dependency is stored as a weak reference to prevent cycles.
    pub fn add_dependency(&self, pass: &Arc<Pass>) {
        match self {
            Pass::Graphics(p) => p.add_dependency(pass),
            Pass::Transfer(p) => p.add_dependency(pass),
            Pass::Compute(p) => p.add_dependency(pass),
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

    /// Get this pass as a transfer pass, if it is one.
    pub fn as_transfer(&self) -> Option<&TransferPass> {
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

/// Inner mutable state of a graphics pass.
#[derive(Debug, Default)]
struct GraphicsPassInner {
    dependencies: Vec<Weak<Pass>>,
    render_targets: Option<RenderTargetConfig>,
}

/// A graphics pass for rasterization work.
///
/// Graphics passes execute vertex and fragment shaders to render geometry.
/// They can have render targets configured to specify where they render to.
#[derive(Debug)]
pub struct GraphicsPass {
    name: String,
    inner: RwLock<GraphicsPassInner>,
}

impl GraphicsPass {
    /// Create a new graphics pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            inner: RwLock::new(GraphicsPassInner::default()),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> Vec<Arc<Pass>> {
        let inner = self.inner.read().unwrap();
        inner
            .dependencies
            .iter()
            .filter_map(|weak| weak.upgrade())
            .collect()
    }

    /// Add a dependency on another pass.
    pub fn add_dependency(&self, pass: &Arc<Pass>) {
        let mut inner = self.inner.write().unwrap();
        let already_exists = inner.dependencies.iter().any(|weak| {
            weak.upgrade()
                .map(|existing| Arc::ptr_eq(&existing, pass))
                .unwrap_or(false)
        });
        if !already_exists {
            inner.dependencies.push(Arc::downgrade(pass));
        }
    }

    /// Get the render target configuration.
    pub fn render_targets(&self) -> Option<RenderTargetConfig> {
        let inner = self.inner.read().unwrap();
        inner.render_targets.clone()
    }

    /// Set the render target configuration.
    pub fn set_render_targets(&self, config: RenderTargetConfig) {
        let mut inner = self.inner.write().unwrap();
        inner.render_targets = Some(config);
    }

    /// Check if this pass has render targets configured.
    pub fn has_render_targets(&self) -> bool {
        let inner = self.inner.read().unwrap();
        inner
            .render_targets
            .as_ref()
            .map(|c| c.has_attachments())
            .unwrap_or(false)
    }
}

// ============================================================================
// Transfer Pass
// ============================================================================

/// Inner mutable state of a transfer pass.
#[derive(Debug, Default)]
struct TransferPassInner {
    dependencies: Vec<Weak<Pass>>,
    transfer_config: Option<TransferConfig>,
}

/// A transfer pass for data copy operations.
///
/// Transfer passes execute buffer and texture copy commands.
#[derive(Debug)]
pub struct TransferPass {
    name: String,
    inner: RwLock<TransferPassInner>,
}

impl TransferPass {
    /// Create a new transfer pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            inner: RwLock::new(TransferPassInner::default()),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> Vec<Arc<Pass>> {
        let inner = self.inner.read().unwrap();
        inner
            .dependencies
            .iter()
            .filter_map(|weak| weak.upgrade())
            .collect()
    }

    /// Add a dependency on another pass.
    pub fn add_dependency(&self, pass: &Arc<Pass>) {
        let mut inner = self.inner.write().unwrap();
        let already_exists = inner.dependencies.iter().any(|weak| {
            weak.upgrade()
                .map(|existing| Arc::ptr_eq(&existing, pass))
                .unwrap_or(false)
        });
        if !already_exists {
            inner.dependencies.push(Arc::downgrade(pass));
        }
    }

    /// Get the transfer configuration.
    pub fn transfer_config(&self) -> Option<TransferConfig> {
        let inner = self.inner.read().unwrap();
        inner.transfer_config.clone()
    }

    /// Set the transfer configuration.
    pub fn set_transfer_config(&self, config: TransferConfig) {
        let mut inner = self.inner.write().unwrap();
        inner.transfer_config = Some(config);
    }

    /// Check if this pass has transfer operations configured.
    pub fn has_transfers(&self) -> bool {
        let inner = self.inner.read().unwrap();
        inner
            .transfer_config
            .as_ref()
            .map(|c| c.has_operations())
            .unwrap_or(false)
    }
}

// ============================================================================
// Compute Pass
// ============================================================================

/// Inner mutable state of a compute pass.
#[derive(Debug, Default)]
struct ComputePassInner {
    dependencies: Vec<Weak<Pass>>,
    // Future: compute-specific configuration (dispatch size, etc.)
}

/// A compute pass for compute shader work.
///
/// Compute passes execute compute shaders for general-purpose GPU computation.
#[derive(Debug)]
pub struct ComputePass {
    name: String,
    inner: RwLock<ComputePassInner>,
}

impl ComputePass {
    /// Create a new compute pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            inner: RwLock::new(ComputePassInner::default()),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pass dependencies.
    pub fn dependencies(&self) -> Vec<Arc<Pass>> {
        let inner = self.inner.read().unwrap();
        inner
            .dependencies
            .iter()
            .filter_map(|weak| weak.upgrade())
            .collect()
    }

    /// Add a dependency on another pass.
    pub fn add_dependency(&self, pass: &Arc<Pass>) {
        let mut inner = self.inner.write().unwrap();
        let already_exists = inner.dependencies.iter().any(|weak| {
            weak.upgrade()
                .map(|existing| Arc::ptr_eq(&existing, pass))
                .unwrap_or(false)
        });
        if !already_exists {
            inner.dependencies.push(Arc::downgrade(pass));
        }
    }
}
