//! Render pass types.

use std::sync::{Arc, RwLock, Weak};

use super::target::RenderTargetConfig;
use super::transfer::TransferConfig;

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

/// Inner mutable state of a render pass.
#[derive(Debug, Default)]
struct RenderPassInner {
    /// Passes that must execute before this one.
    dependencies: Vec<Weak<RenderPass>>,
    /// Render targets for graphics passes.
    render_targets: Option<RenderTargetConfig>,
    /// Transfer operations for transfer passes.
    transfer_config: Option<TransferConfig>,
}

/// A render pass in the graph.
///
/// Render passes describe a unit of GPU work with its resource dependencies.
/// For graphics passes, render targets can be configured to specify where
/// the pass renders to.
///
/// Passes are wrapped in `Arc<RenderPass>` and use interior mutability
/// via `RwLock` for thread-safe access.
#[derive(Debug)]
pub struct RenderPass {
    /// Debug name for the pass.
    name: String,
    /// Type of pass.
    pass_type: PassType,
    /// Mutable state protected by RwLock.
    inner: RwLock<RenderPassInner>,
}

impl RenderPass {
    /// Create a new render pass.
    pub fn new(name: String, pass_type: PassType) -> Self {
        Self {
            name,
            pass_type,
            inner: RwLock::new(RenderPassInner::default()),
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
    ///
    /// Returns only valid (still-alive) dependencies.
    pub fn dependencies(&self) -> Vec<Arc<RenderPass>> {
        let inner = self.inner.read().unwrap();
        inner
            .dependencies
            .iter()
            .filter_map(|weak| weak.upgrade())
            .collect()
    }

    /// Add a dependency on another pass.
    ///
    /// The dependency is stored as a weak reference to prevent cycles.
    pub fn add_dependency(&self, pass: &Arc<RenderPass>) {
        let mut inner = self.inner.write().unwrap();
        // Check if dependency already exists by comparing pointers
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
    ///
    /// This is only meaningful for graphics passes.
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

    /// Get the transfer configuration.
    pub fn transfer_config(&self) -> Option<TransferConfig> {
        let inner = self.inner.read().unwrap();
        inner.transfer_config.clone()
    }

    /// Set the transfer configuration.
    ///
    /// This is only meaningful for transfer passes.
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
