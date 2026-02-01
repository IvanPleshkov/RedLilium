//! Render pass types.

use std::sync::Arc;

use crate::materials::MaterialInstance;
use crate::mesh::Mesh;

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
// Draw Command
// ============================================================================

/// A draw command with mesh and material.
///
/// Draw commands are submitted to graphics passes to render geometry.
/// In debug builds, the mesh and material compatibility is verified.
pub struct DrawCommand {
    /// The mesh to render.
    pub mesh: Arc<Mesh>,
    /// The material instance with bound resources.
    pub material: Arc<MaterialInstance>,
    /// Number of instances to draw (default 1).
    pub instance_count: u32,
    /// First instance index (default 0).
    pub first_instance: u32,
}

impl DrawCommand {
    /// Create a new draw command.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh vertex layout is not compatible with the material's
    /// expected vertex layout.
    pub fn new(mesh: Arc<Mesh>, material: Arc<MaterialInstance>) -> Self {
        // Debug check: verify mesh/material compatibility
        #[cfg(debug_assertions)]
        Self::check_compatibility(&mesh, &material);

        Self {
            mesh,
            material,
            instance_count: 1,
            first_instance: 0,
        }
    }

    /// Set the number of instances to draw.
    pub fn with_instance_count(mut self, count: u32) -> Self {
        self.instance_count = count;
        self
    }

    /// Set the first instance index.
    pub fn with_first_instance(mut self, first: u32) -> Self {
        self.first_instance = first;
        self
    }

    /// Check if the mesh and material are compatible.
    ///
    /// In debug builds, this is called automatically by `new()`.
    /// Returns `true` if compatible, `false` otherwise.
    pub fn is_compatible(mesh: &Mesh, material: &MaterialInstance) -> bool {
        if let Some(expected_layout) = material.material().vertex_layout() {
            // The material expects a specific layout - check if mesh provides it
            expected_layout.is_compatible_with(mesh.layout())
        } else {
            // No expected layout specified - assume compatible
            true
        }
    }

    /// Check compatibility and panic with detailed message if incompatible.
    #[cfg(debug_assertions)]
    fn check_compatibility(mesh: &Mesh, material: &MaterialInstance) {
        if let Some(expected_layout) = material.material().vertex_layout()
            && !expected_layout.is_compatible_with(mesh.layout())
        {
            let mesh_semantics: Vec<_> = mesh
                .layout()
                .attributes
                .iter()
                .map(|a| format!("{:?}", a.semantic))
                .collect();
            let expected_semantics: Vec<_> = expected_layout
                .attributes
                .iter()
                .map(|a| format!("{:?}", a.semantic))
                .collect();

            panic!(
                "Mesh/Material incompatibility!\n\
                 Mesh '{}' layout: [{}]\n\
                 Material '{}' expects: [{}]\n\
                 The mesh must provide all attributes the material expects.",
                mesh.label().unwrap_or("unnamed"),
                mesh_semantics.join(", "),
                material.label().unwrap_or("unnamed"),
                expected_semantics.join(", ")
            );
        }
    }
}

impl std::fmt::Debug for DrawCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawCommand")
            .field("mesh", &self.mesh.label())
            .field("material", &self.material.label())
            .field("instance_count", &self.instance_count)
            .finish()
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
    render_targets: Option<RenderTargetConfig>,
    draw_commands: Vec<DrawCommand>,
}

impl GraphicsPass {
    /// Create a new graphics pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            render_targets: None,
            draw_commands: Vec::new(),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
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

    /// Add a draw command to this pass.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh and material are incompatible.
    pub fn add_draw(&mut self, mesh: Arc<Mesh>, material: Arc<MaterialInstance>) {
        self.draw_commands.push(DrawCommand::new(mesh, material));
    }

    /// Add a draw command with instancing.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh and material are incompatible.
    pub fn add_draw_instanced(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        instance_count: u32,
    ) {
        self.draw_commands
            .push(DrawCommand::new(mesh, material).with_instance_count(instance_count));
    }

    /// Add a pre-built draw command.
    pub fn add_draw_command(&mut self, command: DrawCommand) {
        self.draw_commands.push(command);
    }

    /// Get all draw commands.
    pub fn draw_commands(&self) -> &[DrawCommand] {
        &self.draw_commands
    }

    /// Get mutable access to draw commands.
    pub fn draw_commands_mut(&mut self) -> &mut Vec<DrawCommand> {
        &mut self.draw_commands
    }

    /// Clear all draw commands.
    pub fn clear_draws(&mut self) {
        self.draw_commands.clear();
    }

    /// Check if this pass has any draw commands.
    pub fn has_draws(&self) -> bool {
        !self.draw_commands.is_empty()
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
    transfer_config: Option<TransferConfig>,
}

impl TransferPass {
    /// Create a new transfer pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            transfer_config: None,
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
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
    // Future: compute-specific configuration (dispatch size, etc.)
}

impl ComputePass {
    /// Create a new compute pass.
    pub fn new(name: String) -> Self {
        Self { name }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }
}
