//! Render pass types.

use std::sync::Arc;

use crate::materials::BoundResource;
use crate::materials::MaterialInstance;
use crate::mesh::Mesh;
use crate::resources::Buffer;
use crate::types::ScissorRect;

use super::resource_usage::{
    BufferAccessMode, PassResourceUsage, SurfaceAccess, TextureAccessMode,
};
use super::target::{LoadOp, RenderTarget, RenderTargetConfig};
use super::transfer::{TransferConfig, TransferOperation};

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

    /// Infer resource usage from the pass configuration.
    ///
    /// This examines the pass's render targets, material bindings, and transfer
    /// operations to determine which textures are used and how.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        match self {
            Pass::Graphics(p) => p.infer_resource_usage(),
            Pass::Transfer(p) => p.infer_resource_usage(),
            Pass::Compute(p) => p.infer_resource_usage(),
        }
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
    /// Optional scissor rectangle for clipping.
    pub scissor_rect: Option<ScissorRect>,
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
            scissor_rect: None,
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

    /// Set the scissor rectangle for clipping.
    pub fn with_scissor_rect(mut self, rect: ScissorRect) -> Self {
        self.scissor_rect = Some(rect);
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
// Indirect Draw Command
// ============================================================================

/// An indirect draw command where draw parameters are read from a GPU buffer.
///
/// Indirect draw commands enable GPU-driven rendering where a compute shader
/// or transfer operation prepares the draw arguments. This is essential for:
///
/// - **GPU culling**: Compute shader determines which objects are visible
/// - **LOD selection**: GPU decides detail level based on distance
/// - **Particle systems**: GPU controls particle counts dynamically
/// - **Procedural geometry**: Draw counts determined at runtime on GPU
///
/// # Buffer Layout
///
/// The indirect buffer must contain draw arguments in the appropriate format:
/// - For non-indexed draws: [`DrawIndirectArgs`](crate::types::DrawIndirectArgs)
/// - For indexed draws: [`DrawIndexedIndirectArgs`](crate::types::DrawIndexedIndirectArgs)
///
/// # Multi-Draw Support
///
/// When `draw_count > 1`, multiple draw calls are issued from consecutive
/// entries in the indirect buffer, each offset by `stride` bytes.
///
/// # Example
///
/// ```ignore
/// use redlilium_graphics::{IndirectDrawCommand, BufferUsage};
///
/// // Single indirect draw
/// let cmd = IndirectDrawCommand::new(mesh, material, indirect_buffer);
///
/// // Multi-draw indirect (draw 100 objects from buffer)
/// let cmd = IndirectDrawCommand::new(mesh, material, indirect_buffer)
///     .with_draw_count(100)
///     .with_stride(20);  // stride for DrawIndexedIndirectArgs
/// ```
pub struct IndirectDrawCommand {
    /// The mesh to render.
    pub mesh: Arc<Mesh>,
    /// The material instance with bound resources.
    pub material: Arc<MaterialInstance>,
    /// Buffer containing indirect draw arguments.
    ///
    /// Must have [`BufferUsage::INDIRECT`](crate::types::BufferUsage::INDIRECT) flag.
    pub indirect_buffer: Arc<Buffer>,
    /// Byte offset into the indirect buffer where arguments begin.
    pub indirect_offset: u64,
    /// Number of draw calls to issue (for multi-draw indirect).
    ///
    /// When greater than 1, draw arguments are read from consecutive
    /// entries in the buffer, each separated by `stride` bytes.
    pub draw_count: u32,
    /// Stride between consecutive draw argument entries in bytes.
    ///
    /// Only relevant when `draw_count > 1`. For single draws, this is ignored.
    /// Must be at least the size of the draw argument struct.
    pub stride: u32,
    /// Whether this is an indexed draw (uses index buffer from mesh).
    pub indexed: bool,
}

impl IndirectDrawCommand {
    /// Create a new indirect draw command.
    ///
    /// Creates a single non-indexed indirect draw.
    /// Use builder methods to configure multi-draw or indexed mode.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh vertex layout is not compatible with the material's
    /// expected vertex layout.
    pub fn new(
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
    ) -> Self {
        #[cfg(debug_assertions)]
        DrawCommand::check_compatibility(&mesh, &material);

        #[cfg(debug_assertions)]
        Self::check_indirect_buffer(&indirect_buffer);

        Self {
            mesh,
            material,
            indirect_buffer,
            indirect_offset: 0,
            draw_count: 1,
            stride: 0,
            indexed: false,
        }
    }

    /// Create a new indexed indirect draw command.
    ///
    /// The mesh must have an index buffer, and arguments are read as
    /// [`DrawIndexedIndirectArgs`](crate::types::DrawIndexedIndirectArgs).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh has no index buffer, or if mesh/material are incompatible.
    pub fn new_indexed(
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
    ) -> Self {
        #[cfg(debug_assertions)]
        DrawCommand::check_compatibility(&mesh, &material);

        #[cfg(debug_assertions)]
        Self::check_indirect_buffer(&indirect_buffer);

        #[cfg(debug_assertions)]
        if !mesh.is_indexed() {
            panic!(
                "IndirectDrawCommand::new_indexed requires a mesh with an index buffer, \
                 but mesh '{}' has no indices",
                mesh.label().unwrap_or("unnamed")
            );
        }

        Self {
            mesh,
            material,
            indirect_buffer,
            indirect_offset: 0,
            draw_count: 1,
            stride: 0,
            indexed: true,
        }
    }

    /// Set the byte offset into the indirect buffer.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.indirect_offset = offset;
        self
    }

    /// Set the number of draw calls for multi-draw indirect.
    ///
    /// When greater than 1, also set `stride` to specify the distance
    /// between consecutive draw argument entries.
    pub fn with_draw_count(mut self, count: u32) -> Self {
        self.draw_count = count;
        self
    }

    /// Set the stride between consecutive draw argument entries.
    ///
    /// Only relevant when `draw_count > 1`.
    pub fn with_stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }

    /// Check that the buffer has INDIRECT usage flag.
    #[cfg(debug_assertions)]
    fn check_indirect_buffer(buffer: &Buffer) {
        use crate::types::BufferUsage;
        if !buffer.descriptor().usage.contains(BufferUsage::INDIRECT) {
            panic!(
                "Indirect draw buffer '{}' must have BufferUsage::INDIRECT flag",
                buffer.label().unwrap_or("unnamed")
            );
        }
    }
}

impl std::fmt::Debug for IndirectDrawCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndirectDrawCommand")
            .field("mesh", &self.mesh.label())
            .field("material", &self.material.label())
            .field("indirect_buffer", &self.indirect_buffer.label())
            .field("indirect_offset", &self.indirect_offset)
            .field("draw_count", &self.draw_count)
            .field("indexed", &self.indexed)
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
///
/// # Draw Commands
///
/// The pass supports two types of draw commands:
///
/// - **Direct draws** ([`DrawCommand`]): CPU specifies exact draw parameters
/// - **Indirect draws** ([`IndirectDrawCommand`]): GPU buffer contains draw parameters
///
/// Indirect draws enable GPU-driven rendering where a compute shader determines
/// what to draw, enabling efficient culling, LOD selection, and dynamic batching.
#[derive(Debug)]
pub struct GraphicsPass {
    name: String,
    render_targets: Option<RenderTargetConfig>,
    draw_commands: Vec<DrawCommand>,
    indirect_draw_commands: Vec<IndirectDrawCommand>,
}

impl GraphicsPass {
    /// Create a new graphics pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            render_targets: None,
            draw_commands: Vec::new(),
            indirect_draw_commands: Vec::new(),
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

    /// Add a draw command with a scissor rectangle for clipping.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the mesh and material are incompatible.
    pub fn add_draw_with_scissor(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        scissor_rect: ScissorRect,
    ) {
        self.draw_commands
            .push(DrawCommand::new(mesh, material).with_scissor_rect(scissor_rect));
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
        self.indirect_draw_commands.clear();
    }

    /// Check if this pass has any draw commands (direct or indirect).
    pub fn has_draws(&self) -> bool {
        !self.draw_commands.is_empty() || !self.indirect_draw_commands.is_empty()
    }

    // ========================================================================
    // Indirect Draw Commands
    // ========================================================================

    /// Add an indirect draw command.
    ///
    /// The draw parameters (vertex count, instance count, etc.) are read from
    /// the indirect buffer at runtime, enabling GPU-driven rendering.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh to render
    /// * `material` - The material instance with bound resources
    /// * `indirect_buffer` - Buffer containing [`DrawIndirectArgs`](crate::types::DrawIndirectArgs)
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if:
    /// - The mesh and material are incompatible
    /// - The buffer doesn't have `BufferUsage::INDIRECT` flag
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GPU culling: compute shader writes visible object count to buffer
    /// pass.add_draw_indirect(mesh, material, culled_indirect_buffer);
    /// ```
    pub fn add_draw_indirect(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
    ) {
        self.indirect_draw_commands
            .push(IndirectDrawCommand::new(mesh, material, indirect_buffer));
    }

    /// Add an indexed indirect draw command.
    ///
    /// Similar to `add_draw_indirect`, but uses the mesh's index buffer.
    /// The draw parameters are read as [`DrawIndexedIndirectArgs`](crate::types::DrawIndexedIndirectArgs).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if:
    /// - The mesh has no index buffer
    /// - The mesh and material are incompatible
    /// - The buffer doesn't have `BufferUsage::INDIRECT` flag
    pub fn add_draw_indexed_indirect(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
    ) {
        self.indirect_draw_commands
            .push(IndirectDrawCommand::new_indexed(
                mesh,
                material,
                indirect_buffer,
            ));
    }

    /// Add a multi-draw indirect command.
    ///
    /// Issues multiple draw calls from consecutive entries in the indirect buffer.
    /// Each entry is separated by `stride` bytes.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh to render
    /// * `material` - The material instance with bound resources
    /// * `indirect_buffer` - Buffer containing multiple [`DrawIndirectArgs`](crate::types::DrawIndirectArgs)
    /// * `draw_count` - Number of draw calls to issue
    /// * `stride` - Bytes between consecutive draw argument entries
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw 1000 objects with GPU-prepared parameters
    /// pass.add_multi_draw_indirect(
    ///     mesh,
    ///     material,
    ///     indirect_buffer,
    ///     1000,
    ///     DrawIndirectArgs::SIZE as u32,
    /// );
    /// ```
    pub fn add_multi_draw_indirect(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
        draw_count: u32,
        stride: u32,
    ) {
        self.indirect_draw_commands.push(
            IndirectDrawCommand::new(mesh, material, indirect_buffer)
                .with_draw_count(draw_count)
                .with_stride(stride),
        );
    }

    /// Add a multi-draw indexed indirect command.
    ///
    /// Issues multiple indexed draw calls from consecutive entries in the indirect buffer.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh to render (must have index buffer)
    /// * `material` - The material instance with bound resources
    /// * `indirect_buffer` - Buffer containing multiple [`DrawIndexedIndirectArgs`](crate::types::DrawIndexedIndirectArgs)
    /// * `draw_count` - Number of draw calls to issue
    /// * `stride` - Bytes between consecutive draw argument entries
    pub fn add_multi_draw_indexed_indirect(
        &mut self,
        mesh: Arc<Mesh>,
        material: Arc<MaterialInstance>,
        indirect_buffer: Arc<Buffer>,
        draw_count: u32,
        stride: u32,
    ) {
        self.indirect_draw_commands.push(
            IndirectDrawCommand::new_indexed(mesh, material, indirect_buffer)
                .with_draw_count(draw_count)
                .with_stride(stride),
        );
    }

    /// Add a pre-built indirect draw command.
    pub fn add_indirect_draw_command(&mut self, command: IndirectDrawCommand) {
        self.indirect_draw_commands.push(command);
    }

    /// Get all indirect draw commands.
    pub fn indirect_draw_commands(&self) -> &[IndirectDrawCommand] {
        &self.indirect_draw_commands
    }

    /// Get mutable access to indirect draw commands.
    pub fn indirect_draw_commands_mut(&mut self) -> &mut Vec<IndirectDrawCommand> {
        &mut self.indirect_draw_commands
    }

    /// Check if this pass has any indirect draw commands.
    pub fn has_indirect_draws(&self) -> bool {
        !self.indirect_draw_commands.is_empty()
    }

    /// Infer resource usage from the pass configuration.
    ///
    /// This examines render targets, material bindings, and indirect buffers
    /// to determine which textures and buffers are used and how.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        let mut usage = PassResourceUsage::new();

        // Infer from render targets
        if let Some(targets) = &self.render_targets {
            // Color attachments
            for color in &targets.color_attachments {
                match &color.target {
                    RenderTarget::Texture { texture, .. } => {
                        usage
                            .add_texture(Arc::clone(texture), TextureAccessMode::RenderTargetWrite);
                    }
                    RenderTarget::Surface { .. } => {
                        let access = if matches!(color.load_op, LoadOp::Load) {
                            SurfaceAccess::ReadWrite
                        } else {
                            SurfaceAccess::Write
                        };
                        usage.set_surface_access(access);
                    }
                }
                // Resolve targets are also written to
                if let Some(RenderTarget::Texture { texture, .. }) = &color.resolve_target {
                    usage.add_texture(Arc::clone(texture), TextureAccessMode::RenderTargetWrite);
                }
            }

            // Depth/stencil attachment
            if let Some(depth) = &targets.depth_stencil_attachment
                && let RenderTarget::Texture { texture, .. } = &depth.target
            {
                let access = if depth.depth_read_only && depth.stencil_read_only {
                    TextureAccessMode::DepthStencilReadOnly
                } else {
                    TextureAccessMode::DepthStencilWrite
                };
                usage.add_texture(Arc::clone(texture), access);
            }
        }

        // Infer from draw commands (textures in material bindings are sampled)
        for cmd in &self.draw_commands {
            Self::extract_material_textures(&cmd.material, &mut usage);
        }

        // Infer from indirect draw commands
        for cmd in &self.indirect_draw_commands {
            Self::extract_material_textures(&cmd.material, &mut usage);
            // Indirect buffers are read by the GPU for draw arguments
            usage.add_buffer(
                Arc::clone(&cmd.indirect_buffer),
                BufferAccessMode::IndirectRead,
            );
        }

        usage
    }

    /// Extract textures from a material instance's bindings.
    fn extract_material_textures(material: &MaterialInstance, usage: &mut PassResourceUsage) {
        for group in material.binding_groups() {
            for entry in &group.entries {
                match &entry.resource {
                    BoundResource::Texture(tex) => {
                        usage.add_texture(Arc::clone(tex), TextureAccessMode::ShaderRead);
                    }
                    BoundResource::CombinedTextureSampler { texture, .. } => {
                        usage.add_texture(Arc::clone(texture), TextureAccessMode::ShaderRead);
                    }
                    _ => {}
                }
            }
        }
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

    /// Infer resource usage from the transfer operations.
    ///
    /// This examines transfer operations to determine which textures and buffers
    /// are used as sources or destinations.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        let mut usage = PassResourceUsage::new();

        if let Some(config) = &self.transfer_config {
            for op in &config.operations {
                match op {
                    TransferOperation::TextureToBuffer { src, dst, .. } => {
                        usage.add_texture(Arc::clone(src), TextureAccessMode::TransferRead);
                        usage.add_buffer(Arc::clone(dst), BufferAccessMode::TransferWrite);
                    }
                    TransferOperation::BufferToTexture { src, dst, .. } => {
                        usage.add_buffer(Arc::clone(src), BufferAccessMode::TransferRead);
                        usage.add_texture(Arc::clone(dst), TextureAccessMode::TransferWrite);
                    }
                    TransferOperation::TextureToTexture { src, dst, .. } => {
                        usage.add_texture(Arc::clone(src), TextureAccessMode::TransferRead);
                        usage.add_texture(Arc::clone(dst), TextureAccessMode::TransferWrite);
                    }
                    TransferOperation::BufferToBuffer { src, dst, .. } => {
                        usage.add_buffer(Arc::clone(src), BufferAccessMode::TransferRead);
                        usage.add_buffer(Arc::clone(dst), BufferAccessMode::TransferWrite);
                    }
                }
            }
        }

        usage
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

    /// Infer resource usage from the compute pass.
    ///
    /// Currently returns empty usage as compute pass configuration
    /// is not yet implemented.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        // TODO: Extract usage from compute shader bindings when implemented
        PassResourceUsage::new()
    }
}
