//! Render pass types.

use std::sync::Arc;

use std::collections::HashSet;

use crate::materials::{BindingType, BoundResource, MaterialInstance, SampledDepthLayout};
use crate::mesh::Mesh;
use crate::resources::{Blas, Buffer, CompactionCopy, Tlas};
use crate::types::{ScissorRect, Viewport};

use super::resource_usage::{
    BufferAccessMode, PassResourceUsage, SurfaceAccess, TextureAccessMode,
};
use super::target::{LoadOp, RenderTarget, RenderTargetConfig};
use super::transfer::{TransferConfig, TransferOperation};

/// A pass in the render graph.
///
/// Passes describe units of GPU work with their resource dependencies.
/// Each variant has its own configuration specific to that pass type.
// The size skew is deliberate: passes live in a pooled per-frame Vec
// (`RenderGraph` reuses capacity across frames), so boxing the large
// `GraphicsPass` variant would trade one add-time memcpy for a per-pass
// heap allocation every frame.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Pass {
    /// Graphics pass (vertex/fragment shaders, rasterization).
    Graphics(GraphicsPass),
    /// Transfer pass (copy operations).
    Transfer(TransferPass),
    /// Compute pass (compute shaders).
    Compute(ComputePass),
    /// Acceleration-structure build pass (BLAS/TLAS builds, #110).
    AccelerationStructureBuild(AccelerationStructureBuildPass),
}

impl Pass {
    /// Get the pass name.
    pub fn name(&self) -> &str {
        match self {
            Pass::Graphics(p) => p.name(),
            Pass::Transfer(p) => p.name(),
            Pass::Compute(p) => p.name(),
            Pass::AccelerationStructureBuild(p) => p.name(),
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

    /// Get this pass as an acceleration-structure build pass, if it is one.
    pub fn as_acceleration_structure_build(&self) -> Option<&AccelerationStructureBuildPass> {
        if let Pass::AccelerationStructureBuild(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Check if this is an acceleration-structure build pass.
    pub fn is_acceleration_structure_build(&self) -> bool {
        matches!(self, Pass::AccelerationStructureBuild(_))
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
            Pass::AccelerationStructureBuild(p) => p.infer_resource_usage(),
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
    /// Per-bind-group dynamic byte offsets, indexed by bind-group index.
    ///
    /// Each inner `Vec` holds the offsets for that group's
    /// [`DynamicUniformBuffer`](crate::BindingType::DynamicUniformBuffer)
    /// bindings, in binding order. Empty (or a missing group entry) means no
    /// dynamic offsets for that group. The caller supplies these because it
    /// knows the material layout; the backend does not introspect it.
    pub dynamic_offsets: Vec<Vec<u32>>,
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
            dynamic_offsets: Vec::new(),
        }
    }

    /// Set per-bind-group dynamic uniform offsets (see [`dynamic_offsets`]).
    ///
    /// [`dynamic_offsets`]: Self::dynamic_offsets
    pub fn with_dynamic_offsets(mut self, offsets: Vec<Vec<u32>>) -> Self {
        self.dynamic_offsets = offsets;
        self
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
        let expected_layout = material.material().vertex_layout();
        expected_layout.is_compatible_with(mesh.layout())
    }

    /// Check compatibility and panic with detailed message if incompatible.
    #[cfg(debug_assertions)]
    fn check_compatibility(mesh: &Mesh, material: &MaterialInstance) {
        let expected_layout = material.material().vertex_layout();
        if !expected_layout.is_compatible_with(mesh.layout()) {
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
// Mesh-Tasks Draw Command
// ============================================================================

/// A mesh-shading draw (#111): dispatches `group_count` task (or mesh, when
/// the material has no task stage) workgroups through a material whose
/// pipeline is task?+mesh+fragment. There is no [`Mesh`] — geometry lives in
/// storage buffers bound through the material instance, and the mesh shader
/// fetches it. Requires
/// [`DeviceCapabilities::mesh_shading`](crate::DeviceCapabilities::mesh_shading).
pub struct MeshTasksDrawCommand {
    /// The material instance with bound resources (meshlet buffers included).
    pub material: Arc<MaterialInstance>,
    /// Workgroup counts `[x, y, z]` for `vkCmdDrawMeshTasksEXT`.
    pub group_count: [u32; 3],
    /// Optional scissor rectangle for clipping.
    pub scissor_rect: Option<ScissorRect>,
    /// Per-bind-group dynamic byte offsets (see
    /// [`DrawCommand::dynamic_offsets`]).
    pub dynamic_offsets: Vec<Vec<u32>>,
}

impl MeshTasksDrawCommand {
    /// Create a new mesh-tasks draw command.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the material has no mesh stage — a classic vertex material
    /// cannot be dispatched as mesh tasks.
    pub fn new(material: Arc<MaterialInstance>, group_count: [u32; 3]) -> Self {
        #[cfg(debug_assertions)]
        {
            use crate::materials::ShaderStage;
            debug_assert!(
                material
                    .material()
                    .shaders()
                    .iter()
                    .any(|s| s.stage == ShaderStage::Mesh),
                "MeshTasksDrawCommand requires a material with a mesh shader stage \
                 (material {:?})",
                material.label()
            );
        }
        Self {
            material,
            group_count,
            scissor_rect: None,
            dynamic_offsets: Vec::new(),
        }
    }

    /// Set the scissor rectangle for clipping.
    pub fn with_scissor_rect(mut self, rect: ScissorRect) -> Self {
        self.scissor_rect = Some(rect);
        self
    }

    /// Set per-bind-group dynamic uniform offsets (see
    /// [`DrawCommand::dynamic_offsets`]).
    pub fn with_dynamic_offsets(mut self, offsets: Vec<Vec<u32>>) -> Self {
        self.dynamic_offsets = offsets;
        self
    }
}

impl std::fmt::Debug for MeshTasksDrawCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshTasksDrawCommand")
            .field("material", &self.material.label())
            .field("group_count", &self.group_count)
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
    viewport: Option<Viewport>,
    scissor_rect: Option<ScissorRect>,
    draw_commands: Vec<DrawCommand>,
    indirect_draw_commands: Vec<IndirectDrawCommand>,
    mesh_tasks_commands: Vec<MeshTasksDrawCommand>,
}

impl GraphicsPass {
    /// Create a new graphics pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            render_targets: None,
            viewport: None,
            scissor_rect: None,
            draw_commands: Vec::new(),
            indirect_draw_commands: Vec::new(),
            mesh_tasks_commands: Vec::new(),
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

    /// Get the pass-level viewport override.
    pub fn viewport(&self) -> Option<&Viewport> {
        self.viewport.as_ref()
    }

    /// Set a pass-level viewport override.
    ///
    /// When set, this viewport is used instead of the full render target dimensions.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = Some(viewport);
    }

    /// Get the pass-level scissor rect override.
    pub fn scissor_rect(&self) -> Option<&ScissorRect> {
        self.scissor_rect.as_ref()
    }

    /// Set a pass-level scissor rect override.
    ///
    /// When set, this scissor rect is used instead of the full render target dimensions.
    pub fn set_scissor_rect(&mut self, rect: ScissorRect) {
        self.scissor_rect = Some(rect);
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
        self.mesh_tasks_commands.clear();
    }

    /// Check if this pass has any draw commands (direct, indirect, or mesh
    /// tasks).
    pub fn has_draws(&self) -> bool {
        !self.draw_commands.is_empty()
            || !self.indirect_draw_commands.is_empty()
            || !self.mesh_tasks_commands.is_empty()
    }

    // ========================================================================
    // Mesh-Tasks Draw Commands (#111)
    // ========================================================================

    /// Add a mesh-shading draw (#111): dispatch `group_count` task/mesh
    /// workgroups through a task?+mesh+fragment material. Requires
    /// [`DeviceCapabilities::mesh_shading`](crate::DeviceCapabilities::mesh_shading);
    /// there is no [`Mesh`] — the mesh shader fetches geometry from storage
    /// buffers bound through the material instance.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the material has no mesh shader stage.
    pub fn add_draw_mesh_tasks(&mut self, material: Arc<MaterialInstance>, group_count: [u32; 3]) {
        self.mesh_tasks_commands
            .push(MeshTasksDrawCommand::new(material, group_count));
    }

    /// Add a pre-built mesh-tasks draw command.
    pub fn add_mesh_tasks_command(&mut self, command: MeshTasksDrawCommand) {
        self.mesh_tasks_commands.push(command);
    }

    /// Get all mesh-tasks draw commands.
    pub fn mesh_tasks_commands(&self) -> &[MeshTasksDrawCommand] {
        &self.mesh_tasks_commands
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
                let access = if depth.effective_read_only() {
                    TextureAccessMode::DepthStencilReadOnly
                } else {
                    TextureAccessMode::DepthStencilWrite
                };
                usage.add_texture(Arc::clone(texture), access);
            }
        }

        // Infer from draw commands: sampled textures and buffers in material
        // bindings, plus the mesh's vertex/index buffers (so GPU writes to
        // them — e.g. a WriteBuffer upload or a compute skinning pass — get a
        // barrier before vertex input reads them).
        let mut seen = BufferDeclSet::new();
        for cmd in &self.draw_commands {
            extract_material_resources(&cmd.material, &mut usage, &mut seen);
            declare_mesh_buffers(&cmd.mesh, &mut usage, &mut seen);
        }

        // Infer from indirect draw commands
        for cmd in &self.indirect_draw_commands {
            extract_material_resources(&cmd.material, &mut usage, &mut seen);
            declare_mesh_buffers(&cmd.mesh, &mut usage, &mut seen);
            // Indirect buffers are read by the GPU for draw arguments
            seen.add_buffer(
                &mut usage,
                &cmd.indirect_buffer,
                BufferAccessMode::IndirectRead,
            );
        }

        // Infer from mesh-tasks draws (#111): material resources only — the
        // meshlet geometry lives in storage buffers bound through the
        // material instance, so `extract_material_resources` covers it (a
        // compute pass writing those buffers gets its barrier from the
        // StorageRead/StorageReadWrite declarations).
        for cmd in &self.mesh_tasks_commands {
            extract_material_resources(&cmd.material, &mut usage, &mut seen);
        }

        // Diagnostic: a texture bound as a depth/stencil attachment AND sampled
        // through a *plain* (ShaderRead) group in the same pass is a layout
        // mistake — the barrier can pick only one layout, so the eagerly-written
        // descriptor (SHADER_READ_ONLY_OPTIMAL) mismatches the attachment. The
        // fix is `BindingGroupDescriptor::with_depth_texture_co_attached`, which
        // makes both agree on DEPTH_STENCIL_READ_ONLY_OPTIMAL. Attachment intent
        // still wins the transition (depth-test correctness); this names the
        // real cause the validation layer would otherwise only hint at.
        for (i, attach) in usage.texture_usages.iter().enumerate() {
            if !matches!(
                attach.access,
                TextureAccessMode::DepthStencilReadOnly | TextureAccessMode::DepthStencilWrite
            ) {
                continue;
            }
            let sampled_plain = usage.texture_usages.iter().enumerate().any(|(j, other)| {
                i != j
                    && other.access == TextureAccessMode::ShaderRead
                    && Arc::ptr_eq(&attach.texture, &other.texture)
            });
            if sampled_plain {
                log::error!(
                    "pass '{}': a depth/stencil attachment texture is also sampled through a \
                     plain binding group; use \
                     BindingGroupDescriptor::with_depth_texture_co_attached so the sampled \
                     descriptor records DEPTH_STENCIL_READ_ONLY_OPTIMAL",
                    self.name()
                );
            }
        }

        usage
    }
}

/// Per-pass dedup of buffer usage declarations.
///
/// Draw commands routinely share meshes and materials, so without dedup a
/// pass with N draws would declare the same buffers N times (N tracker
/// lookups per frame in the Vulkan backend).
struct BufferDeclSet {
    seen: HashSet<(*const Buffer, BufferAccessMode)>,
}

impl BufferDeclSet {
    fn new() -> Self {
        Self {
            seen: HashSet::new(),
        }
    }

    /// Add a buffer usage declaration unless an identical one exists.
    fn add_buffer(
        &mut self,
        usage: &mut PassResourceUsage,
        buffer: &Arc<Buffer>,
        access: BufferAccessMode,
    ) {
        if self.seen.insert((Arc::as_ptr(buffer), access)) {
            usage.add_buffer(Arc::clone(buffer), access);
        }
    }
}

/// Extract resource usages from a material instance's bindings.
///
/// Textures are declared as `ShaderRead`. Buffers are classified by the
/// material's binding layout: uniform bindings read, `StorageBufferReadOnly`
/// reads, `StorageBuffer` may be written by the shader and is declared
/// read-write — this is what lets the Vulkan backend place a barrier between
/// a compute pass writing a storage buffer and the pass that consumes it.
/// A buffer whose binding type cannot be determined (legacy material without
/// binding layouts) is conservatively treated as read-write.
fn extract_material_resources(
    material: &MaterialInstance,
    usage: &mut PassResourceUsage,
    seen: &mut BufferDeclSet,
) {
    let binding_layouts = material.material().binding_layouts();
    for (group_index, group) in material.binding_groups().iter().enumerate() {
        let layout = binding_layouts.get(group_index);
        for entry in group.entries() {
            // A depth texture co-used as a read-only depth attachment declares
            // the depth-read-only layout so its transition matches the
            // attachment (and its eagerly-written descriptor); everything else
            // samples in SHADER_READ_ONLY_OPTIMAL. Both derive from the same
            // creation-time `sampled_depth_layout`, so descriptor == barrier.
            let sampled_access = match entry.sampled_depth_layout {
                SampledDepthLayout::DepthStencilReadOnly => TextureAccessMode::DepthStencilReadOnly,
                SampledDepthLayout::ShaderReadOnly => TextureAccessMode::ShaderRead,
            };
            let buffer = match &entry.resource {
                BoundResource::Texture(tex) => {
                    usage.add_texture(Arc::clone(tex), sampled_access);
                    continue;
                }
                BoundResource::CombinedTextureSampler { texture, .. } => {
                    usage.add_texture(Arc::clone(texture), sampled_access);
                    continue;
                }
                BoundResource::Buffer(buffer) => buffer,
                BoundResource::BufferRange { buffer, .. } => buffer,
                // Ray traversal reads the TLAS memory AND the memory of every
                // BLAS its instances reference (#110) — declare all of them so
                // the build-to-trace hazard gets a barrier.
                BoundResource::AccelerationStructure(tlas) => {
                    seen.add_buffer(
                        usage,
                        tlas.backing_buffer(),
                        BufferAccessMode::AccelerationStructureShaderRead,
                    );
                    let (_instances, _count, blases) = tlas.current_build_inputs();
                    for blas in &blases {
                        seen.add_buffer(
                            usage,
                            &blas.backing_buffer(),
                            BufferAccessMode::AccelerationStructureShaderRead,
                        );
                    }
                    continue;
                }
                // The bindless heap (#117): every registered texture may be
                // sampled by this draw, so declare them all — that is what
                // keeps automatic layout transitions working (a freshly
                // uploaded texture moves TransferDst → ShaderReadOnly before
                // the first bindless draw).
                BoundResource::BindlessHeap(slots) => {
                    slots.for_each_live_texture(|texture| {
                        usage.add_texture(Arc::clone(texture), TextureAccessMode::ShaderRead);
                    });
                    continue;
                }
                _ => continue,
            };

            let binding_type = layout
                .and_then(|l| l.entries.iter().find(|e| e.binding == entry.binding))
                .map(|e| e.binding_type);
            let access = match binding_type {
                Some(BindingType::UniformBuffer | BindingType::DynamicUniformBuffer) => {
                    BufferAccessMode::UniformRead
                }
                Some(BindingType::StorageBufferReadOnly) => BufferAccessMode::StorageRead,
                // StorageBuffer shaders may write; unknown/missing layouts are
                // treated the same so they can never hide a hazard.
                _ => BufferAccessMode::StorageReadWrite,
            };
            seen.add_buffer(usage, buffer, access);
        }
    }
}

/// Declare a mesh's vertex and index buffers as pass inputs.
fn declare_mesh_buffers(mesh: &Mesh, usage: &mut PassResourceUsage, seen: &mut BufferDeclSet) {
    for buffer in mesh.vertex_buffers() {
        seen.add_buffer(usage, buffer, BufferAccessMode::VertexBuffer);
    }
    if let Some(index_buffer) = mesh.index_buffer() {
        seen.add_buffer(usage, index_buffer, BufferAccessMode::IndexBuffer);
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

    /// Whether any operation is legal only on a graphics-capable queue (#96):
    /// today that is `GenerateMipmaps` (`vkCmdBlitImage`).
    pub fn requires_graphics_queue(&self) -> bool {
        self.transfer_config.as_ref().is_some_and(|c| {
            c.operations
                .iter()
                .any(|op| matches!(op, TransferOperation::GenerateMipmaps { .. }))
        })
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
                    TransferOperation::WriteBuffer { dst, .. } => {
                        usage.add_buffer(Arc::clone(dst), BufferAccessMode::TransferWrite);
                    }
                    // Drained by the frame pipeline after the fence; no GPU work
                    // here, so no barrier. The GPU->src copy is a separate op.
                    TransferOperation::ReadbackBuffer { .. } => {}
                    // Whole-image TransferWrite (#96): the op both reads lower
                    // mips and writes higher ones, but the per-mip transitions
                    // are internal — for hazard purposes the write wins, and the
                    // op starts and ends in TRANSFER_DST so the whole-image
                    // tracker model stays truthful.
                    TransferOperation::GenerateMipmaps { texture } => {
                        usage.add_texture(Arc::clone(texture), TextureAccessMode::TransferWrite);
                    }
                }
            }
        }

        usage
    }
}

// ============================================================================
// Dispatch Command
// ============================================================================

/// A dispatch command for compute shader work.
///
/// Dispatch commands are submitted to compute passes to execute compute shaders.
pub struct DispatchCommand {
    /// The material instance with bound compute shader and resources.
    pub material: Arc<MaterialInstance>,
    /// Number of workgroups in the X dimension.
    pub workgroup_count_x: u32,
    /// Number of workgroups in the Y dimension.
    pub workgroup_count_y: u32,
    /// Number of workgroups in the Z dimension.
    pub workgroup_count_z: u32,
}

impl DispatchCommand {
    /// Create a new dispatch command.
    pub fn new(
        material: Arc<MaterialInstance>,
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        workgroup_count_z: u32,
    ) -> Self {
        Self {
            material,
            workgroup_count_x,
            workgroup_count_y,
            workgroup_count_z,
        }
    }
}

impl std::fmt::Debug for DispatchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchCommand")
            .field("material", &self.material.label())
            .field(
                "workgroups",
                &(
                    self.workgroup_count_x,
                    self.workgroup_count_y,
                    self.workgroup_count_z,
                ),
            )
            .finish()
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
    dispatch_commands: Vec<DispatchCommand>,
}

impl ComputePass {
    /// Create a new compute pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            dispatch_commands: Vec::new(),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a dispatch command to this pass.
    pub fn add_dispatch(
        &mut self,
        material: Arc<MaterialInstance>,
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        workgroup_count_z: u32,
    ) {
        self.dispatch_commands.push(DispatchCommand::new(
            material,
            workgroup_count_x,
            workgroup_count_y,
            workgroup_count_z,
        ));
    }

    /// Add a pre-built dispatch command.
    pub fn add_dispatch_command(&mut self, command: DispatchCommand) {
        self.dispatch_commands.push(command);
    }

    /// Get all dispatch commands.
    pub fn dispatch_commands(&self) -> &[DispatchCommand] {
        &self.dispatch_commands
    }

    /// Clear all dispatch commands.
    pub fn clear_dispatches(&mut self) {
        self.dispatch_commands.clear();
    }

    /// Check if this pass has any dispatch commands.
    pub fn has_dispatches(&self) -> bool {
        !self.dispatch_commands.is_empty()
    }

    /// Infer resource usage from the compute pass.
    ///
    /// Examines material bindings in dispatch commands to determine which
    /// textures and buffers are used. Storage buffers are declared read-write
    /// so writes performed by the dispatch are synchronized against
    /// subsequent passes that read them.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        let mut usage = PassResourceUsage::new();

        let mut seen = BufferDeclSet::new();
        for cmd in &self.dispatch_commands {
            extract_material_resources(&cmd.material, &mut usage, &mut seen);
        }

        usage
    }
}

// ============================================================================
// Acceleration Structure Build Pass
// ============================================================================

/// One build submitted to an [`AccelerationStructureBuildPass`] (#110).
#[derive(Debug)]
pub enum AccelerationStructureBuild {
    /// Build (or rebuild) a BLAS from its descriptor's geometry buffers.
    Blas(Arc<Blas>),
    /// Build (or rebuild) a TLAS from the instances most recently written via
    /// [`Tlas::write_instances`].
    Tlas(Arc<Tlas>),
    /// Copy a built BLAS into a smaller compacted structure (#110 phase 3).
    /// Engine-internal — produced by the maintenance flush, never the public
    /// build API.
    Compact(CompactionCopy),
}

/// A pass that builds acceleration structures on the GPU (#110, ADR-032).
///
/// All builds within one pass are **independent** — they may execute without
/// ordering between them. A build consuming another's output (the classic
/// BLAS build → TLAS build) must go in a **separate, dependent pass**: the
/// automatic barrier system works at pass granularity, deriving the
/// build-to-build and build-to-trace hazards from the AS backing buffers each
/// pass declares.
///
/// Runs on any compute-capable queue (graphics or async compute); a
/// [`QueuePreference::Transfer`](crate::QueuePreference) graph containing one
/// falls back — AS builds are not transfer work.
///
/// ```ignore
/// let blas_pass = {
///     let mut pass = AccelerationStructureBuildPass::new("blas".into());
///     pass.add_blas_build(Arc::clone(&blas));
///     graph.add_acceleration_structure_build_pass(pass)
/// };
/// tlas.write_instances(&instances)?;
/// let tlas_pass = {
///     let mut pass = AccelerationStructureBuildPass::new("tlas".into());
///     pass.add_tlas_build(Arc::clone(&tlas));
///     graph.add_acceleration_structure_build_pass(pass)
/// };
/// graph.add_dependency(tlas_pass, blas_pass);
/// ```
#[derive(Debug)]
pub struct AccelerationStructureBuildPass {
    name: String,
    builds: Vec<AccelerationStructureBuild>,
}

impl AccelerationStructureBuildPass {
    /// Create a new acceleration-structure build pass.
    pub fn new(name: String) -> Self {
        Self {
            name,
            builds: Vec::new(),
        }
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Queue a BLAS build.
    pub fn add_blas_build(&mut self, blas: Arc<Blas>) {
        self.builds.push(AccelerationStructureBuild::Blas(blas));
    }

    /// Queue a TLAS build. Write the frame's instances with
    /// [`Tlas::write_instances`] **before** the graph executes.
    pub fn add_tlas_build(&mut self, tlas: Arc<Tlas>) {
        self.builds.push(AccelerationStructureBuild::Tlas(tlas));
    }

    /// Queue a BLAS compaction copy (#110 phase 3). Engine-internal — the
    /// maintenance flush produces these; callers use
    /// [`BlasDescriptor::with_compaction`](crate::BlasDescriptor::with_compaction)
    /// and let compaction happen transparently.
    pub(crate) fn add_compaction_copy(&mut self, copy: CompactionCopy) {
        self.builds.push(AccelerationStructureBuild::Compact(copy));
    }

    /// Get all queued builds.
    pub fn builds(&self) -> &[AccelerationStructureBuild] {
        &self.builds
    }

    /// Check if this pass has any builds.
    pub fn has_builds(&self) -> bool {
        !self.builds.is_empty()
    }

    /// Infer resource usage for automatic barriers (#110).
    ///
    /// - BLAS build: geometry buffers as build **input**; backing + scratch
    ///   as build **write** (scratch is read-write within the build).
    /// - TLAS build: the active instance buffer as build input; referenced
    ///   BLAS backings as build **read**; backing + scratch as build write.
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        let mut usage = PassResourceUsage::new();
        let mut seen = BufferDeclSet::new();

        for build in &self.builds {
            match build {
                AccelerationStructureBuild::Blas(blas) => {
                    for geometry in &blas.descriptor().geometries {
                        seen.add_buffer(
                            &mut usage,
                            &geometry.vertex_buffer,
                            BufferAccessMode::AccelerationStructureBuildInput,
                        );
                        if let Some(index_buffer) = &geometry.index_buffer {
                            seen.add_buffer(
                                &mut usage,
                                index_buffer,
                                BufferAccessMode::AccelerationStructureBuildInput,
                            );
                        }
                    }
                    seen.add_buffer(
                        &mut usage,
                        &blas.backing_buffer(),
                        BufferAccessMode::AccelerationStructureWrite,
                    );
                    seen.add_buffer(
                        &mut usage,
                        blas.scratch_buffer(),
                        BufferAccessMode::AccelerationStructureWrite,
                    );
                }
                AccelerationStructureBuild::Tlas(tlas) => {
                    let (instance_buffer, _count, blases) = tlas.current_build_inputs();
                    seen.add_buffer(
                        &mut usage,
                        &instance_buffer,
                        BufferAccessMode::AccelerationStructureBuildInput,
                    );
                    for blas in &blases {
                        seen.add_buffer(
                            &mut usage,
                            &blas.backing_buffer(),
                            BufferAccessMode::AccelerationStructureBuildRead,
                        );
                    }
                    seen.add_buffer(
                        &mut usage,
                        tlas.backing_buffer(),
                        BufferAccessMode::AccelerationStructureWrite,
                    );
                    seen.add_buffer(
                        &mut usage,
                        tlas.scratch_buffer(),
                        BufferAccessMode::AccelerationStructureWrite,
                    );
                }
                AccelerationStructureBuild::Compact(copy) => {
                    // Compaction copy: read the original structure, write the
                    // freshly allocated compacted one. No scratch, no geometry.
                    seen.add_buffer(
                        &mut usage,
                        &copy.src_backing,
                        BufferAccessMode::AccelerationStructureBuildRead,
                    );
                    seen.add_buffer(
                        &mut usage,
                        &copy.dst_backing,
                        BufferAccessMode::AccelerationStructureWrite,
                    );
                }
            }
        }

        usage
    }
}
