//! Binding layout definitions for materials.
//!
//! Bindings describe what resources a shader expects. Layouts are shared via `Arc`
//! to enable efficient batching - the renderer can compare `Arc` pointers to group
//! draw calls that share the same binding layouts.

/// Type of resource that can be bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// Uniform buffer (read-only, small, frequently updated).
    UniformBuffer,

    /// Storage buffer (read-write, larger data).
    StorageBuffer,

    /// Sampled texture (for reading in shaders).
    Texture,

    /// Texture sampler.
    Sampler,

    /// Combined texture and sampler.
    CombinedTextureSampler,
}

/// Describes a single binding slot in a layout.
#[derive(Debug, Clone)]
pub struct BindingLayoutEntry {
    /// Binding index within the group.
    pub binding: u32,

    /// Type of resource expected at this binding.
    pub binding_type: BindingType,

    /// Shader stages that can access this binding.
    pub visibility: ShaderStageFlags,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl BindingLayoutEntry {
    /// Create a new binding layout entry.
    pub fn new(binding: u32, binding_type: BindingType) -> Self {
        Self {
            binding,
            binding_type,
            visibility: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
            label: None,
        }
    }

    /// Set the shader stage visibility.
    pub fn with_visibility(mut self, visibility: ShaderStageFlags) -> Self {
        self.visibility = visibility;
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

bitflags::bitflags! {
    /// Shader stages that can access a binding.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderStageFlags: u32 {
        /// Vertex shader stage.
        const VERTEX = 1 << 0;
        /// Fragment shader stage.
        const FRAGMENT = 1 << 1;
        /// Compute shader stage.
        const COMPUTE = 1 << 2;
    }
}

/// Describes the layout of bindings for a bind group.
///
/// Layouts are typically wrapped in `Arc` and shared between materials
/// to enable efficient batching by pointer comparison.
#[derive(Debug, Clone)]
pub struct BindingLayout {
    /// The binding entries in this layout.
    pub entries: Vec<BindingLayoutEntry>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl BindingLayout {
    /// Create a new empty binding layout.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            label: None,
        }
    }

    /// Add a binding entry to the layout.
    pub fn with_entry(mut self, entry: BindingLayoutEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Add a uniform buffer binding.
    pub fn with_uniform_buffer(self, binding: u32) -> Self {
        self.with_entry(BindingLayoutEntry::new(binding, BindingType::UniformBuffer))
    }

    /// Add a texture binding.
    pub fn with_texture(self, binding: u32) -> Self {
        self.with_entry(BindingLayoutEntry::new(binding, BindingType::Texture))
    }

    /// Add a sampler binding.
    pub fn with_sampler(self, binding: u32) -> Self {
        self.with_entry(BindingLayoutEntry::new(binding, BindingType::Sampler))
    }

    /// Add a combined texture+sampler binding.
    pub fn with_combined_texture_sampler(self, binding: u32) -> Self {
        self.with_entry(BindingLayoutEntry::new(
            binding,
            BindingType::CombinedTextureSampler,
        ))
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for BindingLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_layout_builder() {
        let layout = BindingLayout::new()
            .with_uniform_buffer(0)
            .with_texture(1)
            .with_sampler(2)
            .with_label("material_bindings");

        assert_eq!(layout.entries.len(), 3);
        assert_eq!(layout.label, Some("material_bindings".to_string()));
    }

    #[test]
    fn test_binding_entry_visibility() {
        let entry = BindingLayoutEntry::new(0, BindingType::UniformBuffer)
            .with_visibility(ShaderStageFlags::VERTEX);

        assert_eq!(entry.visibility, ShaderStageFlags::VERTEX);
        assert!(!entry.visibility.contains(ShaderStageFlags::FRAGMENT));
    }
}
