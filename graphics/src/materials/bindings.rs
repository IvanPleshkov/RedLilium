//! Binding layout definitions for materials.
//!
//! Bindings are organized by frequency to minimize GPU state changes during rendering.

/// Frequency at which a binding group is updated.
///
/// This determines which bind group slot (0, 1, 2) the bindings belong to,
/// enabling efficient batching by update frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingFrequency {
    /// Updated once per frame (bind group 0).
    /// Examples: camera matrices, global lighting, time uniforms.
    PerFrame,

    /// Updated once per material (bind group 1).
    /// Examples: material textures, material properties buffer.
    PerMaterial,

    /// Updated once per object/draw call (bind group 2).
    /// Examples: model matrix, object-specific properties.
    PerObject,
}

impl BindingFrequency {
    /// Get the bind group index for this frequency.
    pub fn group_index(&self) -> u32 {
        match self {
            Self::PerFrame => 0,
            Self::PerMaterial => 1,
            Self::PerObject => 2,
        }
    }
}

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

/// Describes the layout of bindings for a specific frequency group.
#[derive(Debug, Clone)]
pub struct BindingLayout {
    /// Which frequency group this layout belongs to.
    pub frequency: BindingFrequency,

    /// The binding entries in this layout.
    pub entries: Vec<BindingLayoutEntry>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl BindingLayout {
    /// Create a new binding layout for the given frequency.
    pub fn new(frequency: BindingFrequency) -> Self {
        Self {
            frequency,
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

    /// Get the bind group index for this layout.
    pub fn group_index(&self) -> u32 {
        self.frequency.group_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_frequency_indices() {
        assert_eq!(BindingFrequency::PerFrame.group_index(), 0);
        assert_eq!(BindingFrequency::PerMaterial.group_index(), 1);
        assert_eq!(BindingFrequency::PerObject.group_index(), 2);
    }

    #[test]
    fn test_binding_layout_builder() {
        let layout = BindingLayout::new(BindingFrequency::PerMaterial)
            .with_uniform_buffer(0)
            .with_texture(1)
            .with_sampler(2)
            .with_label("material_bindings");

        assert_eq!(layout.frequency, BindingFrequency::PerMaterial);
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
