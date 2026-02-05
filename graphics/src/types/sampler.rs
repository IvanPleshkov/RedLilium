//! Sampler types and descriptors.

// Re-export CPU-side types from core.
pub use redlilium_core::sampler::{AddressMode, CompareFunction, CpuSampler, FilterMode};

/// Descriptor for creating a sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerDescriptor {
    /// Debug label for the sampler.
    pub label: Option<String>,
    /// Address mode for U coordinate.
    pub address_mode_u: AddressMode,
    /// Address mode for V coordinate.
    pub address_mode_v: AddressMode,
    /// Address mode for W coordinate.
    pub address_mode_w: AddressMode,
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Mipmap filter.
    pub mipmap_filter: FilterMode,
    /// Minimum LOD clamp.
    pub lod_min_clamp: f32,
    /// Maximum LOD clamp.
    pub lod_max_clamp: f32,
    /// Comparison function for depth sampling.
    pub compare: Option<CompareFunction>,
    /// Maximum anisotropy level.
    pub anisotropy_clamp: u16,
}

impl SamplerDescriptor {
    /// Create a new sampler descriptor with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a linear filtering sampler.
    pub fn linear() -> Self {
        Self {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        }
    }

    /// Create a nearest neighbor filtering sampler.
    pub fn nearest() -> Self {
        Self {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        }
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set address mode for all coordinates.
    pub fn with_address_mode(mut self, mode: AddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    /// Set comparison function for depth sampling.
    pub fn with_compare(mut self, compare: CompareFunction) -> Self {
        self.compare = Some(compare);
        self
    }

    /// Set anisotropic filtering level.
    pub fn with_anisotropy(mut self, level: u16) -> Self {
        self.anisotropy_clamp = level;
        self
    }
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
        }
    }
}
