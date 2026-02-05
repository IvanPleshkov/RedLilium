//! CPU-side sampler types and filter/address mode definitions.

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    /// Nearest neighbor filtering.
    #[default]
    Nearest,
    /// Linear filtering.
    Linear,
}

/// Texture address mode (wrapping behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    /// Clamp to edge.
    #[default]
    ClampToEdge,
    /// Repeat.
    Repeat,
    /// Mirrored repeat.
    MirrorRepeat,
    /// Clamp to border color.
    ClampToBorder,
}

/// Comparison function for depth/shadow sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// Never pass.
    Never,
    /// Pass if less than.
    Less,
    /// Pass if equal.
    Equal,
    /// Pass if less than or equal.
    LessEqual,
    /// Pass if greater than.
    Greater,
    /// Pass if not equal.
    NotEqual,
    /// Pass if greater than or equal.
    GreaterEqual,
    /// Always pass.
    Always,
}

/// CPU-side sampler configuration.
///
/// Describes how a texture is sampled: filtering, address modes, LOD clamping,
/// and optional comparison function. This is a format-agnostic descriptor
/// separate from any GPU resource.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuSampler {
    /// Sampler name.
    pub name: Option<String>,
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

impl CpuSampler {
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
        Self::default()
    }

    /// Set the sampler name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
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

impl Default for CpuSampler {
    fn default() -> Self {
        Self {
            name: None,
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
