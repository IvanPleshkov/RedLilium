//! Backend-neutral views of immutable texture subresource ranges.
use super::GpuTextureView;
use crate::{Extent3d, GraphicsError, Texture, TextureDimension, TextureFormat};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureAspect {
    All,
    DepthOnly,
    StencilOnly,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}
impl From<TextureDimension> for TextureViewDimension {
    fn from(value: TextureDimension) -> Self {
        match value {
            TextureDimension::D1 => Self::D1,
            TextureDimension::D1Array => Self::D1Array,
            TextureDimension::D2 => Self::D2,
            TextureDimension::D2Array => Self::D2Array,
            TextureDimension::D3 => Self::D3,
            TextureDimension::Cube => Self::Cube,
            TextureDimension::CubeArray => Self::CubeArray,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Sint,
    Uint,
    Depth,
}
/// A resolved range; counts are positive and checked when creating a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSubresourceRange {
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
}
impl TextureSubresourceRange {
    pub fn whole(texture: &Texture) -> Self {
        Self {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: texture.mip_level_count(),
            base_array_layer: 0,
            array_layer_count: texture.descriptor().layers_and_depth().0,
        }
    }
    pub fn overlaps(self, other: Self) -> bool {
        (self.aspect == other.aspect
            || self.aspect == TextureAspect::All
            || other.aspect == TextureAspect::All)
            && self.overlaps_levels(other)
    }
    pub(crate) fn merge_adjacent(self, b: Self) -> Option<Self> {
        let mut a = self;
        if a.aspect != b.aspect {
            return None;
        }
        if a.base_mip_level == b.base_mip_level
            && a.mip_level_count == b.mip_level_count
            && (a.base_array_layer + a.array_layer_count == b.base_array_layer
                || b.base_array_layer + b.array_layer_count == a.base_array_layer)
        {
            a.base_array_layer = a.base_array_layer.min(b.base_array_layer);
            a.array_layer_count += b.array_layer_count;
            Some(a)
        } else if a.base_array_layer == b.base_array_layer
            && a.array_layer_count == b.array_layer_count
            && (a.base_mip_level + a.mip_level_count == b.base_mip_level
                || b.base_mip_level + b.mip_level_count == a.base_mip_level)
        {
            a.base_mip_level = a.base_mip_level.min(b.base_mip_level);
            a.mip_level_count += b.mip_level_count;
            Some(a)
        } else {
            None
        }
    }
    pub(crate) fn subtract(self, other: Self) -> Vec<Self> {
        if !self.overlaps_levels(other) {
            return vec![self];
        }
        let m0 = self.base_mip_level.max(other.base_mip_level);
        let m1 = (self.base_mip_level + self.mip_level_count)
            .min(other.base_mip_level + other.mip_level_count);
        let l0 = self.base_array_layer.max(other.base_array_layer);
        let l1 = (self.base_array_layer + self.array_layer_count)
            .min(other.base_array_layer + other.array_layer_count);
        [
            (
                self.base_mip_level,
                m0 - self.base_mip_level,
                self.base_array_layer,
                self.array_layer_count,
            ),
            (
                m1,
                self.base_mip_level + self.mip_level_count - m1,
                self.base_array_layer,
                self.array_layer_count,
            ),
            (
                m0,
                m1 - m0,
                self.base_array_layer,
                l0 - self.base_array_layer,
            ),
            (
                m0,
                m1 - m0,
                l1,
                self.base_array_layer + self.array_layer_count - l1,
            ),
        ]
        .into_iter()
        .filter(|&(_, mc, _, lc)| mc > 0 && lc > 0)
        .map(|(m, mc, l, lc)| Self {
            base_mip_level: m,
            mip_level_count: mc,
            base_array_layer: l,
            array_layer_count: lc,
            ..self
        })
        .collect()
    }
    pub(crate) fn overlaps_levels(self, other: Self) -> bool {
        self.base_mip_level < other.base_mip_level.saturating_add(other.mip_level_count)
            && other.base_mip_level < self.base_mip_level.saturating_add(self.mip_level_count)
            && self.base_array_layer
                < other
                    .base_array_layer
                    .saturating_add(other.array_layer_count)
            && other.base_array_layer < self.base_array_layer.saturating_add(self.array_layer_count)
    }
}
/// Selects a texture range without changing its format.
/// Optional counts default to the remaining mip levels or array layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureViewDescriptor {
    pub dimension: TextureViewDimension,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}
impl TextureViewDescriptor {
    pub fn new(dimension: TextureViewDimension) -> Self {
        Self {
            dimension,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        }
    }
    pub fn with_mip_levels(mut self, base: u32, count: u32) -> Self {
        self.base_mip_level = base;
        self.mip_level_count = Some(count);
        self
    }
    pub fn with_array_layers(mut self, base: u32, count: u32) -> Self {
        self.base_array_layer = base;
        self.array_layer_count = Some(count);
        self
    }
    pub fn with_aspect(mut self, aspect: TextureAspect) -> Self {
        self.aspect = aspect;
        self
    }
    pub(crate) fn resolve(self, t: &Texture) -> Result<Self, GraphicsError> {
        let invalid = || {
            GraphicsError::InvalidParameter(
                "texture view range, aspect or dimension is incompatible with texture".into(),
            )
        };
        let full = TextureSubresourceRange::whole(t);
        let mips = self
            .mip_level_count
            .unwrap_or(full.mip_level_count.saturating_sub(self.base_mip_level));
        let layers = self
            .array_layer_count
            .unwrap_or(full.array_layer_count.saturating_sub(self.base_array_layer));
        if mips == 0
            || layers == 0
            || self
                .base_mip_level
                .checked_add(mips)
                .is_none_or(|v| v > full.mip_level_count)
            || self
                .base_array_layer
                .checked_add(layers)
                .is_none_or(|v| v > full.array_layer_count)
        {
            return Err(invalid());
        }
        use TextureViewDimension::*;
        let valid = match (t.dimension(), self.dimension) {
            (TextureDimension::D1, D1) => true,
            (TextureDimension::D1Array, D1 | D1Array) => true,
            (TextureDimension::D3, D3) => true,
            (
                TextureDimension::D2
                | TextureDimension::D2Array
                | TextureDimension::Cube
                | TextureDimension::CubeArray,
                D2 | D2Array,
            ) => true,
            (TextureDimension::Cube | TextureDimension::CubeArray, Cube | CubeArray) => true,
            _ => false,
        };
        if !valid
            || (matches!(self.dimension, D1 | D2 | D3) && layers != 1)
            || (self.dimension == Cube && layers != 6)
            || (matches!(self.dimension, Cube | CubeArray)
                && (self.base_array_layer % 6 != 0 || layers % 6 != 0))
            || (t.sample_count() > 1 && (self.dimension != D2 || mips != 1))
            || (self.aspect == TextureAspect::DepthOnly && !t.format().is_depth_stencil())
            || (self.aspect == TextureAspect::StencilOnly && !t.format().has_stencil())
        {
            return Err(invalid());
        }
        Ok(Self {
            mip_level_count: Some(mips),
            array_layer_count: Some(layers),
            ..self
        })
    }
    pub(crate) fn range(self) -> TextureSubresourceRange {
        TextureSubresourceRange {
            aspect: self.aspect,
            base_mip_level: self.base_mip_level,
            mip_level_count: self.mip_level_count.unwrap(),
            base_array_layer: self.base_array_layer,
            array_layer_count: self.array_layer_count.unwrap(),
        }
    }
}
/// Immutable, validated view. Keeps its texture alive; textures cache views weakly.
pub struct TextureView {
    pub(crate) native: Arc<GpuTextureView>,
    descriptor: TextureViewDescriptor,
    texture: Arc<Texture>,
}
impl std::fmt::Debug for TextureView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureView")
            .field("descriptor", &self.descriptor)
            .field("texture", &self.texture)
            .finish()
    }
}
impl TextureView {
    pub(crate) fn new(
        texture: Arc<Texture>,
        descriptor: TextureViewDescriptor,
        native: Arc<GpuTextureView>,
    ) -> Self {
        Self {
            texture,
            descriptor,
            native,
        }
    }
    pub fn texture(&self) -> &Arc<Texture> {
        &self.texture
    }
    pub fn descriptor(&self) -> &TextureViewDescriptor {
        &self.descriptor
    }
    pub fn range(&self) -> TextureSubresourceRange {
        self.descriptor.range()
    }
    pub fn dimension(&self) -> TextureViewDimension {
        self.descriptor.dimension
    }
    pub fn format(&self) -> TextureFormat {
        self.texture.format()
    }
    pub fn size(&self) -> Extent3d {
        let shift = self.descriptor.base_mip_level;
        Extent3d::new_3d(
            (self.texture.width() >> shift).max(1),
            (self.texture.height() >> shift).max(1),
            if self.dimension() == TextureViewDimension::D3 {
                (self.texture.depth() >> shift).max(1)
            } else {
                self.range().array_layer_count
            },
        )
    }
    pub fn sample_count(&self) -> u32 {
        self.texture.sample_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    fn device() -> Arc<GraphicsDevice> {
        GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }
    #[test]
    fn ranges_dimensions_cache_and_lifetime() {
        let d = device();
        let t = d
            .create_texture(
                &TextureDescriptor {
                    dimension: TextureDimension::CubeArray,
                    size: Extent3d::new_3d(8, 8, 2),
                    ..TextureDescriptor::new_cube(
                        8,
                        TextureFormat::Rgba8Unorm,
                        TextureUsage::TEXTURE_BINDING,
                    )
                }
                .with_mip_levels(4),
            )
            .unwrap();
        let desc = TextureViewDescriptor::new(TextureViewDimension::Cube)
            .with_array_layers(6, 6)
            .with_mip_levels(1, 2);
        let a = d.create_texture_view(&t, &desc).unwrap();
        let b = d.create_texture_view(&t, &desc).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.size(), Extent3d::new_3d(4, 4, 6));
        for bad in [
            desc.with_array_layers(1, 6),
            desc.with_array_layers(6, 7),
            desc.with_mip_levels(4, 1),
            desc.with_mip_levels(0, 0),
            desc.with_aspect(TextureAspect::DepthOnly),
            TextureViewDescriptor::new(TextureViewDimension::D3),
        ] {
            assert!(d.create_texture_view(&t, &bad).is_err());
        }
        assert!(device().create_texture_view(&t, &desc).is_err());
        let weak = Arc::downgrade(&t);
        drop(t);
        assert!(weak.upgrade().is_some());
        drop(a);
        drop(b);
        assert!(
            weak.upgrade().is_none(),
            "view cache must not retain the texture through a cycle"
        );
    }
    #[test]
    fn subtraction_preserves_exact_uncovered_cells() {
        let full = TextureSubresourceRange {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: 4,
            base_array_layer: 0,
            array_layer_count: 6,
        };
        let cut = TextureSubresourceRange {
            base_mip_level: 1,
            mip_level_count: 2,
            base_array_layer: 2,
            array_layer_count: 2,
            ..full
        };
        let pieces = full.subtract(cut);
        for mip in 0..4 {
            for layer in 0..6 {
                let cell = TextureSubresourceRange {
                    base_mip_level: mip,
                    mip_level_count: 1,
                    base_array_layer: layer,
                    array_layer_count: 1,
                    ..full
                };
                assert_eq!(
                    pieces.iter().filter(|r| r.overlaps(cell)).count(),
                    usize::from(!cut.overlaps(cell))
                );
            }
        }
    }
}
