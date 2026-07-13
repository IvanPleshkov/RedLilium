//! The texture loader. `Asset = Texture` (the GPU resource), `Deps = ()`.
//!
//! `pipeline` composes stages per source:
//! - `File`  → `[read, decode, upload]` — an image file (png/jpg) decoded to
//!   RGBA8; the record's [`TextureSettings`] pick linear vs sRGB sampling.
//! - `Solid` → `[make, upload]` — a 1×1 constant-color texture, no IO. Shading
//!   model schemas use these as texture-slot defaults (white, flat normal),
//!   so a material with no texture assigned still binds something valid.
//!
//! Both end in the same upload stage: allocate the GPU texture and stage its
//! pixels as a `TransferOperation` (flushed through the frame graph by
//! `AssetGpuFlush` — never a synchronous write).

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetPath, AssetSource, AssetStage, Executor, GpuValue,
    Guid, LoadEnv, StageFuture,
};
use redlilium_core::sampler::{AddressMode, CpuSampler, FilterMode};
use redlilium_core::texture::{CpuTexture, TextureFormat};
use redlilium_graphics::{
    Extent3d, GraphicsDevice, Texture, TextureDescriptor, TextureUsage, TransferOperation,
};
use redlilium_vfs::Vfs;
use std::sync::Arc;

/// Identity of a texture asset. `File` resolves an image file from `guid` via
/// the DB; `Solid` is a 1×1 constant color (always linear — normal-map and
/// factor defaults must not be sRGB-decoded); `Virtual` is a texture whose
/// GPU resource is *published* at runtime by an engine system rather than
/// loaded — e.g. a camera's offscreen output (ADR-029). Serialized in material
/// properties and component `AssetRef`s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TextureSource {
    File(Guid),
    Solid([u8; 4]),
    /// Provided via `TextureManager::publish_virtual`, never via the loader.
    /// Until the producing system publishes it, the source stays unresolved
    /// (consumers keep waiting exactly like for a still-loading file).
    Virtual(Guid),
}

impl TextureSource {
    /// The 1×1 white texture — the usual default for color texture slots
    /// (sampling it is a no-op factor).
    pub const WHITE: Self = Self::Solid([255, 255, 255, 255]);
    /// The 1×1 flat normal (+Z) texture — the default for normal-map slots.
    pub const FLAT_NORMAL: Self = Self::Solid([128, 128, 255, 255]);
}

// A dropped/bare guid references the file-backed variant.
impl From<Guid> for TextureSource {
    fn from(guid: Guid) -> Self {
        Self::File(guid)
    }
}

impl AssetSource for TextureSource {
    fn file_guid(&self) -> Option<Guid> {
        match self {
            Self::File(guid) => Some(*guid),
            Self::Solid(_) | Self::Virtual(_) => None,
        }
    }
}

/// Per-record import settings for a texture asset (the DB record's `settings`,
/// RON). Defaults apply when the record has none; every field is
/// `#[serde(default)]` so partial records stay parseable as settings grow.
///
/// Sampling parameters live here too — a sampler is texture metadata, not an
/// asset of its own (it has no payload and only a handful of meaningful
/// combinations). The texture manager interns the resulting GPU samplers by
/// content, so textures sharing parameters share one `Arc<Sampler>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TextureSettings {
    /// Decode as sRGB (color data — base color, emissive). Linear (`false`)
    /// suits data textures: normal maps, roughness/metallic masks.
    #[serde(default = "default_srgb")]
    pub srgb: bool,
    /// Mag/min/mip filtering.
    #[serde(default = "default_filter")]
    pub filter: FilterMode,
    /// UV(W) wrapping.
    #[serde(default = "default_address")]
    pub address: AddressMode,
    /// Maximum anisotropy level (1 = off).
    #[serde(default = "default_anisotropy")]
    pub anisotropy: u16,
}

fn default_srgb() -> bool {
    true
}
fn default_filter() -> FilterMode {
    FilterMode::Linear
}
fn default_address() -> AddressMode {
    AddressMode::Repeat
}
fn default_anisotropy() -> u16 {
    1
}

impl Default for TextureSettings {
    fn default() -> Self {
        Self {
            srgb: default_srgb(),
            filter: default_filter(),
            address: default_address(),
            anisotropy: default_anisotropy(),
        }
    }
}

impl TextureSettings {
    /// The sampler these settings describe (interned by the texture manager).
    pub fn to_sampler(&self) -> CpuSampler {
        CpuSampler {
            mag_filter: self.filter,
            min_filter: self.filter,
            mipmap_filter: self.filter,
            anisotropy_clamp: self.anisotropy,
            ..CpuSampler::default()
        }
        .with_address_mode(self.address)
    }
}

/// Loads an image (file or solid color) and uploads it as a GPU [`Texture`].
pub struct TextureLoader;

impl AssetLoader for TextureLoader {
    const NAME: &'static str = "texture";
    const EXTENSIONS: &'static [&'static str] = &["png", "jpg", "jpeg"];
    type Source = TextureSource;
    type Asset = Texture;
    type Deps = ();

    fn pipeline(source: &TextureSource, _deps: &(), env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
        match source {
            TextureSource::File(_) => {
                if let Some(path) = &env.path {
                    stages.push(Box::new(ReadImageStage {
                        path: path.clone(),
                        vfs: env.vfs.clone(),
                    }));
                }
                let settings = env
                    .settings
                    .as_deref()
                    .and_then(|s| ron::from_str::<TextureSettings>(s).ok())
                    .unwrap_or_default();
                stages.push(Box::new(DecodeImageStage { settings }));
            }
            TextureSource::Solid(rgba) => {
                stages.push(Box::new(MakeSolidStage { rgba: *rgba }));
            }
            // Virtual textures are published by their producing system, never
            // loaded; the manager filters them out before requesting. An empty
            // pipeline (below) fails the request loudly if one slips through.
            TextureSource::Virtual(guid) => {
                log::error!("virtual texture {guid:?} reached the loader; it must be published");
                return Vec::new();
            }
        }
        stages.push(Box::new(UploadTextureStage {
            device: env.device.clone(),
        }));
        stages
    }
}

/// IO stage: read the image file's bytes.
struct ReadImageStage {
    path: AssetPath,
    vfs: Vfs,
}

impl AssetStage for ReadImageStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            let raw = format!("{}/{}", path.mount, path.path);
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(bytes) as AnyAsset)
        })
    }
}

/// CPU stage: decode the image to an RGBA8 [`CpuTexture`] (linear or sRGB per
/// the record settings).
struct DecodeImageStage {
    settings: TextureSettings,
}

impl AssetStage for DecodeImageStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        let srgb = self.settings.srgb;
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("texture: expected file bytes".into()))?;
            let img = image::load_from_memory(&bytes)
                .map_err(|e| AssetError::Decode(format!("texture: {e}")))?;
            let rgba = img.to_rgba8();
            let (width, height) = (img.width(), img.height());
            let format = if srgb {
                TextureFormat::Rgba8UnormSrgb
            } else {
                TextureFormat::Rgba8Unorm
            };
            let cpu = CpuTexture::new(width, height, format, rgba.into_raw());
            Ok(Box::new(cpu) as AnyAsset)
        })
    }
}

/// CPU stage: synthesize a 1×1 constant-color texture (always linear).
struct MakeSolidStage {
    rgba: [u8; 4],
}

impl AssetStage for MakeSolidStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let rgba = self.rgba;
        Box::pin(async move {
            let cpu = CpuTexture::new(1, 1, TextureFormat::Rgba8Unorm, rgba.to_vec());
            Ok(Box::new(cpu) as AnyAsset)
        })
    }
}

/// GPU stage: allocate the texture and stage its pixel upload through the
/// frame graph.
struct UploadTextureStage {
    device: Arc<GraphicsDevice>,
}

impl AssetStage for UploadTextureStage {
    fn executor(&self) -> Executor {
        Executor::Gpu
    }
    fn run_gpu(&self, input: AnyAsset) -> Result<(GpuValue, Vec<TransferOperation>), AssetError> {
        let cpu = *input
            .downcast::<CpuTexture>()
            .map_err(|_| AssetError::Decode("texture: upload stage expected CpuTexture".into()))?;
        let descriptor = TextureDescriptor {
            label: cpu.name.clone(),
            size: Extent3d::new_2d(cpu.width, cpu.height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: cpu.dimension,
            format: cpu.format,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            // Sampled asset textures stay EXCLUSIVE (keeps compression, #88);
            // upload graphs touching them are routed to the graphics queue.
            cross_queue: false,
        };
        let texture = self.device.create_texture(&descriptor)?;
        let op =
            TransferOperation::upload_texture_data(&self.device, Arc::clone(&texture), &cpu.data)?;
        Ok((Box::new(texture) as GpuValue, vec![op]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Partial settings RON (old records / hand-authored) parses with defaults
    /// filling the omitted fields — settings can grow without breaking records.
    #[test]
    fn settings_parse_with_defaults() {
        let s: TextureSettings = ron::from_str("(srgb:false)").expect("partial settings parse");
        assert!(!s.srgb);
        assert_eq!(s.filter, FilterMode::Linear);
        assert_eq!(s.address, AddressMode::Repeat);
        assert_eq!(s.anisotropy, 1);
    }

    /// The settings→sampler mapping applies filter to all three filters and the
    /// address mode to all three axes.
    #[test]
    fn settings_to_sampler() {
        let s = TextureSettings {
            filter: FilterMode::Nearest,
            address: AddressMode::ClampToEdge,
            anisotropy: 4,
            ..Default::default()
        };
        let cpu = s.to_sampler();
        assert_eq!(cpu.mag_filter, FilterMode::Nearest);
        assert_eq!(cpu.min_filter, FilterMode::Nearest);
        assert_eq!(cpu.mipmap_filter, FilterMode::Nearest);
        assert_eq!(cpu.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(cpu.address_mode_w, AddressMode::ClampToEdge);
        assert_eq!(cpu.anisotropy_clamp, 4);
    }
}
