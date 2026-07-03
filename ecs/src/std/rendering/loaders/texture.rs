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
use redlilium_core::texture::{CpuTexture, TextureFormat};
use redlilium_graphics::{
    Extent3d, GraphicsDevice, Texture, TextureDescriptor, TextureUsage, TransferOperation,
};
use redlilium_vfs::Vfs;
use std::sync::Arc;

/// Identity of a texture asset. `File` resolves an image file from `guid` via
/// the DB; `Solid` is a 1×1 constant color (always linear — normal-map and
/// factor defaults must not be sRGB-decoded). Serialized in material
/// properties and component `AssetRef`s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TextureSource {
    File(Guid),
    Solid([u8; 4]),
}

impl TextureSource {
    /// The 1×1 white texture — the usual default for color texture slots
    /// (sampling it is a no-op factor).
    pub const WHITE: Self = Self::Solid([255, 255, 255, 255]);
    /// The 1×1 flat normal (+Z) texture — the default for normal-map slots.
    pub const FLAT_NORMAL: Self = Self::Solid([128, 128, 255, 255]);
}

impl AssetSource for TextureSource {
    fn file_guid(&self) -> Option<Guid> {
        match self {
            Self::File(guid) => Some(*guid),
            Self::Solid(_) => None,
        }
    }
}

/// Per-record import settings for a texture asset (the DB record's `settings`,
/// RON). Defaults apply when the record has none.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextureSettings {
    /// Decode as sRGB (color data — base color, emissive). Linear (`false`)
    /// suits data textures: normal maps, roughness/metallic masks.
    pub srgb: bool,
}

impl Default for TextureSettings {
    fn default() -> Self {
        Self { srgb: true }
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
        };
        let texture = self.device.create_texture(&descriptor)?;
        let op =
            TransferOperation::upload_texture_data(&self.device, Arc::clone(&texture), &cpu.data)?;
        Ok((Box::new(texture) as GpuValue, vec![op]))
    }
}
