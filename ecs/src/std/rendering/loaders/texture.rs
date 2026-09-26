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
pub use redlilium_core::texture::MipmapFilter;
use redlilium_core::texture::{CpuTexture, TextureDimension, TextureFormat};
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
    /// A 1×1 constant-color **2D array** texture with the given layer count
    /// (always linear). The default for `TextureArray` material-property slots
    /// (`Texture2DArray` in the shader): a `Solid` produces a plain 2D view,
    /// which fails layout validation against a D2Array binding, so an array
    /// slot needs a source that resolves to a D2Array texture even when
    /// unassigned. `layers` must be ≥ 1.
    SolidArray([u8; 4], u32),
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
    /// A 1×1×1 white 2D array texture — the default for `TextureArray` slots
    /// (a single white layer, so an unassigned array samples as a no-op factor).
    pub const WHITE_ARRAY: Self = Self::SolidArray([255, 255, 255, 255], 1);
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
            Self::Solid(_) | Self::SolidArray(..) | Self::Virtual(_) => None,
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
    /// Generate a full mip chain at load time. Ordinary color uses GPU
    /// reduction when available; other supported formats use a CPU import stage.
    /// Compressed textures must supply their own chain.
    #[serde(default = "default_generate_mips")]
    pub generate_mips: bool,
    /// Content-aware CPU filtering. NormalMap also decodes ordinary images
    /// as linear data regardless of `srgb`. Stored container chains are preserved.
    #[serde(default)]
    pub mip_filter: MipmapFilter,
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
fn default_generate_mips() -> bool {
    true
}

impl Default for TextureSettings {
    fn default() -> Self {
        Self {
            srgb: default_srgb(),
            filter: default_filter(),
            address: default_address(),
            anisotropy: default_anisotropy(),
            generate_mips: default_generate_mips(),
            mip_filter: MipmapFilter::default(),
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
    const EXTENSIONS: &'static [&'static str] = &["png", "jpg", "jpeg", "ktx2"];
    type Source = TextureSource;
    type Asset = Texture;
    type Deps = ();

    fn pipeline(source: &TextureSource, _deps: &(), env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
        // Solid 1×1 defaults never need mips; only file textures opt in (#96).
        let mut generate_mips = false;
        let mut mip_filter = MipmapFilter::default();
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
                generate_mips = settings.generate_mips;
                mip_filter = settings.mip_filter;
                stages.push(Box::new(DecodeImageStage { settings }));
            }
            TextureSource::Solid(rgba) => {
                stages.push(Box::new(MakeSolidStage { rgba: *rgba }));
            }
            TextureSource::SolidArray(rgba, layers) => {
                stages.push(Box::new(MakeSolidArrayStage {
                    rgba: *rgba,
                    layers: (*layers).max(1),
                }));
            }
            // Virtual textures are published by their producing system, never
            // loaded; the manager filters them out before requesting. An empty
            // pipeline (below) fails the request loudly if one slips through.
            TextureSource::Virtual(guid) => {
                log::error!("virtual texture {guid:?} reached the loader; it must be published");
                return Vec::new();
            }
        }
        if generate_mips {
            stages.push(Box::new(PrepareMipmapsStage {
                device: env.device.clone(),
                filter: mip_filter,
            }));
        }
        stages.push(Box::new(UploadTextureStage {
            device: env.device.clone(),
            generate_mips,
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

/// CPU stage: decode the image to a [`CpuTexture`]. KTX2 containers (sniffed
/// by magic, #120) carry their own format/mips/layers and bypass the `image`
/// crate — including the record's `srgb` flag, which the container's vkFormat
/// already encodes. Everything else decodes to RGBA8 (linear or sRGB per the
/// record settings). A KTX2 whose format the engine cannot express fails with
/// a named error — never a silent RGBA8 fallback.
struct DecodeImageStage {
    settings: TextureSettings,
}

impl AssetStage for DecodeImageStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        let srgb = self.settings.srgb && self.settings.mip_filter != MipmapFilter::NormalMap;
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("texture: expected file bytes".into()))?;
            if redlilium_core::texture::ktx2::is_ktx2(&bytes) {
                let cpu = redlilium_core::texture::ktx2::parse_ktx2(&bytes)
                    .map_err(|e| AssetError::Decode(format!("texture: {e}")))?;
                return Ok(Box::new(cpu) as AnyAsset);
            }
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

/// CPU stage: synthesize a 1×1 constant-color 2D **array** texture with
/// `layers` layers (always linear) — the D2Array-shaped default for
/// `TextureArray` material-property slots. Each layer is the same color, so
/// sampling any layer of an unassigned slot is a no-op factor.
struct MakeSolidArrayStage {
    rgba: [u8; 4],
    layers: u32,
}

impl AssetStage for MakeSolidArrayStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let rgba = self.rgba;
        let layers = self.layers.max(1);
        Box::pin(async move {
            // One RGBA texel per layer, laid out layer-major (the upload stage's
            // `byte_range` walks (mip, layer) in this order).
            let mut data = Vec::with_capacity(layers as usize * 4);
            for _ in 0..layers {
                data.extend_from_slice(&rgba);
            }
            let cpu = CpuTexture::new(1, 1, TextureFormat::Rgba8Unorm, data)
                .with_dimension(TextureDimension::D2Array)
                .with_depth_or_array_layers(layers);
            Ok(Box::new(cpu) as AnyAsset)
        })
    }
}

/// GPU stage: allocate the texture and stage its pixel upload through the
/// frame graph.
struct UploadTextureStage {
    device: Arc<GraphicsDevice>,
    /// Whether to request GPU mips when the CPU stage has not supplied a chain.
    /// Still gated by device support and the 2D/array/cubemap restriction.
    generate_mips: bool,
}

/// CPU work stays on the asset worker, before the GPU upload stage. No GPU
/// readback is needed for coverage statistics or unsupported GPU formats.
struct PrepareMipmapsStage {
    device: Arc<GraphicsDevice>,
    filter: MipmapFilter,
}

impl AssetStage for PrepareMipmapsStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }

    fn run_async(&self, input: AnyAsset) -> StageFuture {
        let device = self.device.clone();
        let filter = self.filter;
        Box::pin(async move {
            let mut cpu = input
                .downcast::<CpuTexture>()
                .map_err(|_| AssetError::Decode("mip stage expected CpuTexture".into()))?;
            if cpu.mip_level_count == 1
                && matches!(
                    cpu.dimension,
                    TextureDimension::D2
                        | TextureDimension::D2Array
                        | TextureDimension::Cube
                        | TextureDimension::CubeArray
                )
                && (filter != MipmapFilter::Color || !device.supports_mipmap_generation(cpu.format))
            {
                if cpu.format.is_compressed() || cpu.format.is_depth_stencil() {
                    if filter != MipmapFilter::Color {
                        return Err(AssetError::Decode("content-aware mip filtering requires uncompressed color data or a precomputed chain".into()));
                    }
                } else {
                    cpu.generate_mipmaps(filter)
                        .map_err(|e| AssetError::Decode(e.to_string()))?;
                }
            }
            Ok(cpu as AnyAsset)
        })
    }
}

/// Full mip count for a `width × height` 2D image: `floor(log2(max)) + 1`.
fn full_mip_level_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

impl AssetStage for UploadTextureStage {
    fn executor(&self) -> Executor {
        Executor::Gpu
    }
    fn run_gpu(&self, input: AnyAsset) -> Result<(GpuValue, Vec<TransferOperation>), AssetError> {
        let cpu = *input
            .downcast::<CpuTexture>()
            .map_err(|_| AssetError::Decode("texture: upload stage expected CpuTexture".into()))?;

        // Stored mip chains are preserved. Having multiple layers alone does
        // not imply an authored chain: base-only arrays/cubes can generate mips.
        let stored_chain = cpu.mip_level_count > 1;
        let layered_upload =
            stored_chain || cpu.layer_count() > 1 || cpu.dimension == TextureDimension::D2Array;

        // CPU-generated and container-supplied chains arrive complete. GPU
        // generation handles 2D, arrays and cubemaps; unsupported compressed
        // formats stay as supplied and log once.
        let wants_mips = self.generate_mips
            && !stored_chain
            && matches!(
                cpu.dimension,
                TextureDimension::D2
                    | TextureDimension::D2Array
                    | TextureDimension::Cube
                    | TextureDimension::CubeArray
            )
            && (cpu.width > 1 || cpu.height > 1);
        let can_mip = wants_mips && self.device.supports_mipmap_generation(cpu.format);
        if wants_mips && !can_mip {
            log_mip_fallback(cpu.format);
        }
        let mip_level_count = if can_mip {
            full_mip_level_count(cpu.width, cpu.height)
        } else {
            cpu.mip_level_count
        };

        // Blit reads lower mips as TRANSFER_SRC, so the texture needs COPY_SRC
        // when a chain is generated.
        let mut usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
        if can_mip {
            usage |= TextureUsage::COPY_SRC;
        }

        // Arrays/3D carry their layer count / depth in `size.depth`
        // (`TextureDescriptor::layers_and_depth`); cubemaps get their 6 faces
        // from the dimension alone.
        let depth = match cpu.dimension {
            TextureDimension::D1 | TextureDimension::D2 | TextureDimension::Cube => 1,
            _ => cpu.depth_or_array_layers,
        };
        let descriptor = TextureDescriptor {
            label: cpu.name.clone(),
            size: Extent3d {
                width: cpu.width,
                height: cpu.height,
                depth,
            },
            mip_level_count,
            sample_count: 1,
            dimension: cpu.dimension,
            format: cpu.format,
            usage,
            // Declared cross-queue (#89): the asset-upload transfer graph is
            // routed to the dedicated transfer queue (DMA engines), which
            // requires the image to be legally accessible from both families
            // (CONCURRENT, or EXCLUSIVE under the maintenance9 implicit
            // fast path). On single-queue devices the flag is inert.
            cross_queue: true,
        };
        let texture = self.device.create_texture(&descriptor)?;
        let mut ops = Vec::new();
        if layered_upload {
            for mip in 0..cpu.mip_level_count {
                for layer in 0..cpu.layer_count() {
                    ops.push(TransferOperation::upload_texture_level(
                        &self.device,
                        Arc::clone(&texture),
                        mip,
                        layer,
                        &cpu.data[cpu.byte_range(mip, layer)],
                    )?);
                }
            }
        } else {
            ops.push(TransferOperation::upload_texture_data(
                &self.device,
                Arc::clone(&texture),
                &cpu.data,
            )?);
        }
        // Generate only after every base layer has been uploaded. flush_gpu
        // routes this op to the graphics queue, ordered after all uploads.
        if can_mip {
            ops.push(TransferOperation::generate_mipmaps(Arc::clone(&texture)));
        }
        Ok((Box::new(texture) as GpuValue, ops))
    }
}

/// Log once per format when no generation path is available.
fn log_mip_fallback(format: TextureFormat) {
    use std::collections::HashSet;
    use std::sync::{LazyLock, Mutex};
    static LOGGED: LazyLock<Mutex<HashSet<TextureFormat>>> =
        LazyLock::new(|| Mutex::new(HashSet::new()));
    if LOGGED.lock().unwrap().insert(format) {
        log::info!(
            "texture mip generation unavailable for {format:?}; supply a precomputed mip chain"
        );
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
        // Old records with no `generate_mips` field default to on (#96).
        assert!(s.generate_mips);
        assert_eq!(s.mip_filter, MipmapFilter::Color);
    }

    /// `floor(log2(max(w,h))) + 1` — the full 2D mip count.
    #[test]
    fn mip_level_count_matches_dimension() {
        assert_eq!(full_mip_level_count(1, 1), 1);
        assert_eq!(full_mip_level_count(4, 4), 3); // 4→2→1
        assert_eq!(full_mip_level_count(256, 256), 9);
        assert_eq!(full_mip_level_count(640, 480), 10); // driven by max
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

    // These CPU stages do not await IO. Poll once so a future that unexpectedly
    // starts depending on a reactor fails instead of hanging the test.
    fn ready(mut future: StageFuture) -> Result<AnyAsset, AssetError> {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => panic!("CPU stage unexpectedly pending"),
        }
    }

    fn dummy_device() -> Arc<GraphicsDevice> {
        redlilium_graphics::GraphicsInstance::with_parameters(
            redlilium_graphics::InstanceParameters::new()
                .with_backend(redlilium_graphics::BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }

    #[test]
    fn content_filter_settings_roundtrip() {
        for filter in [
            MipmapFilter::Color,
            MipmapFilter::NormalMap,
            MipmapFilter::AlphaCoverage { cutoff: 128 },
        ] {
            let settings = TextureSettings {
                mip_filter: filter,
                ..Default::default()
            };
            let encoded = ron::to_string(&settings).unwrap();
            assert_eq!(
                ron::from_str::<TextureSettings>(&encoded).unwrap(),
                settings
            );
        }
    }

    #[test]
    fn cpu_fallback_uploads_the_full_chain_through_graph_operations() {
        let device = dummy_device();
        for filter in [
            MipmapFilter::Color,
            MipmapFilter::NormalMap,
            MipmapFilter::AlphaCoverage { cutoff: 128 },
        ] {
            let stage = PrepareMipmapsStage {
                device: device.clone(),
                filter,
            };
            let cpu = CpuTexture::new(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                [128, 128, 255, 255].repeat(16),
            );
            let prepared = ready(stage.run_async(Box::new(cpu))).unwrap();
            let cpu = prepared.downcast_ref::<CpuTexture>().unwrap();
            assert_eq!(cpu.mip_level_count, 3);
            let upload = UploadTextureStage {
                device: device.clone(),
                generate_mips: true,
            };
            let (value, operations) = upload.run_gpu(prepared).unwrap();
            assert_eq!(
                value
                    .downcast_ref::<Arc<Texture>>()
                    .unwrap()
                    .mip_level_count(),
                3
            );
            assert_eq!(operations.len(), 3);
            assert!(
                operations
                    .iter()
                    .all(|op| matches!(op, TransferOperation::BufferToTexture { .. }))
            );
        }
    }

    #[test]
    fn normal_setting_decodes_color_files_as_linear_data() {
        let img =
            image::RgbaImage::from_raw(2, 1, vec![255, 128, 128, 255, 128, 255, 128, 255]).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let decode = DecodeImageStage {
            settings: TextureSettings {
                mip_filter: MipmapFilter::NormalMap,
                ..Default::default()
            },
        };
        let cpu = ready(decode.run_async(Box::new(png.into_inner())))
            .unwrap()
            .downcast::<CpuTexture>()
            .unwrap();
        assert_eq!(cpu.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn cpu_stage_keeps_container_chains_and_does_not_generate_volume_mips() {
        let device = dummy_device();
        let stage = PrepareMipmapsStage {
            device: device.clone(),
            filter: MipmapFilter::NormalMap,
        };
        let mut cpu = CpuTexture::new(2, 2, TextureFormat::Rgba8UnormSrgb, vec![42; 20]);
        cpu.mip_level_count = 2;
        let cpu = ready(stage.run_async(Box::new(cpu)))
            .unwrap()
            .downcast::<CpuTexture>()
            .unwrap();
        assert_eq!(cpu.data, vec![42; 20]);
        let cpu = CpuTexture::new(2, 2, TextureFormat::Rgba8Unorm, vec![42; 16 * 6])
            .with_dimension(TextureDimension::D3)
            .with_depth_or_array_layers(6);
        let cpu = ready(stage.run_async(Box::new(cpu))).unwrap();
        let upload = UploadTextureStage {
            device,
            generate_mips: true,
        };
        let (value, operations) = upload.run_gpu(cpu).unwrap();
        assert_eq!(
            value
                .downcast_ref::<Arc<Texture>>()
                .unwrap()
                .mip_level_count(),
            1
        );
        assert!(
            !operations
                .iter()
                .any(|op| matches!(op, TransferOperation::GenerateMipmaps { .. }))
        );
    }

    #[test]
    fn layered_cpu_import_uploads_each_mip_and_layer_and_preserves_authored_chain() {
        let device = dummy_device();
        for (dimension, count) in [
            (TextureDimension::D2Array, 1),
            (TextureDimension::D2Array, 3),
            (TextureDimension::Cube, 1),
            (TextureDimension::CubeArray, 2),
        ] {
            let layers = dimension.layer_count(count);
            for filter in [
                MipmapFilter::Color,
                MipmapFilter::NormalMap,
                MipmapFilter::AlphaCoverage { cutoff: 128 },
            ] {
                let stage = PrepareMipmapsStage {
                    device: device.clone(),
                    filter,
                };
                let cpu = CpuTexture::new(
                    4,
                    4,
                    TextureFormat::Rgba8Unorm,
                    [128, 128, 255, 255].repeat(16 * layers as usize),
                )
                .with_dimension(dimension)
                .with_depth_or_array_layers(count);
                let prepared = ready(stage.run_async(Box::new(cpu))).unwrap();
                let cpu = prepared.downcast_ref::<CpuTexture>().unwrap();
                assert_eq!(cpu.mip_level_count, 3);
                let authored = cpu.data.clone();
                // A stored chain passes through without a second reduction.
                let prepared = ready(stage.run_async(prepared)).unwrap();
                assert_eq!(
                    prepared.downcast_ref::<CpuTexture>().unwrap().data,
                    authored
                );
                let upload = UploadTextureStage {
                    device: device.clone(),
                    generate_mips: true,
                };
                let (value, operations) = upload.run_gpu(prepared).unwrap();
                let texture = value.downcast_ref::<Arc<Texture>>().unwrap();
                assert_eq!(texture.mip_level_count(), 3);
                assert_eq!(texture.dimension(), dimension);
                assert_eq!(texture.depth(), count);
                assert_eq!(operations.len(), (3 * layers) as usize);
                for (index, op) in operations.iter().enumerate() {
                    let TransferOperation::BufferToTexture { regions, .. } = op else {
                        panic!("expected level upload")
                    };
                    assert_eq!(regions[0].texture_location.mip_level, index as u32 / layers);
                    assert_eq!(regions[0].texture_location.origin.z, index as u32 % layers);
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_layered_import_generates_after_all_base_layers_and_honors_opt_out() {
        let Ok(instance) = redlilium_graphics::GraphicsInstance::with_parameters(
            redlilium_graphics::InstanceParameters::new()
                .with_backend(redlilium_graphics::BackendType::Wgpu),
        ) else {
            eprintln!("wgpu unavailable");
            return;
        };
        let device = instance.create_device().unwrap();
        assert!(device.supports_mipmap_generation(TextureFormat::Rgba8Unorm));
        for (dimension, count) in [
            (TextureDimension::D2Array, 1),
            (TextureDimension::D2Array, 3),
            (TextureDimension::Cube, 1),
            (TextureDimension::CubeArray, 2),
        ] {
            let layers = dimension.layer_count(count);
            for generate_mips in [false, true] {
                let cpu = CpuTexture::new(
                    4,
                    4,
                    TextureFormat::Rgba8Unorm,
                    vec![42; 64 * layers as usize],
                )
                .with_dimension(dimension)
                .with_depth_or_array_layers(count);
                let input = if generate_mips {
                    let stage = PrepareMipmapsStage {
                        device: device.clone(),
                        filter: MipmapFilter::Color,
                    };
                    let input = ready(stage.run_async(Box::new(cpu))).unwrap();
                    assert_eq!(
                        input.downcast_ref::<CpuTexture>().unwrap().mip_level_count,
                        1
                    );
                    input
                } else {
                    Box::new(cpu) as AnyAsset
                };
                let upload = UploadTextureStage {
                    device: device.clone(),
                    generate_mips,
                };
                let (value, operations) = upload.run_gpu(input).unwrap();
                let texture = value.downcast_ref::<Arc<Texture>>().unwrap();
                assert_eq!(texture.mip_level_count(), if generate_mips { 3 } else { 1 });
                assert_eq!(
                    operations.len(),
                    layers as usize + usize::from(generate_mips)
                );
                for layer in 0..layers {
                    let TransferOperation::BufferToTexture { dst, regions, .. } =
                        &operations[layer as usize]
                    else {
                        panic!("expected layer upload")
                    };
                    assert!(Arc::ptr_eq(dst, texture));
                    assert_eq!(regions[0].texture_location.mip_level, 0);
                    assert_eq!(regions[0].texture_location.origin.z, layer);
                }
                if generate_mips {
                    let Some(TransferOperation::GenerateMipmaps { texture: generated }) =
                        operations.last()
                    else {
                        panic!("expected mip generation after all uploads")
                    };
                    assert!(Arc::ptr_eq(generated, texture));
                }
            }
        }
    }
}
