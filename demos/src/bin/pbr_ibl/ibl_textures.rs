//! IBL (Image-Based Lighting) texture management.
//!
//! The IBL inputs are baked offline (`cargo run -p xtask -- bake-ibl`, #137)
//! into Zstd-supercompressed KTX2 under `std-assets/textures/ibl/` and
//! embedded here. Parsing and per-(mip, layer) upload go through the same
//! path the asset loader uses (#120): `parse_ktx2` → `CpuTexture` →
//! `TransferOperation::upload_texture_level` through the frame graph.

use std::sync::Arc;

use redlilium_core::profiling::profile_scope;
use redlilium_core::texture::ktx2::parse_ktx2;
use redlilium_graphics::{
    CpuSampler, CpuTexture, GraphicsDevice, Texture, TextureDescriptor, TextureDimension,
    TextureUsage, TransferConfig, TransferOperation,
};

const BRDF_LUT_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/brdf_lut.ktx2");
const IRRADIANCE_KTX2: &[u8] =
    include_bytes!("../../../../std-assets/textures/ibl/irradiance_cube.ktx2");
const PREFILTER_KTX2: &[u8] =
    include_bytes!("../../../../std-assets/textures/ibl/prefilter_cube.ktx2");
const SKY_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/sky_cube.ktx2");

/// IBL cubemaps, BRDF LUT, and the background sky cubemap.
pub struct IblTextures {
    pub irradiance_cubemap: Arc<Texture>,
    pub prefilter_cubemap: Arc<Texture>,
    pub brdf_lut: Arc<Texture>,
    pub sky_cubemap: Arc<Texture>,
    pub sampler: Arc<redlilium_graphics::Sampler>,
    /// Mip count of the sky cubemap (drives the skybox LOD controls).
    pub sky_mip_levels: u32,
    /// Per-(mip, layer) upload ops, staged once and consumed on the first frame.
    pending_ops: Vec<TransferOperation>,
}

impl IblTextures {
    /// Parse the baked KTX2 set and create GPU textures with staged uploads.
    pub fn create(device: &Arc<GraphicsDevice>) -> Self {
        profile_scope!("IblTextures::create");

        let brdf_cpu = parse_and_check(BRDF_LUT_KTX2, TextureDimension::D2, "ibl_brdf_lut");
        let irradiance_cpu =
            parse_and_check(IRRADIANCE_KTX2, TextureDimension::Cube, "ibl_irradiance");
        let prefilter_cpu =
            parse_and_check(PREFILTER_KTX2, TextureDimension::Cube, "ibl_prefilter");
        let sky_cpu = parse_and_check(SKY_KTX2, TextureDimension::Cube, "ibl_sky");
        let sky_mip_levels = sky_cpu.mip_level_count;

        let mut pending_ops = Vec::new();
        let brdf_lut = create_texture(device, &brdf_cpu, &mut pending_ops);
        let irradiance_cubemap = create_texture(device, &irradiance_cpu, &mut pending_ops);
        let prefilter_cubemap = create_texture(device, &prefilter_cpu, &mut pending_ops);
        let sky_cubemap = create_texture(device, &sky_cpu, &mut pending_ops);

        let sampler = device
            .create_sampler_from_cpu(&CpuSampler::linear().with_name("ibl_sampler"))
            .expect("Failed to create IBL sampler");

        log::info!("IBL resources created from baked KTX2 set");

        Self {
            irradiance_cubemap,
            prefilter_cubemap,
            brdf_lut,
            sky_cubemap,
            sampler,
            sky_mip_levels,
            pending_ops,
        }
    }

    /// If the first-frame IBL upload is pending, returns its transfer config.
    pub fn take_transfer_config(&mut self) -> Option<TransferConfig> {
        if self.pending_ops.is_empty() {
            return None;
        }
        let ops = std::mem::take(&mut self.pending_ops);
        log::info!("IBL textures upload config created");
        Some(TransferConfig::new().with_operations(ops))
    }
}

/// Parse a baked KTX2 blob and assert it has the expected dimension.
fn parse_and_check(bytes: &[u8], dimension: TextureDimension, name: &str) -> CpuTexture {
    let cpu = parse_ktx2(bytes)
        .unwrap_or_else(|e| panic!("baked IBL asset {name} failed to parse: {e}"))
        .with_name(name);
    assert_eq!(cpu.dimension, dimension, "baked IBL asset {name}");
    cpu
}

/// Create the GPU texture for a parsed [`CpuTexture`] and stage every
/// (mip, layer) image through the frame graph.
fn create_texture(
    device: &Arc<GraphicsDevice>,
    cpu: &CpuTexture,
    ops: &mut Vec<TransferOperation>,
) -> Arc<Texture> {
    let usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
    let descriptor = match cpu.dimension {
        TextureDimension::Cube => TextureDescriptor::new_cube(cpu.width, cpu.format, usage),
        _ => TextureDescriptor::new_2d(cpu.width, cpu.height, cpu.format, usage),
    }
    .with_mip_levels(cpu.mip_level_count)
    .with_label(cpu.name.as_deref().unwrap_or("ibl_texture"));
    let texture = device
        .create_texture(&descriptor)
        .unwrap_or_else(|e| panic!("create IBL texture {:?}: {e}", cpu.name));
    for mip in 0..cpu.mip_level_count {
        for layer in 0..cpu.layer_count() {
            ops.push(
                TransferOperation::upload_texture_level(
                    device,
                    Arc::clone(&texture),
                    mip,
                    layer,
                    &cpu.data[cpu.byte_range(mip, layer)],
                )
                .unwrap_or_else(|e| panic!("stage IBL upload {:?}: {e}", cpu.name)),
            );
        }
    }
    texture
}
