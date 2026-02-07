//! IBL (Image-Based Lighting) texture management.

use std::sync::Arc;

use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    BufferDescriptor, BufferTextureCopyRegion, BufferTextureLayout, BufferUsage, CpuSampler,
    Extent3d, GraphicsDevice, TextureCopyLocation, TextureDescriptor, TextureFormat, TextureOrigin,
    TextureUsage, TransferConfig, TransferOperation,
};

use crate::ibl::compute_ibl_cpu;
use crate::resources::{BRDF_LUT_URL, HDR_URL, load_brdf_lut_from_url, load_hdr_from_url};
use crate::{IRRADIANCE_SIZE, PREFILTER_SIZE};

/// IBL cubemap textures, BRDF LUT, and staging buffers for GPU upload.
pub struct IblTextures {
    pub irradiance_cubemap: Arc<redlilium_graphics::Texture>,
    pub prefilter_cubemap: Arc<redlilium_graphics::Texture>,
    pub brdf_lut: Arc<redlilium_graphics::Texture>,
    pub sampler: Arc<redlilium_graphics::Sampler>,
    // Staging state for first-frame upload
    irradiance_staging: Option<Arc<redlilium_graphics::Buffer>>,
    prefilter_staging: Option<Vec<Arc<redlilium_graphics::Buffer>>>,
    prefilter_aligned_bytes_per_row: Vec<u32>,
    needs_upload: bool,
}

impl IblTextures {
    /// Load HDR environment, compute IBL cubemaps on CPU, and create GPU textures.
    pub fn create(device: &Arc<GraphicsDevice>) -> Self {
        profile_scope!("IblTextures::create");

        // Load HDR environment and compute IBL data on CPU
        log::info!("Loading HDR environment map...");
        let (hdr_width, hdr_height, hdr_data) =
            load_hdr_from_url(HDR_URL).expect("Failed to load HDR texture");

        log::info!("Computing IBL cubemaps on CPU...");
        let (irradiance_data, prefilter_data) = compute_ibl_cpu(&hdr_data, hdr_width, hdr_height);

        // Load BRDF LUT
        let brdf_cpu = load_brdf_lut_from_url(BRDF_LUT_URL).expect("Failed to load BRDF LUT");

        // Create IBL textures
        let irradiance_cubemap = device
            .create_texture(
                &TextureDescriptor::new_cube(
                    IRRADIANCE_SIZE,
                    TextureFormat::Rgba16Float,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label("irradiance_cubemap"),
            )
            .expect("Failed to create irradiance cubemap");

        let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;
        let prefilter_cubemap = device
            .create_texture(
                &TextureDescriptor::new_cube(
                    PREFILTER_SIZE,
                    TextureFormat::Rgba16Float,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_mip_levels(mip_levels)
                .with_label("prefilter_cubemap"),
            )
            .expect("Failed to create prefilter cubemap");

        let brdf_lut = device
            .create_texture_from_cpu(&brdf_cpu)
            .expect("Failed to create BRDF LUT");

        // Create staging buffers for IBL data upload
        let irradiance_bytes: &[u8] = bytemuck::cast_slice(&irradiance_data);
        let irradiance_staging = device
            .create_buffer(&BufferDescriptor::new(
                irradiance_bytes.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create irradiance staging buffer");
        device
            .write_buffer(&irradiance_staging, 0, irradiance_bytes)
            .expect("Failed to write irradiance staging buffer");

        // Create staging buffers for each mip level with aligned bytes per row
        const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
        let bytes_per_pixel = 8u32; // Rgba16Float = 4 channels * 2 bytes
        let mut prefilter_staging_buffers = Vec::new();
        let mut prefilter_aligned_bytes_per_row = Vec::new();

        for (mip, mip_data) in prefilter_data.iter().enumerate() {
            let mip_size = (PREFILTER_SIZE >> mip).max(1);
            let bytes_per_row = mip_size * bytes_per_pixel;
            let aligned_bytes_per_row =
                bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
            prefilter_aligned_bytes_per_row.push(aligned_bytes_per_row);

            let bytes: &[u8] = bytemuck::cast_slice(mip_data);

            // Pad data if alignment is needed
            let padded_data = if aligned_bytes_per_row != bytes_per_row {
                let face_size = (mip_size * mip_size) as usize * bytes_per_pixel as usize;
                let padded_face_size = (aligned_bytes_per_row * mip_size) as usize;
                let mut padded = vec![0u8; padded_face_size * 6];
                for face in 0..6 {
                    for y in 0..mip_size {
                        let src_start = face * face_size + (y as usize * bytes_per_row as usize);
                        let src_end = src_start + bytes_per_row as usize;
                        let dst_start =
                            face * padded_face_size + (y as usize * aligned_bytes_per_row as usize);
                        padded[dst_start..dst_start + bytes_per_row as usize]
                            .copy_from_slice(&bytes[src_start..src_end]);
                    }
                }
                padded
            } else {
                bytes.to_vec()
            };

            let buffer = device
                .create_buffer(&BufferDescriptor::new(
                    padded_data.len() as u64,
                    BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                ))
                .expect("Failed to create prefilter staging buffer");
            device
                .write_buffer(&buffer, 0, &padded_data)
                .expect("Failed to write prefilter staging buffer");
            prefilter_staging_buffers.push(buffer);
        }

        // Create IBL sampler
        let sampler = device
            .create_sampler_from_cpu(&CpuSampler::linear().with_name("ibl_sampler"))
            .expect("Failed to create IBL sampler");

        log::info!("IBL resources created successfully");

        Self {
            irradiance_cubemap,
            prefilter_cubemap,
            brdf_lut,
            sampler,
            irradiance_staging: Some(irradiance_staging),
            prefilter_staging: Some(prefilter_staging_buffers),
            prefilter_aligned_bytes_per_row,
            needs_upload: true,
        }
    }

    /// If an IBL upload is pending, returns the transfer config and clears the staging state.
    pub fn take_transfer_config(&mut self) -> Option<TransferConfig> {
        if !self.needs_upload {
            return None;
        }
        self.needs_upload = false;

        let mut config = TransferConfig::new();

        // Upload irradiance cubemap (6 faces)
        if let Some(staging) = &self.irradiance_staging {
            let face_bytes = (IRRADIANCE_SIZE * IRRADIANCE_SIZE * 4 * 2) as u64;
            for face in 0..6u32 {
                let region = BufferTextureCopyRegion::new(
                    BufferTextureLayout::new(
                        face as u64 * face_bytes,
                        Some(IRRADIANCE_SIZE * 4 * 2),
                        None,
                    ),
                    TextureCopyLocation::new(0, TextureOrigin::new(0, 0, face)),
                    Extent3d::new_2d(IRRADIANCE_SIZE, IRRADIANCE_SIZE),
                );
                config = config.with_operation(TransferOperation::upload_texture(
                    staging.clone(),
                    self.irradiance_cubemap.clone(),
                    vec![region],
                ));
            }
        }

        // Upload prefilter cubemap (all mip levels, 6 faces each)
        if let Some(staging_buffers) = &self.prefilter_staging {
            for (mip, staging) in staging_buffers.iter().enumerate() {
                let mip_size = (PREFILTER_SIZE >> mip).max(1);
                let aligned_bytes_per_row = self.prefilter_aligned_bytes_per_row[mip];
                let face_bytes = (aligned_bytes_per_row * mip_size) as u64;
                for face in 0..6u32 {
                    let region = BufferTextureCopyRegion::new(
                        BufferTextureLayout::new(
                            face as u64 * face_bytes,
                            Some(aligned_bytes_per_row),
                            None,
                        ),
                        TextureCopyLocation::new(mip as u32, TextureOrigin::new(0, 0, face)),
                        Extent3d::new_2d(mip_size, mip_size),
                    );
                    config = config.with_operation(TransferOperation::upload_texture(
                        staging.clone(),
                        self.prefilter_cubemap.clone(),
                        vec![region],
                    ));
                }
            }
        }

        // Clear staging buffers after building the config
        self.irradiance_staging = None;
        self.prefilter_staging = None;

        log::info!("IBL textures upload config created");
        Some(config)
    }
}
