//! Texture loading and management

use crate::backend::traits::*;
use crate::backend::types::*;
use image::{DynamicImage, GenericImageView};
use std::path::Path;

/// Loaded texture data
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub data: Vec<u8>,
    pub name: String,
}

impl TextureData {
    /// Load texture from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let img = image::open(path).map_err(|e| e.to_string())?;
        Self::from_image(img, &name)
    }

    /// Load texture from bytes
    pub fn from_bytes(bytes: &[u8], name: &str) -> Result<Self, String> {
        let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
        Self::from_image(img, name)
    }

    /// Create texture from image
    fn from_image(img: DynamicImage, name: &str) -> Result<Self, String> {
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();
        let data = rgba.into_raw();

        Ok(Self {
            width,
            height,
            format: TextureFormat::Rgba8UnormSrgb,
            data,
            name: name.to_string(),
        })
    }

    /// Create a solid color texture
    pub fn solid_color(color: [u8; 4], name: &str) -> Self {
        Self {
            width: 1,
            height: 1,
            format: TextureFormat::Rgba8UnormSrgb,
            data: color.to_vec(),
            name: name.to_string(),
        }
    }

    /// Create a default white texture
    pub fn white() -> Self {
        Self::solid_color([255, 255, 255, 255], "white")
    }

    /// Create a default black texture
    pub fn black() -> Self {
        Self::solid_color([0, 0, 0, 255], "black")
    }

    /// Create a default normal map (pointing up)
    pub fn default_normal() -> Self {
        // Normal pointing up: (0, 0, 1) in tangent space
        // Encoded as RGB: (0.5, 0.5, 1.0) * 255 = (128, 128, 255)
        Self::solid_color([128, 128, 255, 255], "default_normal")
    }

    /// Create a checkerboard texture
    pub fn checkerboard(size: u32, color1: [u8; 4], color2: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity((size * size * 4) as usize);

        for y in 0..size {
            for x in 0..size {
                let is_even = ((x / 8) + (y / 8)) % 2 == 0;
                let color = if is_even { color1 } else { color2 };
                data.extend_from_slice(&color);
            }
        }

        Self {
            width: size,
            height: size,
            format: TextureFormat::Rgba8UnormSrgb,
            data,
            name: "checkerboard".to_string(),
        }
    }
}

/// GPU texture with associated view and sampler
pub struct GpuTexture {
    pub handle: TextureHandle,
    pub view: TextureViewHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub name: String,
}

impl GpuTexture {
    /// Create and upload texture to GPU
    pub fn create<B: GraphicsBackend>(
        backend: &mut B,
        data: &TextureData,
    ) -> BackendResult<Self> {
        let handle = backend.create_texture(&TextureDescriptor {
            label: Some(data.name.clone()),
            width: data.width,
            height: data.height,
            depth: 1,
            mip_levels: 1,
            format: data.format,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        })?;

        let view = backend.create_texture_view(handle)?;
        backend.write_texture(handle, &data.data, data.width, data.height);

        Ok(Self {
            handle,
            view,
            width: data.width,
            height: data.height,
            format: data.format,
            name: data.name.clone(),
        })
    }
}
