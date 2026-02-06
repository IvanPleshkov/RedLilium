//! HDR and texture loading utilities.

use redlilium_graphics::{CpuTexture, TextureFormat};

/// URL for the HDR environment map.
pub const HDR_URL: &str = "https://raw.githubusercontent.com/JoeyDeVries/LearnOpenGL/master/resources/textures/hdr/newport_loft.hdr";

/// URL for the pre-computed BRDF Look-Up Table.
pub const BRDF_LUT_URL: &str = "https://learnopengl.com/img/pbr/ibl_brdf_lut.png";

/// Load a BRDF LUT from a URL.
///
/// The LUT is expected to be a PNG image with R and G channels encoding
/// the scale and bias terms of the split-sum approximation.
pub fn load_brdf_lut_from_url(url: &str) -> Result<CpuTexture, String> {
    use std::io::Read;

    log::info!("Downloading BRDF LUT from: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Failed to download BRDF LUT: {e}"))?;

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| format!("Failed to read BRDF LUT data: {e}"))?;

    log::info!("Downloaded {} bytes, parsing PNG...", data.len());

    let img =
        image::load_from_memory(&data).map_err(|e| format!("Failed to decode BRDF LUT: {e}"))?;

    let width = img.width();
    let height = img.height();

    log::info!("BRDF LUT image: {}x{}", width, height);

    // Convert to RG8 (we only need red and green channels)
    let rgba = img.to_rgba8();
    let mut rg_data = Vec::with_capacity((width * height * 2) as usize);
    for pixel in rgba.pixels() {
        rg_data.push(pixel[0]); // R channel = scale
        rg_data.push(pixel[1]); // G channel = bias
    }

    Ok(CpuTexture::new(width, height, TextureFormat::Rg8Unorm, rg_data).with_name("brdf_lut"))
}

/// Load an HDR image from a URL.
///
/// Returns the image dimensions and RGBA float data.
pub fn load_hdr_from_url(url: &str) -> Result<(u32, u32, Vec<f32>), String> {
    use std::io::Read;

    log::info!("Downloading HDR texture from: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Failed to download HDR: {e}"))?;

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| format!("Failed to read HDR data: {e}"))?;

    log::info!("Downloaded {} bytes, parsing HDR...", data.len());

    let img = image::load_from_memory_with_format(&data, image::ImageFormat::Hdr)
        .map_err(|e| format!("Failed to decode HDR: {e}"))?;

    let width = img.width();
    let height = img.height();

    log::info!("HDR image: {}x{}", width, height);

    let rgba32f = img.to_rgba32f();
    let rgba_data: Vec<f32> = rgba32f.into_raw();

    Ok((width, height, rgba_data))
}
