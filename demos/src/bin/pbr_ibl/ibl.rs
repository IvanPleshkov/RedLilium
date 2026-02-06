//! Image-Based Lighting (IBL) computation on CPU.
//!
//! This module implements the IBL pre-computation algorithms from the
//! LearnOpenGL PBR tutorials:
//! - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
//! - https://learnopengl.com/PBR/IBL/Specular-IBL

use std::f32::consts::PI;

use glam::Vec3;

use crate::{IRRADIANCE_SIZE, PREFILTER_SIZE};

/// Compute IBL cubemaps from an equirectangular HDR image.
///
/// Returns irradiance cubemap data and pre-filtered environment map with mipmaps.
pub fn compute_ibl_cpu(
    hdr_data: &[f32],
    hdr_width: u32,
    hdr_height: u32,
) -> (Vec<u16>, Vec<Vec<u16>>) {
    // Sample direction from equirectangular map
    let sample_equirect = |dir: Vec3| -> Vec3 {
        let inv_atan = glam::vec2(
            0.5 * std::f32::consts::FRAC_1_PI,
            std::f32::consts::FRAC_1_PI,
        );
        let uv = glam::vec2(dir.z.atan2(dir.x), dir.y.asin()) * inv_atan + 0.5;

        let x = ((uv.x * hdr_width as f32) as u32).min(hdr_width - 1);
        let y = (((1.0 - uv.y) * hdr_height as f32) as u32).min(hdr_height - 1);
        let idx = ((y * hdr_width + x) * 4) as usize;

        Vec3::new(hdr_data[idx], hdr_data[idx + 1], hdr_data[idx + 2])
    };

    // Compute irradiance cubemap
    log::info!("Computing irradiance cubemap...");
    let mut irradiance_data =
        Vec::with_capacity((IRRADIANCE_SIZE * IRRADIANCE_SIZE * 6 * 4) as usize);

    for face in 0..6 {
        for y in 0..IRRADIANCE_SIZE {
            for x in 0..IRRADIANCE_SIZE {
                let dir = cubemap_dir(face, x, y, IRRADIANCE_SIZE);
                let irradiance = compute_irradiance(dir, &sample_equirect);
                irradiance_data.push(f32_to_f16_bits(irradiance.x));
                irradiance_data.push(f32_to_f16_bits(irradiance.y));
                irradiance_data.push(f32_to_f16_bits(irradiance.z));
                irradiance_data.push(f32_to_f16_bits(1.0));
            }
        }
    }

    // Compute pre-filtered environment map with mipmaps
    log::info!("Computing pre-filtered environment map...");
    let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;
    let mut prefilter_data = Vec::with_capacity(mip_levels as usize);

    for mip in 0..mip_levels {
        let mip_size = (PREFILTER_SIZE >> mip).max(1);
        let roughness = mip as f32 / (mip_levels - 1) as f32;
        let mut mip_data = Vec::with_capacity((mip_size * mip_size * 6 * 4) as usize);

        for face in 0..6 {
            for y in 0..mip_size {
                for x in 0..mip_size {
                    let dir = cubemap_dir(face, x, y, mip_size);
                    let prefiltered = compute_prefiltered(dir, roughness, &sample_equirect);
                    mip_data.push(f32_to_f16_bits(prefiltered.x));
                    mip_data.push(f32_to_f16_bits(prefiltered.y));
                    mip_data.push(f32_to_f16_bits(prefiltered.z));
                    mip_data.push(f32_to_f16_bits(1.0));
                }
            }
        }
        prefilter_data.push(mip_data);
    }

    (irradiance_data, prefilter_data)
}

/// Convert f32 to f16 bits (IEEE 754 half-precision).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or denormalized
        0
    } else if exp == 0xFF {
        // Infinity or NaN
        ((sign << 15) | 0x7C00 | (mantissa >> 13).min(0x3FF)) as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            ((sign << 15) | 0x7C00) as u16
        } else if new_exp <= 0 {
            // Underflow to zero or denorm
            0
        } else {
            ((sign << 15) | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
        }
    }
}

/// Get cubemap direction for a given face, pixel, and size.
fn cubemap_dir(face: u32, x: u32, y: u32, size: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

    let dir = match face {
        0 => Vec3::new(1.0, -v, -u),  // +X
        1 => Vec3::new(-1.0, -v, u),  // -X
        2 => Vec3::new(u, 1.0, v),    // +Y
        3 => Vec3::new(u, -1.0, -v),  // -Y
        4 => Vec3::new(u, -v, 1.0),   // +Z
        _ => Vec3::new(-u, -v, -1.0), // -Z
    };

    dir.normalize()
}

/// Compute irradiance for a given normal direction.
fn compute_irradiance<F: Fn(Vec3) -> Vec3>(normal: Vec3, sample_env: &F) -> Vec3 {
    let mut irradiance = Vec3::ZERO;

    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = normal.cross(up).normalize();
    let up = normal.cross(right);

    let sample_delta = 0.05;
    let mut nr_samples = 0.0;

    let mut phi = 0.0f32;
    while phi < 2.0 * PI {
        let mut theta = 0.0f32;
        while theta < 0.5 * PI {
            let tangent_sample = Vec3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            let sample_vec =
                tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * normal;

            irradiance += sample_env(sample_vec) * theta.cos() * theta.sin();
            nr_samples += 1.0;

            theta += sample_delta;
        }
        phi += sample_delta;
    }

    PI * irradiance / nr_samples
}

/// Compute pre-filtered environment map value.
fn compute_prefiltered<F: Fn(Vec3) -> Vec3>(normal: Vec3, roughness: f32, sample_env: &F) -> Vec3 {
    let r = normal;
    let v = r;

    let sample_count = 128u32;
    let mut prefiltered = Vec3::ZERO;
    let mut total_weight = 0.0;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, normal, roughness);
        let l = (2.0 * v.dot(h) * h - v).normalize();

        let n_dot_l = normal.dot(l).max(0.0);
        if n_dot_l > 0.0 {
            prefiltered += sample_env(l) * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    prefiltered / total_weight.max(0.001)
}

/// Hammersley sequence for low-discrepancy sampling.
fn hammersley(i: u32, n: u32) -> glam::Vec2 {
    glam::vec2(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

/// GGX importance sampling.
fn importance_sample_ggx(xi: glam::Vec2, n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;

    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let tangent = n.cross(up).normalize();
    let bitangent = n.cross(tangent);

    (tangent * h.x + bitangent * h.y + n * h.z).normalize()
}
