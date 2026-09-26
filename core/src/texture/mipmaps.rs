//! CPU import-time mip reduction. GPU uploads remain the graphics graph's job.

use super::{CpuTexture, TextureDimension, TextureFormat};
use half::f16;

/// How texture content is reduced during import. Existing stored mip chains
/// are never regenerated. Normal maps store XYZ in RGB [0, 1].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum MipmapFilter {
    /// Area average, in linear light for sRGB formats.
    #[default]
    Color,
    /// Decode RGB to [-1, 1], average, normalize, and encode back to [0, 1].
    /// Requires a linear RGB(A) format. A cancelling vector becomes +Z.
    /// For cubemaps, all faces must use a common vector basis (e.g. world
    /// space), not separate face-local tangent frames.
    NormalMap,
    /// Average color and rescale alpha to retain mip 0's fraction of texels
    /// passing `alpha >= cutoff / 255`. Cutoff must be in 1..=254.
    /// Coverage is approximate: finite texel counts and equal alpha values
    /// make some fractions impossible, especially near the end of the chain.
    AlphaCoverage { cutoff: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MipmapError {
    UnsupportedFormat(TextureFormat),
    InvalidTexture(&'static str),
    InvalidFilter(&'static str),
}

impl std::fmt::Display for MipmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => {
                write!(f, "CPU mip generation does not support {format:?}")
            }
            Self::InvalidTexture(message) | Self::InvalidFilter(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for MipmapError {}

impl CpuTexture {
    /// Generate a full chain for a 2D texture, 2D array, cubemap or cube array.
    /// 2D array layers are independent. Cubes use a cross-face tent filter;
    /// alpha coverage is corrected per cube, and cubes never mix. Output
    /// remains mip-major with all layers/faces stored within each mip.
    /// Supports every uncompressed color format represented by TextureFormat,
    /// including non-filterable float and integer channels. Integer averages
    /// round to the nearest representable value. Depth/stencil and compressed
    /// textures require a precomputed chain. Failure leaves the texture intact.
    /// Already populated chains are preserved without applying `filter`.
    pub fn generate_mipmaps(&mut self, filter: MipmapFilter) -> Result<(), MipmapError> {
        if self.mip_level_count > 1 {
            return Ok(());
        }
        if !matches!(
            self.dimension,
            TextureDimension::D2
                | TextureDimension::D2Array
                | TextureDimension::Cube
                | TextureDimension::CubeArray
        ) || self.depth_or_array_layers == 0
            || (matches!(
                self.dimension,
                TextureDimension::D2 | TextureDimension::Cube
            ) && self.depth_or_array_layers != 1)
            || (self.dimension.is_cubemap() && self.width != self.height)
            || self.width == 0
            || self.height == 0
            || self.mip_level_count == 0
        {
            return Err(MipmapError::InvalidTexture(
                "mip generation requires a nonempty 2D texture, array, or square cubemap",
            ));
        }
        let format = self.format;
        if format.is_compressed() || format.is_depth_stencil() {
            return Err(MipmapError::UnsupportedFormat(format));
        }
        let rgba = matches!(
            format,
            TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba8UnormSrgb
                | TextureFormat::Bgra8Unorm
                | TextureFormat::Bgra8UnormSrgb
                | TextureFormat::Rgba16Float
                | TextureFormat::Rgba32Float
                | TextureFormat::Rgba10a2Unorm
                | TextureFormat::Bgra10a2Unorm
        );
        match filter {
            MipmapFilter::NormalMap if !rgba || format.is_srgb() => {
                return Err(MipmapError::InvalidFilter(
                    "normal-map mips require a linear RGB(A) format",
                ));
            }
            MipmapFilter::AlphaCoverage { cutoff } if !rgba || cutoff == 0 || cutoff == 255 => {
                return Err(MipmapError::InvalidFilter(
                    "alpha-coverage mips require RGBA and a cutoff in 1..=254",
                ));
            }
            _ => {}
        }
        let stride = format.block_size() as usize;
        let layer_bytes = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(stride))
            .ok_or(MipmapError::InvalidTexture("mip byte size overflows"))?;
        let layers = if self.dimension == TextureDimension::CubeArray {
            self.depth_or_array_layers
                .checked_mul(6)
                .ok_or(MipmapError::InvalidTexture("cube face count overflows"))?
        } else {
            self.layer_count()
        };
        let expected = layer_bytes
            .checked_mul(layers as usize)
            .ok_or(MipmapError::InvalidTexture("mip array byte size overflows"))?;
        if self.data.len() != expected {
            return Err(MipmapError::InvalidTexture(
                "mip 0 byte count does not match its dimensions",
            ));
        }
        let mut sources: Vec<Vec<[f64; 4]>> = self
            .data
            .chunks_exact(layer_bytes)
            .map(|layer| {
                layer
                    .chunks_exact(stride)
                    .map(|p| decode(format, p))
                    .collect()
            })
            .collect();
        if sources.iter().flatten().flatten().any(|v| !v.is_finite()) {
            return Err(MipmapError::InvalidTexture(
                "mip source contains non-finite values",
            ));
        }
        let group_size = if self.dimension.is_cubemap() { 6 } else { 1 };
        let coverage: Vec<_> = sources
            .chunks_exact(group_size)
            .map(|group| match filter {
                MipmapFilter::AlphaCoverage { cutoff } => {
                    group
                        .iter()
                        .map(|source| coverage(source, f64::from(cutoff) / 255.0))
                        .sum::<f64>()
                        / group_size as f64
                }
                _ => 0.0,
            })
            .collect();
        let mut data = self.data.clone();
        let (mut width, mut height) = (self.width, self.height);
        let mut levels = 1;
        while width > 1 || height > 1 {
            let (w, h) = ((width / 2).max(1), (height / 2).max(1));
            // All faces must read the same source level before any is replaced.
            let mut next: Vec<_> = sources
                .iter()
                .enumerate()
                .map(|(layer, source)| {
                    if self.dimension.is_cubemap() {
                        reduce_cube(&sources, layer, width, w)
                    } else {
                        reduce(source, width, height, w, h)
                    }
                })
                .collect();
            for reduced in &mut next {
                if filter == MipmapFilter::NormalMap {
                    for pixel in reduced {
                        let n = [
                            pixel[0] * 2.0 - 1.0,
                            pixel[1] * 2.0 - 1.0,
                            pixel[2] * 2.0 - 1.0,
                        ];
                        let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
                        if length > 1e-6 {
                            for i in 0..3 {
                                pixel[i] = (n[i] / length + 1.0) * 0.5;
                            }
                        } else {
                            pixel[..3].copy_from_slice(&[0.5, 0.5, 1.0]);
                        }
                    }
                }
            }
            for (group_index, group) in next.chunks_exact(group_size).enumerate() {
                let scale = match filter {
                    MipmapFilter::AlphaCoverage { cutoff } => {
                        alpha_scale(group, coverage[group_index], cutoff, format)
                    }
                    _ => 1.0,
                };
                for &pixel in group.iter().flatten() {
                    let mut output = pixel;
                    if matches!(filter, MipmapFilter::AlphaCoverage { .. }) {
                        output[3] = (output[3] * scale).clamp(0.0, 1.0);
                    }
                    encode(format, output, &mut data);
                }
                // Do not feed the alpha correction back into the next reduction:
                // each level independently targets the original base coverage.
            }
            sources = next;
            width = w;
            height = h;
            levels += 1;
        }
        self.data = data;
        self.mip_level_count = levels;
        Ok(())
    }
}

// KTX/Vulkan face convention: +X, -X, +Y, -Y, +Z, -Z.
// https://github.khronos.org/Vulkan-Site/spec/latest/chapters/textures.html
fn cube_texel(face: usize, x: i32, y: i32, size: u32) -> (usize, usize) {
    // Integer texel-center rays avoid CPU/GPU floating-point disagreement
    // when a projected tap lands exactly on a texel boundary (notably NPOT).
    let n = i64::from(size);
    let u = 2 * i64::from(x) + 1 - n;
    let v = 2 * i64::from(y) + 1 - n;
    let [rx, ry, rz] = match face {
        0 => [n, -v, -u],
        1 => [-n, -v, u],
        2 => [u, n, v],
        3 => [u, -n, -v],
        4 => [u, -v, n],
        5 => [-u, -v, -n],
        _ => unreachable!(),
    };
    let (ax, ay, az) = (rx.abs(), ry.abs(), rz.abs());
    let (face, s, t, major) = if ax >= ay && ax >= az {
        if rx > 0 {
            (0, -rz, -ry, ax)
        } else {
            (1, rz, -ry, ax)
        }
    } else if ay >= az {
        if ry > 0 {
            (2, rx, rz, ay)
        } else {
            (3, rx, -rz, ay)
        }
    } else if rz > 0 {
        (4, rx, -ry, az)
    } else {
        (5, -rx, -ry, az)
    };
    let coord = |c: i64| {
        (((c + major) as u128 * size as u128 / (2 * major) as u128) as usize).min(size as usize - 1)
    };
    (face, coord(t) * size as usize + coord(s))
}

/// Tent reduction with taps projected onto neighboring faces, including at
/// corners. The GPU shader uses the same convention, kernel and tie order.
fn reduce_cube(sources: &[Vec<[f64; 4]>], layer: usize, size: u32, dst_size: u32) -> Vec<[f64; 4]> {
    let scale = f64::from(size) / f64::from(dst_size);
    let cube_base = layer / 6 * 6;
    let mut result = Vec::with_capacity(dst_size as usize * dst_size as usize);
    for y in 0..dst_size {
        for x in 0..dst_size {
            let cx = (f64::from(x) + 0.5) * scale;
            let cy = (f64::from(y) + 0.5) * scale;
            let mut sum = [0.0; 4];
            let mut total = 0.0;
            for iy in (cy - scale).floor() as i32..(cy + scale).ceil() as i32 {
                for ix in (cx - scale).floor() as i32..(cx + scale).ceil() as i32 {
                    let weight = (1.0 - ((f64::from(ix) + 0.5 - cx) / scale).abs()).max(0.0)
                        * (1.0 - ((f64::from(iy) + 0.5 - cy) / scale).abs()).max(0.0);
                    let (face, index) = cube_texel(layer % 6, ix, iy, size);
                    let p = sources[cube_base + face][index];
                    for c in 0..4 {
                        sum[c] += p[c] * weight;
                    }
                    total += weight;
                }
            }
            for v in &mut sum {
                *v /= total;
            }
            result.push(sum);
        }
    }
    result
}

fn reduce(source: &[[f64; 4]], width: u32, height: u32, w: u32, h: u32) -> Vec<[f64; 4]> {
    let sx = f64::from(width) / f64::from(w);
    let sy = f64::from(height) / f64::from(h);
    let mut dst = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            let (x0, x1) = (f64::from(x) * sx, f64::from(x + 1) * sx);
            let (y0, y1) = (f64::from(y) * sy, f64::from(y + 1) * sy);
            let mut sum = [0.0; 4];
            for iy in y0.floor() as u32..(y1.ceil() as u32).min(height) {
                for ix in x0.floor() as u32..(x1.ceil() as u32).min(width) {
                    let weight = (x1.min(f64::from(ix + 1)) - x0.max(f64::from(ix)))
                        * (y1.min(f64::from(iy + 1)) - y0.max(f64::from(iy)))
                        / (sx * sy);
                    let pixel = source[iy as usize * width as usize + ix as usize];
                    for c in 0..4 {
                        sum[c] += pixel[c] * weight;
                    }
                }
            }
            dst.push(sum);
        }
    }
    dst
}

fn coverage(pixels: &[[f64; 4]], cutoff: f64) -> f64 {
    pixels.iter().filter(|p| p[3] >= cutoff).count() as f64 / pixels.len() as f64
}

fn alpha_scale(group: &[Vec<[f64; 4]>], target: f64, cutoff: u8, format: TextureFormat) -> f64 {
    let cutoff = f64::from(cutoff) / 255.0;
    let quantize = |a: f64| match format {
        TextureFormat::Rgba16Float => f16::from_f64(a).to_f64(),
        TextureFormat::Rgba32Float => f64::from(a as f32),
        TextureFormat::Rgba10a2Unorm | TextureFormat::Bgra10a2Unorm => (a * 3.0).round() / 3.0,
        _ => (a * 255.0).round() / 255.0,
    };
    let error = |scale: f64| {
        let count = group
            .iter()
            .flatten()
            .filter(|p| quantize((p[3] * scale).clamp(0.0, 1.0)) >= cutoff)
            .count();
        count as f64 / (group.len() * group[0].len()) as f64 - target
    };
    let mut best = 1.0;
    let mut best_error = error(best).abs();
    let min_positive = group
        .iter()
        .flatten()
        .map(|p| p[3])
        .filter(|a| *a > 0.0)
        .fold(1.0, f64::min);
    let limit = (1.0 / min_positive).max(1.0);
    let (mut lo, mut hi) = (0.0, 1.0);
    while error(hi) < 0.0 && hi < limit {
        lo = hi;
        hi = (hi * 2.0).min(limit);
    }
    for _ in 0..48 {
        let scale = (lo + hi) * 0.5;
        let e = error(scale);
        if e.abs() < best_error {
            best = scale;
            best_error = e.abs();
        }
        if e < 0.0 {
            lo = scale;
        } else {
            hi = scale;
        }
    }
    best
}

fn linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn decode(format: TextureFormat, bytes: &[u8]) -> [f64; 4] {
    use TextureFormat::*;
    let mut p = [0.0, 0.0, 0.0, 1.0];
    match format {
        R8Unorm | Rg8Unorm | Rgba8Unorm | Rgba8UnormSrgb | Bgra8Unorm | Bgra8UnormSrgb => {
            for (dst, &b) in p.iter_mut().zip(bytes) {
                *dst = f64::from(b) / 255.0;
            }
            if matches!(format, Bgra8Unorm | Bgra8UnormSrgb) {
                p.swap(0, 2);
            }
        }
        R8Snorm => p[0] = (f64::from(bytes[0] as i8) / 127.0).max(-1.0),
        R8Uint => p[0] = f64::from(bytes[0]),
        R8Sint => p[0] = f64::from(bytes[0] as i8),
        R16Unorm => p[0] = f64::from(u16::from_le_bytes(bytes.try_into().unwrap())) / 65535.0,
        R16Float | Rg16Float | Rgba16Float => {
            for (dst, b) in p.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = f16::from_le_bytes(b.try_into().unwrap()).to_f64();
            }
        }
        R32Float | Rg32Float | Rgba32Float => {
            for (dst, b) in p.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
            }
        }
        R32Uint => p[0] = f64::from(u32::from_le_bytes(bytes.try_into().unwrap())),
        Rgba10a2Unorm | Bgra10a2Unorm => {
            let bits = u32::from_le_bytes(bytes.try_into().unwrap());
            for (i, value) in p[..3].iter_mut().enumerate() {
                *value = f64::from((bits >> (10 * i)) & 1023) / 1023.0;
            }
            p[3] = f64::from(bits >> 30) / 3.0;
            if format == Bgra10a2Unorm {
                p.swap(0, 2);
            }
        }
        _ => unreachable!("validated uncompressed color format"),
    }
    if format.is_srgb() {
        for v in &mut p[..3] {
            *v = linear(*v);
        }
    }
    p
}

fn encode(format: TextureFormat, mut p: [f64; 4], bytes: &mut Vec<u8>) {
    use TextureFormat::*;
    if format.is_srgb() {
        for v in &mut p[..3] {
            *v = srgb(*v);
        }
    }
    match format {
        R8Unorm | Rg8Unorm | Rgba8Unorm | Rgba8UnormSrgb | Bgra8Unorm | Bgra8UnormSrgb => {
            if matches!(format, Bgra8Unorm | Bgra8UnormSrgb) {
                p.swap(0, 2);
            }
            for v in &p[..format.block_size() as usize] {
                bytes.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        R8Snorm => bytes.push((p[0].clamp(-1.0, 1.0) * 127.0).round() as i8 as u8),
        R8Uint => bytes.push(p[0].round() as u8),
        R8Sint => bytes.push(p[0].round() as i8 as u8),
        R16Unorm => bytes
            .extend_from_slice(&((p[0].clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()),
        R16Float | Rg16Float | Rgba16Float => {
            for v in &p[..format.block_size() as usize / 2] {
                bytes.extend_from_slice(&f16::from_f64(*v).to_le_bytes());
            }
        }
        R32Float | Rg32Float | Rgba32Float => {
            for v in &p[..format.block_size() as usize / 4] {
                bytes.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        R32Uint => bytes.extend_from_slice(&(p[0].round() as u32).to_le_bytes()),
        Rgba10a2Unorm | Bgra10a2Unorm => {
            if format == Bgra10a2Unorm {
                p.swap(0, 2);
            }
            let mut bits = 0u32;
            for (i, v) in p[..3].iter().enumerate() {
                bits |= ((v.clamp(0.0, 1.0) * 1023.0).round() as u32) << (10 * i);
            }
            bits |= ((p[3].clamp(0.0, 1.0) * 3.0).round() as u32) << 30;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        _ => unreachable!("validated uncompressed color format"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(format: TextureFormat, width: u32, height: u32, pixels: &[[f64; 4]]) -> CpuTexture {
        let mut bytes = Vec::new();
        for &p in pixels {
            encode(format, p, &mut bytes);
        }
        CpuTexture::new(width, height, format, bytes)
    }
    fn top(texture: &CpuTexture) -> [f64; 4] {
        decode(
            texture.format,
            &texture.data[texture.byte_range(texture.mip_level_count - 1, 0)],
        )
    }

    #[test]
    fn odd_sizes_and_one_pixel_axis_include_edge_texels() {
        for (w, h) in [(3, 1), (1, 3), (3, 3), (5, 3)] {
            let mut pixels = vec![[0.0; 4]; (w * h) as usize];
            pixels.last_mut().unwrap()[0] = 1.0;
            let mut cpu = texture(TextureFormat::Rgba32Float, w, h, &pixels);
            cpu.generate_mipmaps(MipmapFilter::Color).unwrap();
            assert!((top(&cpu)[0] - 1.0 / f64::from(w * h)).abs() < 1e-6);
            assert_eq!(cpu.data.len(), cpu.expected_data_len());
        }
    }

    #[test]
    fn srgb_averages_in_linear_light_and_preserves_base() {
        let mut cpu = CpuTexture::new(
            2,
            1,
            TextureFormat::Rgba8UnormSrgb,
            vec![0, 0, 0, 255, 255, 255, 255, 255],
        );
        let base = cpu.data.clone();
        cpu.generate_mipmaps(MipmapFilter::Color).unwrap();
        assert_eq!(&cpu.data[..8], &base);
        assert_eq!(&cpu.data[8..], &[188, 188, 188, 255]);
    }

    #[test]
    fn every_uncompressed_color_encoding_roundtrips_and_reduces() {
        use TextureFormat::*;
        for format in [
            R8Unorm,
            R8Snorm,
            R8Uint,
            R8Sint,
            R16Unorm,
            R16Float,
            Rg8Unorm,
            R32Float,
            R32Uint,
            Rg16Float,
            Rgba8Unorm,
            Rgba8UnormSrgb,
            Bgra8Unorm,
            Bgra8UnormSrgb,
            Rgba10a2Unorm,
            Bgra10a2Unorm,
            Rgba16Float,
            Rg32Float,
            Rgba32Float,
        ] {
            let mut cpu = texture(format, 2, 2, &[[1.0, 0.5, 0.25, 1.0]; 4]);
            let base = cpu.data.clone();
            cpu.generate_mipmaps(MipmapFilter::Color).unwrap();
            assert_eq!(&cpu.data[..base.len()], &base);
            assert_eq!(cpu.data.len(), cpu.expected_data_len());
            let base_pixel = decode(format, &base[..format.block_size() as usize]);
            let result = top(&cpu);
            for c in 0..4 {
                assert!(
                    (base_pixel[c] - result[c]).abs() < 0.01,
                    "{format:?}: {result:?}"
                );
            }
        }
        // HDR and integer values must not be treated as normalized colors.
        for format in [R32Float, R16Float, R32Uint, R8Uint] {
            let mut cpu = texture(format, 2, 1, &[[2.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]);
            cpu.generate_mipmaps(MipmapFilter::Color).unwrap();
            assert_eq!(top(&cpu)[0], 6.0);
        }
    }

    #[test]
    fn normals_are_renormalized_and_cancellation_is_finite() {
        let mut cpu = texture(
            TextureFormat::Rgba32Float,
            2,
            1,
            &[[1.0, 0.5, 0.5, 1.0], [0.5, 1.0, 0.5, 1.0]],
        );
        cpu.generate_mipmaps(MipmapFilter::NormalMap).unwrap();
        let p = top(&cpu);
        let length = p[..3]
            .iter()
            .map(|v| (v * 2.0 - 1.0).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((length - 1.0).abs() < 1e-6);
        assert!((p[0] - p[1]).abs() < 1e-6);
        let mut cpu = texture(
            TextureFormat::Rgba32Float,
            2,
            1,
            &[[1.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]],
        );
        cpu.generate_mipmaps(MipmapFilter::NormalMap).unwrap();
        assert_eq!(top(&cpu), [0.5, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn alpha_coverage_corrects_disappearing_mask_and_keeps_rgb() {
        // Half the base passes 0.75; box filtering alone leaves only one of
        // four mip texels above the cutoff. Scaling restores two of four.
        let alpha = [1.0, 1.0, 1.0, 0.4, 1.0, 0.2, 0.0, 0.0];
        let pixels: Vec<_> = alpha.iter().map(|&a| [0.2, 0.4, 0.6, a]).collect();
        let mut cpu = texture(TextureFormat::Rgba8Unorm, 8, 1, &pixels);
        let base = cpu.data.clone();
        cpu.generate_mipmaps(MipmapFilter::AlphaCoverage { cutoff: 191 })
            .unwrap();
        assert_eq!(&cpu.data[..base.len()], &base);
        let mip = &cpu.data[cpu.byte_range(1, 0)];
        assert_eq!(mip.chunks_exact(4).filter(|p| p[3] >= 191).count(), 2);
        for p in mip.chunks_exact(4) {
            assert_eq!(&p[..3], &[51, 102, 153]);
        }
    }

    #[test]
    fn rejects_invalid_inputs_without_mutation_and_preserves_authored_mips() {
        let mut cpu = CpuTexture::new(2, 2, TextureFormat::Bc1RgbaUnorm, vec![0; 8]);
        assert!(matches!(
            cpu.generate_mipmaps(MipmapFilter::Color),
            Err(MipmapError::UnsupportedFormat(_))
        ));
        assert_eq!(cpu.data, vec![0; 8]);
        let mut cpu = texture(TextureFormat::Rgba8UnormSrgb, 2, 2, &[[1.0; 4]; 4]);
        assert!(cpu.generate_mipmaps(MipmapFilter::NormalMap).is_err());
        assert!(
            cpu.generate_mipmaps(MipmapFilter::AlphaCoverage { cutoff: 0 })
                .is_err()
        );
        cpu.generate_mipmaps(MipmapFilter::Color).unwrap();
        let authored = cpu.data.clone();
        cpu.generate_mipmaps(MipmapFilter::NormalMap).unwrap();
        assert_eq!(cpu.data, authored);
        for dimension in [TextureDimension::D1, TextureDimension::D3] {
            let mut cpu =
                texture(TextureFormat::Rgba8Unorm, 2, 2, &[[1.0; 4]; 4]).with_dimension(dimension);
            assert!(cpu.generate_mipmaps(MipmapFilter::Color).is_err());
        }
        let mut cpu = CpuTexture::new(u32::MAX, u32::MAX, TextureFormat::Rgba32Float, vec![]);
        assert!(cpu.generate_mipmaps(MipmapFilter::Color).is_err());
    }

    #[test]
    fn array_layers_match_independent_color_normal_and_coverage_chains() {
        let alphas = [1.0, 1.0, 1.0, 0.4, 1.0, 0.2, 0.0, 0.0];
        let a: Vec<_> = alphas.iter().map(|&a| [1.0, 0.5, 0.5, a]).collect();
        let b = [[0.5, 1.0, 0.5, 0.9]; 8];
        for filter in [
            MipmapFilter::Color,
            MipmapFilter::NormalMap,
            MipmapFilter::AlphaCoverage { cutoff: 191 },
        ] {
            let mut layers = [
                texture(TextureFormat::Rgba8Unorm, 8, 1, &a),
                texture(TextureFormat::Rgba8Unorm, 8, 1, &b),
            ];
            let base: Vec<_> = layers.iter().flat_map(|t| t.data.iter().copied()).collect();
            let mut array = CpuTexture::new(8, 1, TextureFormat::Rgba8Unorm, base.clone())
                .with_dimension(TextureDimension::D2Array)
                .with_depth_or_array_layers(2);
            array.generate_mipmaps(filter).unwrap();
            assert_eq!(array.mip_level_count, 4);
            assert_eq!(array.layer_count(), 2);
            assert_eq!(&array.data[..base.len()], &base);
            assert_eq!(array.data.len(), array.expected_data_len());
            for (layer, solo) in layers.iter_mut().enumerate() {
                solo.generate_mipmaps(filter).unwrap();
                for mip in 0..4 {
                    assert_eq!(
                        &array.data[array.byte_range(mip, layer as u32)],
                        &solo.data[solo.byte_range(mip, 0)],
                        "{filter:?}, mip {mip}, layer {layer}"
                    );
                }
            }
        }
    }

    #[test]
    fn array_rejects_bad_layer_counts_and_incomplete_base_without_mutation() {
        for (layers, bytes) in [(0, 0), (2, 16), (u32::MAX, 16)] {
            let mut array = CpuTexture::new(2, 2, TextureFormat::Rgba8Unorm, vec![42; bytes])
                .with_dimension(TextureDimension::D2Array)
                .with_depth_or_array_layers(layers);
            assert!(array.generate_mipmaps(MipmapFilter::Color).is_err());
            assert_eq!(array.data, vec![42; bytes]);
            assert_eq!(array.mip_level_count, 1);
        }
    }

    #[test]
    fn cubemap_all_twelve_edges_have_correct_neighbor_and_orientation() {
        // Four directed edges per face (left, right, top, bottom); every
        // physical edge appears twice. Expected transforms are independent
        // of the direction/major-axis implementation under test.
        for k in 0..8usize {
            let r = 7 - k;
            let expected = [
                [(4, 7, k), (5, 0, k), (2, 7, r), (3, 7, k)],
                [(5, 7, k), (4, 0, k), (2, 0, k), (3, 0, r)],
                [(1, k, 0), (0, r, 0), (5, r, 0), (4, k, 0)],
                [(1, r, 7), (0, k, 7), (4, k, 7), (5, r, 7)],
                [(1, 7, k), (0, 0, k), (2, k, 7), (3, k, 0)],
                [(0, 7, k), (1, 0, k), (2, r, 0), (3, r, 7)],
            ];
            for (face, edges) in expected.iter().enumerate() {
                for (edge, &(f, x, y)) in edges.iter().enumerate() {
                    let (px, py) =
                        [(-1, k as i32), (8, k as i32), (k as i32, -1), (k as i32, 8)][edge];
                    assert_eq!(
                        cube_texel(face, px, py, 8),
                        (f, y * 8 + x),
                        "face {face}, edge {edge}, texel {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn cube_filter_crosses_edges_corners_and_isolates_cube_arrays() {
        let mut faces = vec![vec![[0.0; 4]; 64]; 12];
        // A constant +X face must contribute to all its four neighbors, but
        // not the opposite face or any face of the second cube.
        faces[0].fill([1.0; 4]);
        for face in 0..12 {
            let reduced = reduce_cube(&faces, face, 8, 4);
            let nonzero = reduced.iter().any(|p| p[0] > 0.0);
            assert_eq!(nonzero, [0, 2, 3, 4, 5].contains(&face), "face {face}");
        }
        // At each corner, both incident neighboring faces enter the footprint.
        for face in 0..6 {
            for (x, y) in [(0usize, 0usize), (3, 0), (0, 3), (3, 3)] {
                let mut contributors = 0;
                for source in 0..6 {
                    let mut faces = vec![vec![[0.0; 4]; 64]; 6];
                    faces[source].fill([1.0; 4]);
                    if reduce_cube(&faces, face, 8, 4)[y * 4 + x][0] > 0.0 {
                        contributors += 1;
                    }
                }
                assert_eq!(contributors, 3, "face {face}, corner {x},{y}");
            }
        }
    }

    #[test]
    fn cube_constant_hdr_and_per_cube_alpha_coverage_survive_all_levels() {
        let mut cube = texture(
            TextureFormat::Rgba32Float,
            5,
            5,
            &[[2.0, 4.0, 8.0, 0.75]; 25 * 6],
        )
        .with_dimension(TextureDimension::Cube);
        cube.generate_mipmaps(MipmapFilter::Color).unwrap();
        for mip in 0..cube.mip_level_count {
            for face in 0..6 {
                for p in cube.data[cube.byte_range(mip, face)].chunks_exact(16) {
                    assert_eq!(decode(cube.format, p), [2.0, 4.0, 8.0, 0.75]);
                }
            }
        }
        let mut faces = Vec::new();
        for face in 0..12 {
            faces.extend(vec![
                [
                    0.5,
                    0.5,
                    1.0,
                    if face < 3 || face >= 6 { 1.0 } else { 0.0 }
                ];
                16
            ]);
        }
        let mut array = texture(TextureFormat::Rgba8Unorm, 4, 4, &faces)
            .with_dimension(TextureDimension::CubeArray)
            .with_depth_or_array_layers(2);
        array
            .generate_mipmaps(MipmapFilter::AlphaCoverage { cutoff: 191 })
            .unwrap();
        // Correction uses one scale per cube, preserving the second cube's
        // opaque mask instead of inheriting the first cube's 50% coverage.
        for mip in 1..array.mip_level_count {
            let first: Vec<_> = (0..6)
                .flat_map(|face| {
                    array.data[array.byte_range(mip, face)]
                        .chunks_exact(4)
                        .map(|p| p[3])
                        .collect::<Vec<_>>()
                })
                .collect();
            let passing = first.iter().filter(|&&a| a >= 191).count();
            assert!((passing as f64 / first.len() as f64 - 0.5).abs() <= 1.0 / first.len() as f64);
            for face in 6..12 {
                assert!(
                    array.data[array.byte_range(mip, face)]
                        .chunks_exact(4)
                        .all(|p| p[3] == 255)
                );
            }
        }
    }

    #[test]
    fn cube_validation_rejects_nonsquare_missing_faces_and_overflow() {
        for (w, h, count, dimension) in [
            (4, 2, 1, TextureDimension::Cube),
            (4, 4, u32::MAX, TextureDimension::CubeArray),
            (4, 4, 1, TextureDimension::Cube),
        ] {
            let mut cpu = CpuTexture::new(w, h, TextureFormat::Rgba8Unorm, vec![0; 64])
                .with_dimension(dimension)
                .with_depth_or_array_layers(count);
            assert!(cpu.generate_mipmaps(MipmapFilter::Color).is_err());
            assert_eq!(cpu.data, vec![0; 64]);
        }
    }
}
