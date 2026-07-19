//! CPU-side color transforms mirroring the shader library
//! (`shaders/library/color.slang`). Hosts use them wherever GPU output is
//! post-processed on the CPU — HDR screenshot encoding, golden-image tests —
//! so the CPU path and the shaders apply the *same* math.

/// Linear → sRGB transfer (exact OETF; what the hardware applies writing to
/// an sRGB-typed target).
pub fn srgb_encode(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB → linear transfer (exact EOTF inverse; what the hardware applies
/// sampling an sRGB-typed texture).
pub fn srgb_decode(encoded: f32) -> f32 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// Khronos PBR Neutral tonemap — the CPU port of `tonemap_pbr_neutral` in
/// `shaders/library/color.slang` (hue-preserving peak compression;
/// <https://github.com/KhronosGroup/ToneMapping>).
pub fn tonemap_pbr_neutral(color: [f32; 3]) -> [f32; 3] {
    const START_COMPRESSION: f32 = 0.8 - 0.04;
    const DESATURATION: f32 = 0.15;

    let x = color[0].min(color[1]).min(color[2]);
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let mut color = color.map(|c| c - offset);

    let peak = color[0].max(color[1]).max(color[2]);
    if peak < START_COMPRESSION {
        return color;
    }

    let d = 1.0 - START_COMPRESSION;
    let new_peak = 1.0 - d * d / (peak + d - START_COMPRESSION);
    for c in &mut color {
        *c *= new_peak / peak;
    }

    let g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
    color.map(|c| c + (new_peak - c) * g)
}

/// Decode an IEEE 754 binary16 (as stored in `Rgba16Float` textures) to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let out = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            // Subnormal: renormalize into the f32 exponent range.
            let mut exp = 127 - 15 + 1;
            let mut frac = frac;
            while frac & 0x400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            (sign << 31) | ((exp as u32) << 23) | ((frac & 0x3FF) << 13)
        }
        (0x1F, 0) => (sign << 31) | (0xFF << 23),
        (0x1F, _) => (sign << 31) | (0xFF << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sRGB pair round-trips and hits the standard anchor points.
    #[test]
    fn srgb_round_trip() {
        for v in [0.0, 0.001, 0.0031308, 0.04045, 0.2, 0.5, 0.735, 1.0] {
            let there = srgb_encode(v);
            let back = srgb_decode(there);
            assert!((back - v).abs() < 1e-6, "{v} -> {there} -> {back}");
        }
        assert_eq!(srgb_encode(0.0), 0.0);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-6);
        // 50% linear is ~73.5% encoded — the classic mid-gray anchor.
        assert!((srgb_encode(0.5) - 0.7354).abs() < 1e-3);
    }

    /// Below the compression knee the tonemap only applies the black-level
    /// offset; a saturated over-range red keeps its hue (the property the
    /// curve was chosen for).
    #[test]
    fn tonemap_matches_spec_shape() {
        // Sub-knee color with min channel >= 0.08: offset is exactly 0.04.
        let low = tonemap_pbr_neutral([0.5, 0.3, 0.1]);
        assert!((low[0] - 0.46).abs() < 1e-6);
        assert!((low[1] - 0.26).abs() < 1e-6);
        // 0.1 - 6.25*0.01 = 0.0375 offset for x = 0.1 < 0.08? No: 0.1 >= 0.08
        // so the offset is 0.04 there too.
        assert!((low[2] - 0.06).abs() < 1e-6);

        // Bright saturated red stays red-dominant (no salmon washout) and in
        // range.
        let red = tonemap_pbr_neutral([1.36, 0.27, 0.19]);
        assert!(red[0] > 0.85 && red[0] <= 1.0, "peak compressed: {red:?}");
        assert!(red[0] / red[1].max(1e-3) > 3.0, "hue preserved: {red:?}");
    }

    /// Half-float decode agrees with known bit patterns, including
    /// subnormals and specials.
    #[test]
    fn f16_decode_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // f16::MAX
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }
}
