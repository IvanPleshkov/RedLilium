//! Offline IBL bake (#137): BRDF integration LUT + environment cubemap set,
//! written as Zstd-supercompressed KTX2 into `std-assets/textures/ibl/`.
//!
//! ```text
//! scripts/fetch-hdri.sh                 # provision the pinned source HDRI
//! cargo run -p xtask -- bake-ibl       # regenerate the four KTX2 assets
//! ```
//!
//! The math is a port of the retired pbr_ibl demo's CPU path (LearnOpenGL
//! conventions) so the baked set is a drop-in replacement for what that demo
//! computed at startup (#138): same cube-face layout, same GGX importance
//! sampling, same `roughness = mip / (levels-1)` prefilter convention.
//! Offline we afford more samples and bilinear equirect filtering.
//!
//! Deterministic by construction: no RNG (Hammersley/radical-inverse and
//! fixed-step integrals only), rayon parallelism over disjoint output rows
//! with ordered collection, and pinned zstd — a re-run is byte-identical.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use redlilium_core::math::{Vec2, Vec3};

const HDRI_NAME: &str = "spruit_sunrise_2k.hdr";

const BRDF_LUT_SIZE: u32 = 512;
const BRDF_LUT_SAMPLES: u32 = 1024;
const SKY_SIZE: u32 = 256;
const IRRADIANCE_SIZE: u32 = 32;
const IRRADIANCE_SAMPLE_DELTA: f32 = 0.05;
const PREFILTER_SIZE: u32 = 128;
const PREFILTER_MIPS: u32 = 8;
const PREFILTER_SAMPLES: u32 = 512;

const ZSTD_LEVEL: i32 = 19;

// vkFormat values (Vulkan core enum) — must stay in the engine's
// `map_vk_format` table (core/src/texture/ktx2.rs).
const VK_FORMAT_R16G16_SFLOAT: u32 = 83;
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

pub fn run() {
    let root = workspace_root();
    let hdri_path = root.join(".hdri").join(HDRI_NAME);
    if !hdri_path.exists() {
        eprintln!(
            "bake-ibl: source HDRI {} not found — run scripts/fetch-hdri.sh first; skipping \
             (the checked-in KTX2 set in std-assets stays as-is).",
            hdri_path.display()
        );
        return;
    }

    let out_dir = root.join("std-assets/textures/ibl");
    std::fs::create_dir_all(&out_dir).expect("create std-assets/textures/ibl");

    let start = std::time::Instant::now();
    let env = Equirect::load(&hdri_path);
    println!(
        "bake-ibl: source {} ({}x{}, {} pyramid levels, {:.1}s to decode)",
        HDRI_NAME,
        env.levels[0].width,
        env.levels[0].height,
        env.levels.len(),
        start.elapsed().as_secs_f32()
    );

    bake_one(&out_dir, "brdf_lut.ktx2", || Ktx2Output {
        vk_format: VK_FORMAT_R16G16_SFLOAT,
        bytes_per_texel: 4,
        width: BRDF_LUT_SIZE,
        height: BRDF_LUT_SIZE,
        faces: 1,
        levels: vec![bake_brdf_lut()],
    });
    bake_one(&out_dir, "sky_cube.ktx2", || {
        let levels = bake_sky_cube(&env);
        Ktx2Output {
            vk_format: VK_FORMAT_R16G16B16A16_SFLOAT,
            bytes_per_texel: 8,
            width: SKY_SIZE,
            height: SKY_SIZE,
            faces: 6,
            levels,
        }
    });
    bake_one(&out_dir, "irradiance_cube.ktx2", || Ktx2Output {
        vk_format: VK_FORMAT_R16G16B16A16_SFLOAT,
        bytes_per_texel: 8,
        width: IRRADIANCE_SIZE,
        height: IRRADIANCE_SIZE,
        faces: 6,
        levels: vec![bake_irradiance(&env)],
    });
    bake_one(&out_dir, "prefilter_cube.ktx2", || Ktx2Output {
        vk_format: VK_FORMAT_R16G16B16A16_SFLOAT,
        bytes_per_texel: 8,
        width: PREFILTER_SIZE,
        height: PREFILTER_SIZE,
        faces: 6,
        levels: bake_prefilter(&env),
    });
}

fn bake_one(out_dir: &Path, name: &str, bake: impl FnOnce() -> Ktx2Output) {
    let start = std::time::Instant::now();
    let output = bake();
    let bytes = output.to_ktx2_bytes();
    let path = out_dir.join(name);
    std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "bake-ibl: {name} — {} KiB ({:.1}s)",
        bytes.len() / 1024,
        start.elapsed().as_secs_f32()
    );
}

// === Source environment ===

/// The decoded equirectangular HDR source, RGB f32.
struct EquirectLevel {
    data: Vec<f32>,
    width: u32,
    height: u32,
}

impl EquirectLevel {
    fn texel(&self, x: u32, y: u32) -> Vec3 {
        let idx = ((y * self.width + x) * 3) as usize;
        Vec3::new(self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    /// Bilinear sample in the direction `dir` (same equirect mapping as the
    /// pbr_ibl demo, with filtering instead of nearest).
    fn sample(&self, dir: Vec3) -> Vec3 {
        let uv = Vec2::new(
            dir.z.atan2(dir.x) * 0.5 * std::f32::consts::FRAC_1_PI + 0.5,
            1.0 - (dir.y.asin() * std::f32::consts::FRAC_1_PI + 0.5),
        );
        let fx = (uv.x * self.width as f32 - 0.5).rem_euclid(self.width as f32);
        let fy = (uv.y * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx as u32, fy as u32);
        let x1 = (x0 + 1) % self.width; // longitude wraps
        let y1 = (y0 + 1).min(self.height - 1); // latitude clamps at the poles
        let (tx, ty) = (fx.fract(), fy.fract());
        let top = self.texel(x0, y0) * (1.0 - tx) + self.texel(x1, y0) * tx;
        let bottom = self.texel(x0, y1) * (1.0 - tx) + self.texel(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// 2×2 box-filtered next pyramid level (indices clamp at odd edges).
    fn downsample(&self) -> Self {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let (x2, y2) = (x * 2, y * 2);
                let x2b = (x2 + 1).min(self.width - 1);
                let y2b = (y2 + 1).min(self.height - 1);
                let sum = self.texel(x2, y2)
                    + self.texel(x2b, y2)
                    + self.texel(x2, y2b)
                    + self.texel(x2b, y2b);
                let avg = sum * 0.25;
                data.extend_from_slice(&[avg.x, avg.y, avg.z]);
            }
        }
        Self {
            data,
            width,
            height,
        }
    }
}

/// The source HDRI with a full box-filtered mip pyramid.
///
/// Monte-Carlo convolutions must not sample the finest level directly: an HDR
/// sun spans a handful of texels with radiance ~10⁴, so sparse importance
/// samples either miss it (energy loss at high roughness) or hit it rarely
/// (isolated firefly texels). Filtered importance sampling (Křivánek &
/// Colbert) fixes both: each sample reads the pyramid level whose texel solid
/// angle matches the solid angle the sample stands in for.
struct Equirect {
    levels: Vec<EquirectLevel>,
}

impl Equirect {
    fn load(path: &Path) -> Self {
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
            .to_rgb32f();
        let (width, height) = (img.width(), img.height());
        let mut levels = vec![EquirectLevel {
            data: img.into_raw(),
            width,
            height,
        }];
        while levels.last().unwrap().width > 1 || levels.last().unwrap().height > 1 {
            levels.push(levels.last().unwrap().downsample());
        }
        Self { levels }
    }

    /// Trilinear sample at a fractional pyramid `level`.
    fn sample_level(&self, dir: Vec3, level: f32) -> Vec3 {
        let level = level.clamp(0.0, (self.levels.len() - 1) as f32);
        let lo = level.floor() as usize;
        let hi = (lo + 1).min(self.levels.len() - 1);
        let t = level.fract();
        if t == 0.0 || lo == hi {
            return self.levels[lo].sample(dir);
        }
        self.levels[lo].sample(dir) * (1.0 - t) + self.levels[hi].sample(dir) * t
    }

    /// Average solid angle of one finest-level texel (sr).
    fn texel_solid_angle(&self) -> f32 {
        4.0 * std::f32::consts::PI / (self.levels[0].width * self.levels[0].height) as f32
    }

    /// Pyramid level whose texels cover `sample_solid_angle` steradians.
    fn level_for_solid_angle(&self, sample_solid_angle: f32) -> f32 {
        (0.5 * (sample_solid_angle / self.texel_solid_angle()).log2()).max(0.0)
    }
}

// === Shared sampling math (ported verbatim from the pbr_ibl demo) ===

/// Cube-face direction for texel `(x, y)` of `face` at `size` — the exact
/// layout the demo uploads and the resolve shader samples.
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

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn importance_sample_ggx(xi: Vec2, n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if n.z.abs() < 0.999 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let tangent = n.cross(&up).normalize();
    let bitangent = n.cross(&tangent);
    (tangent * h.x + bitangent * h.y + n * h.z).normalize()
}

// === BRDF integration LUT ===

/// Split-sum BRDF integral (Karis / LearnOpenGL): returns `(scale, bias)`
/// for `F = F0 * scale + bias`. Geometry term uses the IBL `k = a²/2`
/// remapping with `a = roughness`.
fn integrate_brdf(n_dot_v: f32, roughness: f32, samples: u32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v);
    let n = Vec3::new(0.0, 0.0, 1.0);

    let geometry_smith = |n_dot_v: f32, n_dot_l: f32| -> f32 {
        let k = (roughness * roughness) / 2.0;
        let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
        let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
        ggx_v * ggx_l
    };

    let mut scale = 0.0f32;
    let mut bias = 0.0f32;
    for i in 0..samples {
        let xi = hammersley(i, samples);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = (2.0 * v.dot(&h) * h - v).normalize();

        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);
        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-6);
            let fc = (1.0 - v_dot_h).powi(5);
            scale += (1.0 - fc) * g_vis;
            bias += fc * g_vis;
        }
    }
    (scale / samples as f32, bias / samples as f32)
}

/// 512×512 `Rg16Float` LUT: x = N·V, y = roughness (v = 0 row is the
/// smoothest). Tightly packed, single mip.
fn bake_brdf_lut() -> Vec<u8> {
    let rows: Vec<Vec<u8>> = (0..BRDF_LUT_SIZE)
        .into_par_iter()
        .map(|y| {
            let roughness = (y as f32 + 0.5) / BRDF_LUT_SIZE as f32;
            let mut row = Vec::with_capacity((BRDF_LUT_SIZE * 4) as usize);
            for x in 0..BRDF_LUT_SIZE {
                let n_dot_v = ((x as f32 + 0.5) / BRDF_LUT_SIZE as f32).max(1e-3);
                let (scale, bias) = integrate_brdf(n_dot_v, roughness, BRDF_LUT_SAMPLES);
                row.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                row.extend_from_slice(&half::f16::from_f32(bias).to_bits().to_le_bytes());
            }
            row
        })
        .collect();
    rows.concat()
}

// === Environment cubemaps ===

/// Largest magnitude an f16 (`Rgba16Float`) texel can hold. The IBL outputs
/// are stored as f16; the Spruit Sunrise sun disk resamples to a radiance above
/// this, so a raw `f16::from_f32` would round it to +Inf and bake a non-finite
/// texel into the asset — which then poisons the runtime (scene_color, bloom,
/// TAA) at the sun (#152). Clamp every radiance channel to this ceiling at pack
/// time so the baked cubemaps are always finite. Mirrors `F16_MAX` in
/// `shaders/library/math.slang`; the runtime `sanitize_finite` clamp is the
/// defense-in-depth backstop, this removes the root cause in the data.
const F16_MAX: f32 = 65504.0;

/// Pack one radiance color channel to clamped-finite f16 little-endian bytes.
/// NaN (a bad source pixel) → 0; anything above `F16_MAX` (or +Inf) → `F16_MAX`.
fn pack_radiance_f16(c: f32) -> [u8; 2] {
    let finite = if c.is_nan() {
        0.0
    } else {
        c.clamp(0.0, F16_MAX)
    };
    half::f16::from_f32(finite).to_bits().to_le_bytes()
}

/// One cube level: every face's texels produced by `shade`, parallel over
/// rows, packed as `Rgba16Float` (alpha 1) in face order.
fn bake_cube_level(size: u32, shade: impl Fn(Vec3) -> Vec3 + Sync) -> Vec<u8> {
    let rows: Vec<Vec<u8>> = (0..6 * size)
        .into_par_iter()
        .map(|face_row| {
            let (face, y) = (face_row / size, face_row % size);
            let mut row = Vec::with_capacity((size * 8) as usize);
            for x in 0..size {
                let color = shade(cubemap_dir(face, x, y, size));
                for c in [color.x, color.y, color.z, 1.0] {
                    row.extend_from_slice(&pack_radiance_f16(c));
                }
            }
            row
        })
        .collect();
    rows.concat()
}

/// Sky cubemap: mip 0 resampled from the equirect pyramid at the level
/// matching the cube's texel density (the 2k source is ~8× denser than the
/// 256 cube — sampling the finest level would alias), the rest of the chain
/// 2×2 box-filtered — a plain mip pyramid for the skybox pass (specular
/// convolution lives in the prefilter map).
fn bake_sky_cube(env: &Equirect) -> Vec<Vec<u8>> {
    let mip_count = 32 - SKY_SIZE.leading_zeros();
    let cube_texel_sa = 4.0 * std::f32::consts::PI / (6 * SKY_SIZE * SKY_SIZE) as f32;
    let level = env.level_for_solid_angle(cube_texel_sa);

    // f32 working pyramid per face, downsampled face-by-face.
    let mut faces: Vec<Vec<Vec3>> = (0..6u32)
        .map(|face| {
            let mut texels = Vec::with_capacity((SKY_SIZE * SKY_SIZE) as usize);
            for y in 0..SKY_SIZE {
                for x in 0..SKY_SIZE {
                    texels.push(env.sample_level(cubemap_dir(face, x, y, SKY_SIZE), level));
                }
            }
            texels
        })
        .collect();

    let mut levels = Vec::with_capacity(mip_count as usize);
    let mut size = SKY_SIZE;
    loop {
        let mut level = Vec::with_capacity((size * size * 6 * 8) as usize);
        for face in faces.iter() {
            for texel in face {
                for c in [texel.x, texel.y, texel.z, 1.0] {
                    level.extend_from_slice(&pack_radiance_f16(c));
                }
            }
        }
        levels.push(level);

        if size == 1 {
            break;
        }
        let next = size / 2;
        faces = faces
            .iter()
            .map(|face| {
                let mut out = Vec::with_capacity((next * next) as usize);
                for y in 0..next {
                    for x in 0..next {
                        let (x2, y2) = (x * 2, y * 2);
                        let sum = face[(y2 * size + x2) as usize]
                            + face[(y2 * size + x2 + 1) as usize]
                            + face[((y2 + 1) * size + x2) as usize]
                            + face[((y2 + 1) * size + x2 + 1) as usize];
                        out.push(sum * 0.25);
                    }
                }
                out
            })
            .collect();
        size = next;
    }
    assert_eq!(levels.len() as u32, mip_count);
    levels
}

/// Diffuse irradiance cubemap — the demo's fixed-step hemisphere integral.
///
/// Samples the pyramid level whose texels match the integration grid's cell
/// solid angle (≈ `delta²`), so the tiny ultra-bright sun contributes its
/// true energy instead of aliasing against the fixed step grid.
fn bake_irradiance(env: &Equirect) -> Vec<u8> {
    let level = env.level_for_solid_angle(IRRADIANCE_SAMPLE_DELTA * IRRADIANCE_SAMPLE_DELTA);
    bake_cube_level(IRRADIANCE_SIZE, |normal| {
        let up = if normal.y.abs() < 0.999 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let right = normal.cross(&up).normalize();
        let up = normal.cross(&right);

        let mut irradiance = Vec3::zeros();
        let mut samples = 0.0f32;
        let mut phi = 0.0f32;
        while phi < 2.0 * std::f32::consts::PI {
            let mut theta = 0.0f32;
            while theta < 0.5 * std::f32::consts::PI {
                let tangent = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let dir = tangent.x * right + tangent.y * up + tangent.z * normal;
                irradiance += env.sample_level(dir, level) * theta.cos() * theta.sin();
                samples += 1.0;
                theta += IRRADIANCE_SAMPLE_DELTA;
            }
            phi += IRRADIANCE_SAMPLE_DELTA;
        }
        std::f32::consts::PI * irradiance / samples
    })
}

/// Pre-filtered specular cubemap: GGX importance sampling with the demo's
/// `N = V = R` assumption; `roughness = mip / (PREFILTER_MIPS - 1)`.
///
/// Each sample reads the equirect pyramid level matched to its GGX pdf
/// (filtered importance sampling — see [`Equirect`]); `roughness = 0` copies
/// the finest level exactly.
fn bake_prefilter(env: &Equirect) -> Vec<Vec<u8>> {
    (0..PREFILTER_MIPS)
        .map(|mip| {
            let size = (PREFILTER_SIZE >> mip).max(1);
            let roughness = mip as f32 / (PREFILTER_MIPS - 1) as f32;
            bake_cube_level(size, |normal| {
                if roughness == 0.0 {
                    // Mirror level: no convolution, but resample at the level
                    // matching this cube's texel density (as bake_sky_cube
                    // does) so the sub-texel sun isn't grid-aliased.
                    let texel_sa = 4.0 * std::f32::consts::PI / (6 * size * size) as f32;
                    return env.sample_level(normal, env.level_for_solid_angle(texel_sa));
                }
                let v = normal;
                let mut prefiltered = Vec3::zeros();
                let mut total_weight = 0.0f32;
                for i in 0..PREFILTER_SAMPLES {
                    let xi = hammersley(i, PREFILTER_SAMPLES);
                    let h = importance_sample_ggx(xi, normal, roughness);
                    let l = (2.0 * v.dot(&h) * h - v).normalize();
                    let n_dot_l = normal.dot(&l).max(0.0);
                    if n_dot_l > 0.0 {
                        // With N = V the sample pdf is D·NdotH/(4·HdotV) = D/4.
                        let n_dot_h = normal.dot(&h).max(0.0);
                        let pdf = ggx_distribution(n_dot_h, roughness) * 0.25 + 1e-4;
                        let sample_sa = 1.0 / (PREFILTER_SAMPLES as f32 * pdf);
                        let level = env.level_for_solid_angle(sample_sa);
                        prefiltered += env.sample_level(l, level) * n_dot_l;
                        total_weight += n_dot_l;
                    }
                }
                prefiltered / total_weight.max(0.001)
            })
        })
        .collect()
}

/// GGX normal distribution `D(h)` with `a = roughness²` (Trowbridge-Reitz).
fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * denom * denom)
}

// === KTX2 writing ===

/// One baked texture ready for KTX2 serialization: uncompressed level
/// payloads (mip 0 first, faces concatenated within a level, no row padding).
struct Ktx2Output {
    vk_format: u32,
    bytes_per_texel: u32,
    width: u32,
    height: u32,
    faces: u32,
    levels: Vec<Vec<u8>>,
}

impl Ktx2Output {
    /// Serialize as KTX2 with Zstandard supercompression: 80-byte header,
    /// level index (logical mip order), a minimal basic DFD, then the level
    /// payloads stored smallest-mip-first as the spec requires. The output
    /// must stay parseable by `redlilium_core::texture::ktx2::parse_ktx2` —
    /// the round-trip unit test below holds the two ends together.
    fn to_ktx2_bytes(&self) -> Vec<u8> {
        for (mip, level) in self.levels.iter().enumerate() {
            let size = (self.width >> mip).max(1) as usize * (self.height >> mip).max(1) as usize;
            assert_eq!(
                level.len(),
                size * self.bytes_per_texel as usize * self.faces as usize,
                "level {mip} payload size mismatch"
            );
        }

        let compressed: Vec<Vec<u8>> = self
            .levels
            .iter()
            .map(|level| zstd::encode_all(&level[..], ZSTD_LEVEL).expect("zstd encode"))
            .collect();

        const MAGIC: [u8; 12] = [
            0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, b'\r', b'\n', 0x1A, b'\n',
        ];
        // totalSize + block header + basic body + one 16-byte sample per
        // channel (the spec requires sample information; ktx2check enforces).
        let channels = self.bytes_per_texel / 2; // 16-bit components throughout
        let dfd_len = 4 + 8 + 16 + 16 * channels;

        let level_count = self.levels.len();
        let index_off = 80usize;
        let dfd_off = index_off + level_count * 24;
        let data_off = dfd_off as u64 + dfd_len as u64;

        // Physical placement: smallest mip first (KTX2 streaming order).
        let mut offsets = vec![0u64; level_count];
        let mut cursor = data_off;
        for mip in (0..level_count).rev() {
            offsets[mip] = cursor;
            cursor += compressed[mip].len() as u64;
        }

        let mut out = Vec::with_capacity(cursor as usize);
        out.extend_from_slice(&MAGIC);
        for v in [
            self.vk_format,
            2, // typeSize: 16-bit components
            self.width,
            self.height,
            0, // pixelDepth
            0, // layerCount
            self.faces,
            level_count as u32,
            2, // supercompressionScheme: Zstandard
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(dfd_off as u32).to_le_bytes());
        out.extend_from_slice(&dfd_len.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // kvd offset + length
        out.extend_from_slice(&[0u8; 16]); // sgd offset + length
        assert_eq!(out.len(), index_off);

        for (mip, comp) in compressed.iter().enumerate() {
            out.extend_from_slice(&offsets[mip].to_le_bytes());
            out.extend_from_slice(&(comp.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.levels[mip].len() as u64).to_le_bytes());
        }

        // Basic DFD: vendor 0 / descriptor type 0, version 2; body =
        // colorModel RGBSDA, primaries BT709, transfer LINEAR, flags 0,
        // 1×1×1×1 texel block, bytesPlane0 = texel size; then one sample per
        // 16-bit SFLOAT channel (R, G, B, A in RGBSDA channel ids).
        assert_eq!(out.len(), dfd_off);
        out.extend_from_slice(&dfd_len.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&((24 + 16 * channels) as u16).to_le_bytes());
        let mut body = [0u8; 16];
        body[0] = 1; // KHR_DF_MODEL_RGBSDA
        body[1] = 1; // KHR_DF_PRIMARIES_BT709
        body[2] = 1; // KHR_DF_TRANSFER_LINEAR
        body[8] = self.bytes_per_texel as u8; // bytesPlane0
        out.extend_from_slice(&body);
        for channel in 0..channels {
            // RGBSDA channel ids: R=0, G=1, B=2, A=15.
            let channel_id: u32 = if channel == 3 { 15 } else { channel };
            // bitOffset(16) | (bitLength-1)(8) | type+qualifiers(8);
            // qualifiers FLOAT|SIGNED for SFLOAT data.
            let word0 = (16 * channel) | (15 << 16) | ((channel_id | 0xC0) << 24);
            out.extend_from_slice(&word0.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // samplePositions
            out.extend_from_slice(&0xBF80_0000u32.to_le_bytes()); // lower: -1.0f
            out.extend_from_slice(&0x3F80_0000u32.to_le_bytes()); // upper: +1.0f
        }

        for mip in (0..level_count).rev() {
            assert_eq!(out.len() as u64, offsets[mip]);
            out.extend_from_slice(&compressed[mip]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::texture::{TextureDimension, TextureFormat, ktx2::parse_ktx2};

    /// The writer and the engine parser agree end to end: a small 2-mip cube
    /// survives the write → zstd → parse round trip byte-for-byte.
    #[test]
    fn ktx2_writer_roundtrip() {
        let mip0: Vec<u8> = (0..4 * 4 * 6 * 8).map(|i| (i % 251) as u8).collect();
        let mip1: Vec<u8> = (0..2 * 2 * 6 * 8).map(|i| (i % 241) as u8).collect();
        let out = Ktx2Output {
            vk_format: VK_FORMAT_R16G16B16A16_SFLOAT,
            bytes_per_texel: 8,
            width: 4,
            height: 4,
            faces: 6,
            levels: vec![mip0.clone(), mip1.clone()],
        };
        let cpu = parse_ktx2(&out.to_ktx2_bytes()).expect("engine parser accepts our output");
        assert_eq!(cpu.format, TextureFormat::Rgba16Float);
        assert_eq!(cpu.dimension, TextureDimension::Cube);
        assert_eq!((cpu.width, cpu.height), (4, 4));
        assert_eq!(cpu.mip_level_count, 2);
        assert_eq!(cpu.layer_count(), 6);
        assert_eq!(&cpu.data[..mip0.len()], &mip0[..]);
        assert_eq!(&cpu.data[mip0.len()..], &mip1[..]);
    }

    /// The checked-in baked set stays parseable by the engine with the exact
    /// shapes #138 consumes. Skips silently only if a file is absent (fresh
    /// clone before the bake artifacts were committed).
    #[test]
    fn baked_ibl_set_parses() {
        let dir = workspace_root().join("std-assets/textures/ibl");
        let cases: [(&str, TextureFormat, TextureDimension, u32, u32); 4] = [
            (
                "brdf_lut.ktx2",
                TextureFormat::Rg16Float,
                TextureDimension::D2,
                1,
                1,
            ),
            (
                "sky_cube.ktx2",
                TextureFormat::Rgba16Float,
                TextureDimension::Cube,
                6,
                9,
            ),
            (
                "irradiance_cube.ktx2",
                TextureFormat::Rgba16Float,
                TextureDimension::Cube,
                6,
                1,
            ),
            (
                "prefilter_cube.ktx2",
                TextureFormat::Rgba16Float,
                TextureDimension::Cube,
                6,
                8,
            ),
        ];
        for (name, format, dimension, layers, mips) in cases {
            let path = dir.join(name);
            let Ok(bytes) = std::fs::read(&path) else {
                eprintln!("{} absent — skipping", path.display());
                continue;
            };
            let cpu = parse_ktx2(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(cpu.format, format, "{name}");
            assert_eq!(cpu.dimension, dimension, "{name}");
            assert_eq!(cpu.layer_count(), layers, "{name}");
            assert_eq!(cpu.mip_level_count, mips, "{name}");
        }
    }

    /// Analytic sanity for the split-sum integral: near-mirror roughness
    /// reflects almost all of F0 (scale ≈ 1, bias ≈ 0), the pair conserves
    /// energy everywhere, and bias stays a small additive term.
    #[test]
    fn brdf_integral_reference_points() {
        let (scale, bias) = integrate_brdf(0.5, 0.02, 256);
        assert!((0.9..=1.0).contains(&scale), "near-mirror scale: {scale}");
        assert!((0.0..0.05).contains(&bias), "near-mirror bias: {bias}");

        for &(n_dot_v, roughness) in &[(0.1f32, 0.1f32), (0.5, 0.5), (0.9, 0.9), (0.3, 0.7)] {
            let (scale, bias) = integrate_brdf(n_dot_v, roughness, 256);
            assert!(
                scale >= 0.0 && bias >= 0.0 && scale + bias <= 1.0 + 1e-3,
                "energy conservation at ({n_dot_v}, {roughness}): scale {scale}, bias {bias}"
            );
        }

        // Rougher surfaces scatter more: the specular scale at fixed N·V
        // decreases monotonically in roughness.
        let s_smooth = integrate_brdf(0.7, 0.1, 256).0;
        let s_rough = integrate_brdf(0.7, 0.9, 256).0;
        assert!(
            s_smooth > s_rough,
            "scale must fall with roughness: {s_smooth} vs {s_rough}"
        );
    }
    /// Firefly / energy guard over the checked-in prefilter chain. Without
    /// filtered importance sampling the pinned sun (radiance ~3·10⁴) leaves
    /// isolated firefly texels in the middle mips and loses its energy
    /// entirely in the wide-roughness tail — both regressions show up here as
    /// either a runaway max or a collapsing mean. Thresholds are tied to the
    /// pinned Spruit Sunrise source; re-tune them when swapping the HDRI.
    #[test]
    fn prefilter_chain_energy_is_stable() {
        let path = workspace_root().join("std-assets/textures/ibl/prefilter_cube.ktx2");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("{} absent — skipping", path.display());
            return;
        };
        let cpu = parse_ktx2(&bytes).expect("prefilter parses");
        let mut offset = 0usize;
        for mip in 0..cpu.mip_level_count {
            let size = (cpu.width >> mip).max(1) as usize;
            let texels = size * size * 6;
            let mut max_lum = 0.0f32;
            let mut mean_lum = 0.0f64;
            for t in 0..texels {
                let base = offset + t * 8;
                let r = half::f16::from_le_bytes([cpu.data[base], cpu.data[base + 1]]).to_f32();
                let g = half::f16::from_le_bytes([cpu.data[base + 2], cpu.data[base + 3]]).to_f32();
                let b = half::f16::from_le_bytes([cpu.data[base + 4], cpu.data[base + 5]]).to_f32();
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                max_lum = max_lum.max(lum);
                mean_lum += lum as f64;
            }
            mean_lum /= texels as f64;
            offset += texels * 8;

            // Energy stays put across the chain (convolution preserves the
            // mean); losing the sun collapses this to ~0.3.
            assert!(
                (0.5..3.0).contains(&mean_lum),
                "mip {mip}: mean luminance {mean_lum:.2} outside the stable band"
            );
            // The narrow-roughness mips legitimately keep a bright compact sun
            // highlight; from mip 3 on the lobe is wide enough that any texel
            // this hot is a firefly.
            if mip >= 3 {
                assert!(
                    max_lum < 100.0,
                    "mip {mip}: max luminance {max_lum:.0} — firefly texel"
                );
            }
        }
    }

    /// The sky cubemap (what the skybox pass samples) must be finite in every
    /// texel. The Spruit Sunrise sun disk resamples to a radiance above the f16
    /// ceiling, so before the pack-time clamp (`pack_radiance_f16`) mip 0 held a
    /// +Inf texel that poisoned scene_color / bloom / the display tonemap on the
    /// sun (#152). This guards the root fix: any future overflow or a NaN source
    /// pixel re-baked into the asset trips here instead of shipping.
    #[test]
    fn sky_cube_texels_are_finite() {
        let path = workspace_root().join("std-assets/textures/ibl/sky_cube.ktx2");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("{} absent — skipping", path.display());
            return;
        };
        let cpu = parse_ktx2(&bytes).expect("sky cube parses");
        assert_eq!(cpu.format, TextureFormat::Rgba16Float);
        let mut peak = 0.0f32;
        let mut offset = 0usize;
        for mip in 0..cpu.mip_level_count {
            let size = (cpu.width >> mip).max(1) as usize;
            let texels = size * size * 6;
            for t in 0..texels {
                let base = offset + t * 8;
                for ch in 0..3 {
                    let c = half::f16::from_le_bytes([
                        cpu.data[base + ch * 2],
                        cpu.data[base + ch * 2 + 1],
                    ])
                    .to_f32();
                    assert!(
                        c.is_finite() && c <= F16_MAX,
                        "mip {mip} texel {t} channel {ch}: non-finite / over-range value {c}"
                    );
                    peak = peak.max(c);
                }
            }
            offset += texels * 8;
        }
        // The sun highlight survives the clamp (it is pinned near the ceiling,
        // not zeroed) — a collapse to ~0 would mean we killed the glow, not the Inf.
        assert!(
            peak > 1.0,
            "sky cube lost its bright highlight (peak {peak})"
        );
    }
}
