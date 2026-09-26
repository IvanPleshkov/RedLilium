//! GPU environment convolution built from ordinary render-graph passes.
//!
//! Reuse an [`EnvironmentFilter`] across probes and an [`EnvironmentFilterJob`]
//! across updates of one captured cubemap. Preparing a job only creates resources;
//! [`EnvironmentFilterJob::add_to_graph`] schedules every upload and draw.
//!
//! ```no_run
//! use std::sync::Arc;
//! use redlilium_graphics::{GraphicsDevice, GraphicsError, RenderGraph, PassHandle, Texture};
//! use redlilium_graphics::ibl::{EnvironmentFilter, EnvironmentFilterSettings};
//!
//! fn prepare_environment(device: Arc<GraphicsDevice>, captured: Arc<Texture>,
//!     graph: &mut RenderGraph, capture_and_mips: PassHandle) -> Result<(), GraphicsError> {
//!     let filter = EnvironmentFilter::new(device)?; // cache across probes
//!     let job = filter.prepare(captured, EnvironmentFilterSettings::default())?;
//!     let ready = job.add_to_graph(graph, &[capture_and_mips]);
//!     // Bind job.specular(), job.diffuse() and the existing BRDF LUT in lighting.
//!     // Use job.max_reflection_lod(); add_dependency(lighting, ready).
//!     // Keep the job to update its textures after the next capture.
//!     Ok(())
//! }
//! ```

use crate::{
    BindingGroupDescriptor, BindingLayout, Buffer, BufferDescriptor, BufferUsage, ColorAttachment,
    CullMode, DrawCommand, GraphicsDevice, GraphicsError, GraphicsPass, Material,
    MaterialDescriptor, MaterialInstance, Mesh, MeshDescriptor, PassHandle, RenderGraph,
    RenderTarget, RenderTargetConfig, Sampler, SamplerDescriptor, ShaderSource, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexBufferLayout, VertexLayout,
};
use std::sync::Arc;

/// Quality and resolution of a filtered environment. Output is linear RGBA16F.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvironmentFilterSettings {
    /// Base face size of the full specular mip chain, at least one texel.
    pub specular_size: u32,
    /// Face size of the single-level diffuse map, at least one texel.
    pub diffuse_size: u32,
    /// GGX samples per texel, 1..=4096 (mip zero is directly resampled).
    pub specular_samples: u32,
    /// Cosine-weighted hemisphere samples per texel, 1..=4096.
    pub diffuse_samples: u32,
}
impl Default for EnvironmentFilterSettings {
    fn default() -> Self {
        Self {
            specular_size: 128,
            diffuse_size: 32,
            specular_samples: 512,
            diffuse_samples: 256,
        }
    }
}

/// Shared pipelines for GGX specular and Lambertian diffuse IBL convolution.
/// No work is submitted by this object. Use the existing BRDF LUT when sampling
/// its specular result; roughness maps linearly to the output mip index.
#[derive(Debug)]
pub struct EnvironmentFilter {
    device: Arc<GraphicsDevice>,
    layout: Arc<BindingLayout>,
    specular: Arc<Material>,
    diffuse: Arc<Material>,
    sampler: Arc<Sampler>,
    triangle: Arc<Mesh>,
}

impl EnvironmentFilter {
    /// Create the shared pipelines, sampler and fullscreen geometry.
    pub fn new(device: Arc<GraphicsDevice>) -> Result<Self, GraphicsError> {
        let layout = Arc::new(
            BindingLayout::new()
                .with_texture_cube(0)
                .with_sampler(1)
                .with_uniform_buffer(2),
        );
        let vertex_layout = Arc::new(VertexLayout::new().with_buffer(VertexBufferLayout::new(4)));
        let material = |entry| {
            let code = include_bytes!("filter.wgsl").to_vec();
            let mut desc = MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(code.clone(), "vs"))
                .with_shader(ShaderSource::fragment(code, entry))
                .with_color_format(TextureFormat::Rgba16Float)
                .with_vertex_layout(vertex_layout.clone());
            desc.binding_layouts = vec![layout.clone()];
            desc.raster.cull_mode = CullMode::None;
            device.create_material(&desc)
        };
        Ok(Self {
            specular: material("specular")?,
            diffuse: material("diffuse")?,
            sampler: device.create_sampler(&SamplerDescriptor::linear())?,
            triangle: device
                .create_mesh(&MeshDescriptor::new(vertex_layout).with_vertex_count(3))?,
            layout,
            device,
        })
    }

    /// Allocate reusable output textures and bindings for one environment.
    ///
    /// `source` must be a square, single-sampled Cube with TEXTURE_BINDING and a
    /// complete ordinary mip chain. Supported inputs are RGBA16F and normalized
    /// RGBA8/BGRA8 (including sRGB). RGBA16F is recommended for HDR probes.
    /// All source levels must be populated before the filtering passes execute;
    /// generate ordinary mips after capture using a graph transfer operation.
    /// The source must contain finite, nonnegative radiance representable in f16.
    pub fn prepare(
        &self,
        source: Arc<Texture>,
        settings: EnvironmentFilterSettings,
    ) -> Result<EnvironmentFilterJob, GraphicsError> {
        let bad = |s: &str| GraphicsError::InvalidParameter(format!("environment filter: {s}"));
        if !Arc::ptr_eq(source.device(), &self.device) {
            return Err(bad("source belongs to another device"));
        }
        if source.dimension() != TextureDimension::Cube
            || source.width() == 0
            || source.width() != source.height()
            || source.sample_count() != 1
            || !source.usage().contains(TextureUsage::TEXTURE_BINDING)
            || source.mip_level_count() != 32 - source.width().leading_zeros()
        {
            return Err(bad(
                "source must be a sampled, square cubemap with a complete mip chain",
            ));
        }
        if !matches!(
            source.format(),
            TextureFormat::Rgba16Float
                | TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba8UnormSrgb
                | TextureFormat::Bgra8Unorm
                | TextureFormat::Bgra8UnormSrgb
        ) {
            return Err(bad("source must be RGBA16F or normalized RGBA8/BGRA8"));
        }
        for size in [settings.specular_size, settings.diffuse_size] {
            if size == 0 || size > self.device.capabilities().max_texture_dimension {
                return Err(bad("output size exceeds device limits or is zero"));
            }
        }
        for samples in [settings.specular_samples, settings.diffuse_samples] {
            if !(1..=4096).contains(&samples) {
                return Err(bad("sample count must be in 1..=4096"));
            }
        }
        let levels = 32 - settings.specular_size.leading_zeros();
        let output = |size, levels, label| {
            self.device.create_texture(
                &TextureDescriptor::new_cube(
                    size,
                    TextureFormat::Rgba16Float,
                    TextureUsage::TEXTURE_BINDING
                        | TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::COPY_SRC,
                )
                .with_mip_levels(levels)
                .with_label(label),
            )
        };
        let specular = output(settings.specular_size, levels, "IBL specular")?;
        let diffuse = output(settings.diffuse_size, 1, "IBL diffuse")?;
        let mut draws = Vec::new();
        // The graph draw API uses a mesh even for a vertex-index fullscreen triangle.
        let mut uploads = vec![(
            self.triangle.vertex_buffer(0).unwrap().clone(),
            Arc::<[u8]>::from([0u8; 12]),
        )];
        for (target, material, samples) in [
            (&specular, &self.specular, settings.specular_samples),
            (&diffuse, &self.diffuse, settings.diffuse_samples),
        ] {
            for mip in 0..target.mip_level_count() {
                let roughness = mip as f32 / (levels - 1).max(1) as f32;
                let size = (target.width() >> mip).max(1);
                let bytes = [
                    size.to_le_bytes(),
                    samples.to_le_bytes(),
                    roughness.to_le_bytes(),
                    0u32.to_le_bytes(),
                ]
                .concat();
                let uniform = self.device.create_buffer(&BufferDescriptor::new(
                    16,
                    BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                ))?;
                uploads.push((uniform.clone(), Arc::<[u8]>::from(bytes)));
                let bindings = self.device.create_binding_group(
                    self.layout.clone(),
                    BindingGroupDescriptor::new()
                        .with_texture(0, source.clone())
                        .with_sampler(1, self.sampler.clone())
                        .with_buffer(2, uniform),
                )?;
                let instance =
                    Arc::new(MaterialInstance::new(material.clone()).with_binding_group(bindings));
                for face in 0..6 {
                    draws.push(FilterDraw {
                        target: RenderTarget::from_texture_layer(target.clone(), mip, face),
                        instance: instance.clone(),
                        face,
                    });
                }
            }
        }
        Ok(EnvironmentFilterJob {
            specular,
            diffuse,
            triangle: self.triangle.clone(),
            draws,
            uploads,
        })
    }
}

#[derive(Debug)]
struct FilterDraw {
    target: RenderTarget,
    instance: Arc<MaterialInstance>,
    face: u32,
}

/// Reusable filtering resources. Results are valid only after the scheduled
/// passes execute. Updating overwrites the same textures; for incremental probe
/// updates keep separate front/back jobs and publish only a completed result.
/// The graph retains all resources even if the job is dropped before submission.
#[derive(Debug)]
pub struct EnvironmentFilterJob {
    specular: Arc<Texture>,
    diffuse: Arc<Texture>,
    triangle: Arc<Mesh>,
    draws: Vec<FilterDraw>,
    uploads: Vec<(Arc<Buffer>, Arc<[u8]>)>,
}
impl EnvironmentFilterJob {
    /// GGX prefiltered radiance; sample at `roughness * max_reflection_lod()`.
    pub fn specular(&self) -> &Arc<Texture> {
        &self.specular
    }
    /// Diffuse irradiance divided by PI, matching the deferred renderer's
    /// `diffuse = irradiance * albedo` convention.
    pub fn diffuse(&self) -> &Arc<Texture> {
        &self.diffuse
    }
    /// Highest specular mip, matching the existing deferred IBL convention.
    pub fn max_reflection_lod(&self) -> f32 {
        (self.specular.mip_level_count() - 1) as f32
    }

    /// Append a complete update. Dependencies must belong to `graph` and should
    /// include capture/mip generation. Make consumers depend on the returned
    /// completion pass. Across graphs, submit capture → filter → lighting in order.
    /// Pipelines, textures and bindings are reused; no CPU waits or GPU submissions.
    pub fn add_to_graph(&self, graph: &mut RenderGraph, dependencies: &[PassHandle]) -> PassHandle {
        let mut upload = TransferPass::new("IBL parameters".into());
        upload.set_transfer_config(
            TransferConfig::new().with_operations(self.uploads.iter().map(|(buffer, data)| {
                TransferOperation::write_buffer(buffer.clone(), 0, data.clone())
            })),
        );
        let mut last = graph.add_transfer_pass(upload);
        for &dependency in dependencies {
            graph.add_dependency(last, dependency);
        }
        // Explicit order also accommodates the graph's whole-image tracking.
        for draw in &self.draws {
            let mut pass = GraphicsPass::new("IBL convolution".into());
            pass.set_render_targets(
                RenderTargetConfig::new().with_color(ColorAttachment::new(draw.target.clone())),
            );
            pass.add_draw_command(
                DrawCommand::new(self.triangle.clone(), draw.instance.clone())
                    .with_first_instance(draw.face),
            );
            let next = graph.add_graphics_pass(pass);
            graph.add_dependency(next, last);
            last = next;
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> Arc<GraphicsDevice> {
        crate::GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }
    #[test]
    fn shader_needs_no_optional_capabilities() {
        let module = naga::front::wgsl::parse_str(include_str!("filter.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
    #[test]
    fn rejects_invalid_sources_and_settings_before_building_jobs() {
        let device = device();
        let filter = EnvironmentFilter::new(device.clone()).unwrap();
        let desc = TextureDescriptor::new_cube(
            8,
            TextureFormat::Rgba16Float,
            TextureUsage::TEXTURE_BINDING,
        )
        .with_mip_levels(4);
        let valid = device.create_texture(&desc).unwrap();
        for settings in [
            EnvironmentFilterSettings {
                specular_size: 0,
                ..Default::default()
            },
            EnvironmentFilterSettings {
                diffuse_size: device.capabilities().max_texture_dimension + 1,
                ..Default::default()
            },
            EnvironmentFilterSettings {
                specular_samples: 0,
                ..Default::default()
            },
            EnvironmentFilterSettings {
                diffuse_samples: 4097,
                ..Default::default()
            },
        ] {
            assert!(filter.prepare(valid.clone(), settings).is_err());
        }
        for invalid in [
            desc.clone().with_mip_levels(1),
            desc.clone().with_dimension(TextureDimension::D2),
            TextureDescriptor {
                sample_count: 4,
                ..desc.clone()
            },
            TextureDescriptor {
                format: TextureFormat::Rgba32Float,
                ..desc.clone()
            },
            TextureDescriptor {
                usage: TextureUsage::COPY_SRC,
                ..desc.clone()
            },
        ] {
            // Keep testing the filter boundary independently of public descriptor validation.
            let texture = Arc::new(Texture::new(
                device.clone(),
                invalid,
                crate::backend::GpuTexture::Dummy,
            ));
            assert!(filter.prepare(texture, Default::default()).is_err());
        }
        let other = EnvironmentFilter::new(self::device()).unwrap();
        assert!(other.prepare(valid, Default::default()).is_err());
    }
}
