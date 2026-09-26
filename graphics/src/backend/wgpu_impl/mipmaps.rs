//! Private render-based lowering of a graph mip-generation operation.

use super::{WgpuBackend, conversion::convert_texture_format};
use crate::{Texture, TextureFormat, backend::GpuTexture};

impl WgpuBackend {
    pub(crate) fn supports_mipgen(&self, format: TextureFormat) -> bool {
        if format.is_compressed()
            || format.is_depth_stencil()
            || format.is_integer()
            || format == TextureFormat::Bgra10a2Unorm
        {
            return false;
        }
        let format = convert_texture_format(format);
        let features = self.device.features();
        features.contains(format.required_features())
            && format
                .guaranteed_format_features(features)
                .allowed_usages
                .contains(
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                )
    }

    fn mip_pipeline(&self, format: wgpu::TextureFormat, cube: bool) -> wgpu::RenderPipeline {
        let mut cache = self.mip_pipelines.lock();
        cache
            .entry((format, cube))
            .or_insert_with(|| {
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("mip reduction"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cube {
                                include_str!("../mipmaps_cube.wgsl")
                            } else {
                                include_str!("mipmaps.wgsl")
                            }
                            .into(),
                        ),
                    });
                let bindings =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("mip source"),
                            entries: &[wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: false,
                                    },
                                    view_dimension: if cube {
                                        wgpu::TextureViewDimension::D2Array
                                    } else {
                                        wgpu::TextureViewDimension::D2
                                    },
                                    multisampled: false,
                                },
                                count: None,
                            }],
                        });
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("mip pipeline layout"),
                        bind_group_layouts: &[&bindings],
                        immediate_size: 0,
                    });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("mip reduction"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: Default::default(),
                        depth_stencil: None,
                        multisample: Default::default(),
                        multiview_mask: None,
                        cache: None,
                    })
            })
            .clone()
    }

    pub(super) fn encode_mipmaps(&self, encoder: &mut wgpu::CommandEncoder, target: &Texture) {
        if target.mip_level_count() <= 1 {
            return;
        }
        let GpuTexture::Wgpu { texture, .. } = target.gpu_handle() else {
            return;
        };
        let format = convert_texture_format(target.format());
        let cube = target.dimension().is_cubemap();
        let pipeline = self.mip_pipeline(format, cube);
        // Scratch keeps sampled/render-attachment usage private. The public
        // resource needs only COPY_SRC/DST, on both Vulkan and wgpu. wgpu retains
        // all these handles in the command buffer until submission completes.
        let scratch = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mip scratch"),
            size: texture.size(),
            mip_level_count: target.mip_level_count(),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            scratch.as_image_copy(),
            texture.size(),
        );
        for mip in 1..target.mip_level_count() {
            for layer in 0..texture.depth_or_array_layers() {
                let view = |level| {
                    scratch.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                };
                let src = if cube {
                    scratch.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        base_mip_level: mip - 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                } else {
                    view(mip - 1)
                };
                let dst = view(mip);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mip source"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src),
                    }],
                });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("generate mip"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &dst,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..3, layer..layer + 1);
                }
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    mip_level: mip,
                    ..scratch.as_image_copy()
                },
                wgpu::TexelCopyTextureInfo {
                    mip_level: mip,
                    ..texture.as_image_copy()
                },
                wgpu::Extent3d {
                    width: (target.width() >> mip).max(1),
                    height: (target.height() >> mip).max(1),
                    depth_or_array_layers: texture.depth_or_array_layers(),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mip_shader_validates_without_optional_capabilities() {
        for source in [
            include_str!("mipmaps.wgsl"),
            include_str!("../mipmaps_cube.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
