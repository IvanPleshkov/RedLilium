//! Validate the complete command stream before any backend submits work.
use super::{Pass, RenderGraph};
use crate::*;
use std::sync::Arc;

fn invalid(message: impl Into<String>) -> GraphicsError {
    GraphicsError::InvalidParameter(message.into())
}
fn context(error: GraphicsError, prefix: &str) -> GraphicsError {
    match error {
        GraphicsError::InvalidParameter(message) => invalid(format!("{prefix}: {message}")),
        GraphicsError::FeatureNotSupported(message) => {
            GraphicsError::FeatureNotSupported(format!("{prefix}: {message}"))
        }
        other => other,
    }
}
fn owner(device: &Arc<GraphicsDevice>, other: &Arc<GraphicsDevice>) -> Result<(), GraphicsError> {
    if Arc::ptr_eq(device, other) {
        Ok(())
    } else {
        Err(invalid("resource belongs to another device"))
    }
}
fn buffer(device: &Arc<GraphicsDevice>, buffer: &Buffer) -> Result<(), GraphicsError> {
    owner(device, buffer.device())?;
    if buffer.is_map_pending() {
        return Err(invalid("buffer has a pending CPU mapping"));
    }
    Ok(())
}
fn bindings(
    device: &Arc<GraphicsDevice>,
    instance: &MaterialInstance,
    offsets: &[Vec<u32>],
    compute: bool,
    mesh: bool,
) -> Result<(), GraphicsError> {
    let material = instance.material();
    owner(device, material.device())?;
    if material
        .shaders()
        .iter()
        .any(|s| s.stage == ShaderStage::Compute)
        != compute
        || material.descriptor().uses_mesh_shading() != mesh
    {
        return Err(invalid("pipeline stage does not match the command"));
    }
    if material.binding_layouts().len() != instance.binding_groups().len() {
        return Err(invalid("binding group count does not match pipeline"));
    }
    if offsets
        .iter()
        .skip(instance.binding_groups().len())
        .any(|v| !v.is_empty())
    {
        return Err(invalid("dynamic offsets reference a nonexistent group"));
    }
    for (index, (expected, group)) in material
        .binding_layouts()
        .iter()
        .zip(instance.binding_groups())
        .enumerate()
    {
        owner(device, group.device())?;
        if !Arc::ptr_eq(expected, group.layout())
            && (expected.entries.len() != group.layout().entries.len()
                || expected.entries.iter().any(|entry| {
                    !group.layout().entries.iter().any(|actual| {
                        actual.binding == entry.binding
                            && actual.binding_type == entry.binding_type
                            && actual.visibility == entry.visibility
                    })
                }))
        {
            return Err(invalid(format!(
                "binding group {index} layout does not match pipeline"
            )));
        }
        for entry in group.entries() {
            if let BoundResource::Buffer(b) | BoundResource::BufferRange { buffer: b, .. } =
                &entry.resource
            {
                buffer(device, b)?;
            }
        }
        let dynamic = group.dynamic_bindings();
        let supplied = offsets.get(index).map(Vec::as_slice).unwrap_or_default();
        if dynamic.len() != supplied.len() {
            return Err(invalid(format!(
                "binding group {index} dynamic offset count mismatch"
            )));
        }
        for (&entry_index, &offset) in dynamic.iter().zip(supplied) {
            let resource = &group.entries()[entry_index].resource;
            let (b, base, size) = match resource {
                BoundResource::Buffer(b) => (b, 0, b.size()),
                BoundResource::BufferRange {
                    buffer,
                    offset,
                    size,
                } => (buffer, *offset, *size),
                _ => return Err(invalid("dynamic binding requires a buffer")),
            };
            if u64::from(offset) % device.capabilities().min_uniform_buffer_offset_alignment != 0
                || base
                    .checked_add(u64::from(offset))
                    .and_then(|v| v.checked_add(size))
                    .is_none_or(|end| end > b.size())
            {
                return Err(invalid(
                    "dynamic uniform offset is misaligned or exceeds buffer",
                ));
            }
        }
    }
    Ok(())
}
fn targets(device: &Arc<GraphicsDevice>, config: &RenderTargetConfig) -> Result<(), GraphicsError> {
    config.validate()?;
    if config.color_attachments.len() > 8 {
        return Err(invalid("at most 8 color attachments are supported"));
    }
    for target in config
        .color_attachments
        .iter()
        .flat_map(|c| std::iter::once(&c.target).chain(c.resolve_target.as_ref()))
        .chain(config.depth_stencil_attachment.iter().map(|d| &d.target))
    {
        if let Some(t) = target.texture() {
            owner(device, t.device())?;
        }
    }
    Ok(())
}
fn pipeline_targets(
    material: &MaterialInstance,
    config: &RenderTargetConfig,
) -> Result<(), GraphicsError> {
    let d = material.material().descriptor();
    if d.depth.is_some() != config.depth_stencil_attachment.is_some() {
        return Err(invalid(
            "pipeline depth state and depth attachment must both be present or absent",
        ));
    }
    if !config.has_attachments() {
        return Err(invalid("draw requires at least one attachment"));
    }
    if d.color_formats.len() != config.color_attachments.len()
        || d.color_formats
            .iter()
            .zip(&config.color_attachments)
            .any(|(format, a)| {
                *format != a.target.format() || d.sample_count != a.target.sample_count()
            })
    {
        return Err(invalid(
            "pipeline color formats or sample count do not match attachments",
        ));
    }
    if let Some(depth) = &d.depth {
        let target = config
            .depth_stencil_attachment
            .as_ref()
            .ok_or_else(|| invalid("pipeline requires a depth attachment"))?;
        if depth.format != target.target.format()
            || d.sample_count != target.target.sample_count()
            || (depth.write && target.effective_read_only())
        {
            return Err(invalid("pipeline depth state does not match attachment"));
        }
    }
    Ok(())
}
impl RenderGraph {
    pub(crate) fn validate_commands(
        &self,
        device: &Arc<GraphicsDevice>,
    ) -> Result<(), GraphicsError> {
        for pass in self.passes() {
            let result = (|| {
                match pass {
                    Pass::Graphics(p) => {
                        if let Some(config) = p.render_targets() {
                            targets(device, config)?;
                        }
                        if p.has_draws()
                            || p.has_indirect_draws()
                            || !p.mesh_tasks_commands().is_empty()
                            || !p.mesh_tasks_indirect_commands().is_empty()
                        {
                            let config = p
                                .render_targets()
                                .ok_or_else(|| invalid("draw requires render targets"))?;
                            for (index, draw) in p.raster_draws().enumerate() {
                                let validate = || {
                                    bindings(
                                        device,
                                        draw.material,
                                        draw.dynamic_offsets,
                                        false,
                                        false,
                                    )?;
                                    pipeline_targets(draw.material, config)?;
                                    owner(device, draw.mesh.device())?;
                                    if !DrawCommand::is_compatible(draw.mesh, draw.material) {
                                        return Err(invalid("mesh layout does not match pipeline"));
                                    }
                                    for b in draw.mesh.vertex_buffers() {
                                        buffer(device, b)?;
                                    }
                                    if let Some(b) = draw.mesh.index_buffer() {
                                        buffer(device, b)?;
                                    }
                                    if draw
                                        .first_instance
                                        .checked_add(draw.instance_count)
                                        .is_none()
                                    {
                                        return Err(invalid("instance range overflows"));
                                    }
                                    if let Some(indirect) = draw.indirect {
                                        indirect.validate()?;
                                        buffer(device, &indirect.indirect_buffer)?;
                                    }
                                    Ok(())
                                };
                                validate().map_err(|e| context(e, &format!("draw {index}")))?;
                            }
                            for draw in p.mesh_tasks_commands() {
                                bindings(
                                    device,
                                    &draw.material,
                                    &draw.dynamic_offsets,
                                    false,
                                    true,
                                )?;
                                pipeline_targets(&draw.material, config)?;
                                let caps = device.capabilities();
                                if !caps.mesh_shading {
                                    return Err(GraphicsError::FeatureNotSupported(
                                        "mesh shading".into(),
                                    ));
                                }
                                if draw
                                    .group_count
                                    .iter()
                                    .zip(caps.mesh_tasks_max_group_count)
                                    .any(|(&n, max)| n > max)
                                    || draw
                                        .group_count
                                        .iter()
                                        .try_fold(1u64, |v, &n| v.checked_mul(n.into()))
                                        .is_none_or(|n| {
                                            n > u64::from(caps.mesh_tasks_max_total_count)
                                        })
                                {
                                    return Err(invalid(
                                        "mesh task group count exceeds device limits",
                                    ));
                                }
                            }
                            for draw in p.mesh_tasks_indirect_commands() {
                                bindings(
                                    device,
                                    &draw.material,
                                    &draw.dynamic_offsets,
                                    false,
                                    true,
                                )?;
                                pipeline_targets(&draw.material, config)?;
                                buffer(device, &draw.indirect_buffer)?;
                                if !device.capabilities().mesh_shading {
                                    return Err(GraphicsError::FeatureNotSupported(
                                        "mesh shading".into(),
                                    ));
                                }
                                draw.validate()?;
                            }
                        }
                    }
                    Pass::Compute(p) => {
                        for (index, dispatch) in p.dispatch_commands().iter().enumerate() {
                            let validate = || {
                                if !device.capabilities().compute_shaders {
                                    return Err(GraphicsError::FeatureNotSupported(
                                        "compute shaders".into(),
                                    ));
                                }
                                bindings(device, &dispatch.material, &[], true, false)?;
                                if [
                                    dispatch.workgroup_count_x,
                                    dispatch.workgroup_count_y,
                                    dispatch.workgroup_count_z,
                                ]
                                .iter()
                                .zip(device.capabilities().max_compute_workgroups)
                                .any(|(&n, max)| n > max)
                                {
                                    return Err(invalid("dispatch count exceeds device limits"));
                                }
                                Ok(())
                            };
                            validate().map_err(|e| context(e, &format!("dispatch {index}")))?;
                        }
                    }
                    _ => {}
                }
                Ok(())
            })();
            result.map_err(|e| context(e, &format!("pass {:?}", pass.name())))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> Arc<GraphicsDevice> {
        GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }
    fn instance(device: &Arc<GraphicsDevice>, layout: Arc<BindingLayout>) -> MaterialInstance {
        MaterialInstance::new(
            device
                .create_material(
                    &MaterialDescriptor::new()
                        .with_shader(ShaderSource::compute(
                            "@compute @workgroup_size(1) fn main() {}",
                            "main",
                        ))
                        .with_binding_layout(layout),
                )
                .unwrap(),
        )
    }
    #[test]
    fn dispatch_requires_compute_pipeline_and_respects_each_axis() {
        let device = device();
        let material = device
            .create_material(
                &MaterialDescriptor::new().with_shader(ShaderSource::compute(
                    "@compute @workgroup_size(1) fn main() {}",
                    "main",
                )),
            )
            .unwrap();
        for axis in 0..3 {
            let mut counts = [1; 3];
            counts[axis] = device.capabilities().max_compute_workgroups[axis] + 1;
            let mut pass = ComputePass::new("oversized dispatch".into());
            pass.add_dispatch(
                Arc::new(MaterialInstance::new(material.clone())),
                counts[0],
                counts[1],
                counts[2],
            );
            let mut graph = RenderGraph::new();
            graph.add_compute_pass(pass);
            let error = graph.validate_commands(&device).unwrap_err().to_string();
            assert!(error.contains("oversized dispatch") && error.contains("dispatch 0"));
        }
        let graphics = device.create_material(&MaterialDescriptor::new()).unwrap();
        assert!(bindings(&device, &MaterialInstance::new(graphics), &[], true, false).is_err());
        assert!(
            bindings(
                &self::device(),
                &MaterialInstance::new(material),
                &[],
                true,
                false
            )
            .is_err()
        );
    }
    #[test]
    fn binding_layout_compatibility_and_dynamic_offset_ranges() {
        let device = device();
        let layout = Arc::new(BindingLayout::new().with_entry(BindingLayoutEntry::new(
            0,
            BindingType::DynamicUniformBuffer,
        )));
        let b = device
            .create_buffer(&BufferDescriptor::new(512, BufferUsage::UNIFORM))
            .unwrap();
        let group = device
            .create_binding_group(
                layout.clone(),
                BindingGroupDescriptor::new().with_buffer_range(0, b, 0, 16),
            )
            .unwrap();
        let material = instance(&device, layout).with_binding_group(group);
        assert!(bindings(&device, &material, &[vec![256]], true, false).is_ok());
        for offsets in [
            vec![],
            vec![vec![1]],
            vec![vec![512]],
            vec![vec![0, 256]],
            vec![vec![0], vec![0]],
        ] {
            assert!(bindings(&device, &material, &offsets, true, false).is_err());
        }
        let missing = instance(&device, Arc::new(BindingLayout::new()));
        assert!(bindings(&device, &missing, &[], true, false).is_err());
        let wrong = instance(&device, Arc::new(BindingLayout::new()))
            .with_binding_group(material.binding_groups()[0].clone());
        assert!(bindings(&device, &wrong, &[], true, false).is_err());
    }
    #[test]
    fn attachment_ownership_and_pipeline_formats() {
        let device = device();
        let texture = self::device()
            .create_texture(&TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();
        let config = RenderTargetConfig::new().with_color(ColorAttachment::from_texture(texture));
        assert!(targets(&device, &config).is_err());
        let material = device
            .create_material(
                &MaterialDescriptor::new().with_color_format(TextureFormat::Rgba16Float),
            )
            .unwrap();
        assert!(pipeline_targets(&MaterialInstance::new(material), &config).is_err());
    }
}

#[cfg(test)]
mod draw_regression {
    use super::*;
    #[test]
    fn incompatible_draw_returns_submit_error_without_debug_panic() {
        let device = GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap();
        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(Arc::new(
                    VertexLayout::new().with_buffer(VertexBufferLayout::new(4)),
                ))
                .with_vertex_count(3),
            )
            .unwrap();
        let expected = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(12))
                .with_attribute(VertexAttribute::position(0)),
        );
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_vertex_layout(expected)
                    .with_color_format(TextureFormat::Rgba8Unorm),
            )
            .unwrap();
        let target = device
            .create_texture(&TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();
        let mut pass = GraphicsPass::new("bad mesh layout".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(ColorAttachment::from_texture(target)),
        );
        pass.add_draw(mesh, Arc::new(MaterialInstance::new(material)));
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(pass);
        let mut pipeline = device.create_pipeline(1);
        let mut schedule = pipeline.begin_frame().unwrap();
        let error = schedule.submit(graph).unwrap_err().to_string();
        assert!(error.contains("bad mesh layout") && error.contains("draw 0"));
        schedule.submit(RenderGraph::new()).unwrap();
        pipeline.end_frame(schedule);
    }
}
