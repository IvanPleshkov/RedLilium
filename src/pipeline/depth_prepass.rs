//! Depth pre-pass for Forward+ rendering

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

/// Depth pre-pass
pub struct DepthPrepass {
    depth_texture: Option<ResourceId>,
}

impl DepthPrepass {
    pub fn new() -> Self {
        Self { depth_texture: None }
    }

    pub fn depth_texture(&self) -> Option<ResourceId> {
        self.depth_texture
    }
}

impl Default for DepthPrepass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for DepthPrepass {
    fn name(&self) -> &str {
        "Depth Prepass"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        let depth = ctx.create_texture_relative(
            "depth_buffer",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Depth32Float,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.depth_texture = Some(depth);
        ctx.write(depth, ResourceUsage::DepthStencilWrite);
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        // Copy values we need before borrowing backend
        let width = ctx.width;
        let height = ctx.height;
        let depth_view = self.depth_texture.and_then(|id| ctx.get_texture(id));

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        let Some(depth_view) = depth_view else {
            return;
        };

        backend.begin_render_pass(&RenderPassDescriptor {
            label: Some("Depth Prepass".into()),
            color_attachments: vec![],
            depth_stencil_attachment: Some(DepthStencilAttachment {
                view: depth_view,
                depth_load_op: LoadOp::Clear([1.0, 0.0, 0.0, 0.0]),
                depth_store_op: StoreOp::Store,
                depth_clear_value: 1.0,
            }),
        });

        backend.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        backend.end_render_pass();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
