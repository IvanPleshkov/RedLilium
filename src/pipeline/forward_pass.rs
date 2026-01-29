//! Forward+ main rendering pass

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

/// Forward+ main rendering pass
pub struct ForwardPlusPass {
    tile_size: u32,
    hdr_color: Option<ResourceId>,
}

impl ForwardPlusPass {
    pub fn new(tile_size: u32) -> Self {
        Self {
            tile_size,
            hdr_color: None,
        }
    }

    pub fn hdr_color(&self) -> Option<ResourceId> {
        self.hdr_color
    }
}

impl RenderPass for ForwardPlusPass {
    fn name(&self) -> &str {
        "Forward+ Pass"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        let hdr_color = ctx.create_texture_relative(
            "hdr_color",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Rgba16Float,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.hdr_color = Some(hdr_color);
        ctx.write(hdr_color, ResourceUsage::RenderTarget);
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        let width = ctx.width;
        let height = ctx.height;
        let hdr_view = self.hdr_color.and_then(|id| ctx.get_texture(id));

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        let Some(hdr_view) = hdr_view else {
            return;
        };

        backend.begin_render_pass(&RenderPassDescriptor {
            label: Some("Forward+ Pass".into()),
            color_attachments: vec![ColorAttachment {
                view: hdr_view,
                resolve_target: None,
                load_op: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                store_op: StoreOp::Store,
            }],
            depth_stencil_attachment: None,
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

pub const FORWARD_PLUS_SHADER: &str = r#"
struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = object.model * vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;
    output.clip_position = camera.view_proj * world_pos;
    output.world_normal = normalize((object.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.world_normal * 0.5 + 0.5, 1.0);
}
"#;
