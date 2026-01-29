//! G-Buffer generation pass for deferred rendering
//!
//! Renders geometry to multiple render targets (MRT):
//! - Albedo (base color)
//! - World-space normals (encoded)
//! - Material properties (metallic, roughness)
//! - Depth buffer

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

/// G-Buffer generation pass for deferred rendering
pub struct GBufferPass {
    /// Albedo/base color texture (RGBA8)
    albedo_texture: Option<ResourceId>,
    /// World-space normal texture (RGB10A2 or RGBA16F)
    normal_texture: Option<ResourceId>,
    /// Material properties: R=metallic, G=roughness (RG8 or RGBA8)
    material_texture: Option<ResourceId>,
    /// Depth buffer
    depth_texture: Option<ResourceId>,
}

impl GBufferPass {
    pub fn new() -> Self {
        Self {
            albedo_texture: None,
            normal_texture: None,
            material_texture: None,
            depth_texture: None,
        }
    }

    pub fn albedo_texture(&self) -> Option<ResourceId> {
        self.albedo_texture
    }

    pub fn normal_texture(&self) -> Option<ResourceId> {
        self.normal_texture
    }

    pub fn material_texture(&self) -> Option<ResourceId> {
        self.material_texture
    }

    pub fn depth_texture(&self) -> Option<ResourceId> {
        self.depth_texture
    }
}

impl Default for GBufferPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "G-Buffer Pass"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        // Create G-buffer textures
        let albedo = ctx.create_texture_relative(
            "gbuffer_albedo",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.albedo_texture = Some(albedo);
        ctx.write(albedo, ResourceUsage::RenderTarget);

        let normal = ctx.create_texture_relative(
            "gbuffer_normal",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Rgba16Float, // Use float for better normal precision
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.normal_texture = Some(normal);
        ctx.write(normal, ResourceUsage::RenderTarget);

        let material = ctx.create_texture_relative(
            "gbuffer_material",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.material_texture = Some(material);
        ctx.write(material, ResourceUsage::RenderTarget);

        let depth = ctx.create_texture_relative(
            "gbuffer_depth",
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
        let width = ctx.width;
        let height = ctx.height;

        // Get texture views
        let albedo_view = self.albedo_texture.and_then(|id| ctx.get_texture(id));
        let normal_view = self.normal_texture.and_then(|id| ctx.get_texture(id));
        let material_view = self.material_texture.and_then(|id| ctx.get_texture(id));
        let depth_view = self.depth_texture.and_then(|id| ctx.get_texture(id));

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        let (Some(albedo_view), Some(normal_view), Some(material_view), Some(depth_view)) =
            (albedo_view, normal_view, material_view, depth_view)
        else {
            return;
        };

        // Begin G-buffer render pass with multiple render targets
        backend.begin_render_pass(&RenderPassDescriptor {
            label: Some("G-Buffer Pass".into()),
            color_attachments: vec![
                // Location 0: Albedo
                ColorAttachment {
                    view: albedo_view,
                    resolve_target: None,
                    load_op: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                    store_op: StoreOp::Store,
                },
                // Location 1: Normal
                ColorAttachment {
                    view: normal_view,
                    resolve_target: None,
                    load_op: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                    store_op: StoreOp::Store,
                },
                // Location 2: Material (metallic, roughness)
                ColorAttachment {
                    view: material_view,
                    resolve_target: None,
                    load_op: LoadOp::Clear([0.0, 0.5, 0.0, 0.0]), // Default roughness 0.5
                    store_op: StoreOp::Store,
                },
            ],
            depth_stencil_attachment: Some(DepthStencilAttachment {
                view: depth_view,
                depth_load_op: LoadOp::Clear([1.0, 0.0, 0.0, 0.0]),
                depth_store_op: StoreOp::Store,
                depth_clear_value: 1.0,
            }),
        });

        backend.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);

        // Note: Actual geometry rendering is done by the engine's render loop
        // This pass just sets up the render targets

        backend.end_render_pass();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// G-Buffer generation shader
pub const GBUFFER_SHADER: &str = r#"
// G-Buffer generation shader for deferred rendering

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

struct MaterialUniforms {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _padding: vec2<f32>,
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

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;

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
fn fs_main(input: VertexOutput) -> GBufferOutput {
    var output: GBufferOutput;

    // Albedo: base color
    output.albedo = material.base_color;

    // Normal: encode world-space normal to [0,1] range
    // Using simple encoding: normal * 0.5 + 0.5
    output.normal = vec4<f32>(input.world_normal * 0.5 + 0.5, 1.0);

    // Material: R = metallic, G = roughness, BA = unused
    output.material = vec4<f32>(material.metallic, material.roughness, 0.0, 1.0);

    return output;
}
"#;
