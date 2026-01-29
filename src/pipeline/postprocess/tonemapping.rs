//! Tonemapping post-processing

use crate::backend::traits::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

/// Tonemapping operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    Reinhard,
    Aces,
    Uncharted2,
    None,
}

impl Default for TonemapOperator {
    fn default() -> Self {
        TonemapOperator::Aces
    }
}

/// Tonemapping post-processing pass
pub struct TonemappingPass {
    pub operator: TonemapOperator,
    pub exposure: f32,
    pub gamma: f32,
    output: ResourceId,
}

impl TonemappingPass {
    pub fn new(output: ResourceId) -> Self {
        Self {
            operator: TonemapOperator::Aces,
            exposure: 1.0,
            gamma: 2.2,
            output,
        }
    }
}

impl RenderPass for TonemappingPass {
    fn name(&self) -> &str {
        "Tonemapping"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        ctx.write(self.output, ResourceUsage::RenderTarget);
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        let width = ctx.width;
        let height = ctx.height;
        let output_view = ctx.get_texture(self.output);

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        let Some(output_view) = output_view else {
            return;
        };

        backend.begin_render_pass(&RenderPassDescriptor {
            label: Some("Tonemapping".into()),
            color_attachments: vec![ColorAttachment {
                view: output_view,
                resolve_target: None,
                load_op: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                store_op: StoreOp::Store,
            }],
            depth_stencil_attachment: None,
        });

        backend.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        backend.draw(0..3, 0..1);

        backend.end_render_pass();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub const TONEMAPPING_SHADER: &str = r#"
struct TonemapParams {
    exposure: f32,
    gamma: f32,
    operator: u32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> params: TonemapParams;

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    color = color * params.exposure;

    var mapped: vec3<f32>;
    switch params.operator {
        case 0u: { mapped = reinhard_tonemap(color); }
        case 1u: { mapped = aces_tonemap(color); }
        default: { mapped = saturate(color); }
    }

    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / params.gamma));
    return vec4<f32>(gamma_corrected, 1.0);
}
"#;
