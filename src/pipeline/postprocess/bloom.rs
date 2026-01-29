//! Bloom post-processing effect

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

const BLOOM_MIP_LEVELS: u32 = 5;

/// Bloom post-processing pass
pub struct BloomPass {
    pub threshold: f32,
    pub intensity: f32,
    bloom_textures: Vec<ResourceId>,
}

impl BloomPass {
    pub fn new() -> Self {
        Self {
            threshold: 1.0,
            intensity: 0.5,
            bloom_textures: Vec::new(),
        }
    }
}

impl Default for BloomPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for BloomPass {
    fn name(&self) -> &str {
        "Bloom"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        self.bloom_textures.clear();

        for i in 0..BLOOM_MIP_LEVELS {
            let scale = 1.0 / (1 << i) as f32;
            let tex = ctx.create_texture_relative(
                &format!("bloom_mip_{}", i),
                TextureSize::Relative {
                    width_scale: scale,
                    height_scale: scale,
                },
                TextureFormat::Rgba16Float,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            );
            self.bloom_textures.push(tex);
            ctx.write(tex, ResourceUsage::RenderTarget);
        }
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        // Collect texture views before borrowing backend
        let texture_views: Vec<_> = self
            .bloom_textures
            .iter()
            .filter_map(|&tex_id| ctx.get_texture(tex_id))
            .collect();

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        for view in texture_views {
            backend.begin_render_pass(&RenderPassDescriptor {
                label: Some("Bloom Clear".into()),
                color_attachments: vec![ColorAttachment {
                    view,
                    resolve_target: None,
                    load_op: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                    store_op: StoreOp::Store,
                }],
                depth_stencil_attachment: None,
            });
            backend.end_render_pass();
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub const BLOOM_EXTRACT_SHADER: &str = r#"
struct BloomParams {
    threshold: f32,
    soft_threshold: f32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BloomParams;

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
    let color = textureSample(hdr_texture, hdr_sampler, input.uv);
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let weight = max(luminance - params.threshold, 0.0) / max(luminance, 0.0001);
    return vec4<f32>(color.rgb * weight, 1.0);
}
"#;
