//! Deferred lighting pass
//!
//! Performs lighting calculations using G-buffer data.
//! Renders a fullscreen quad and accumulates lighting from all lights.

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::any::Any;

/// Deferred lighting pass
pub struct LightingPass {
    /// HDR output texture
    hdr_output: Option<ResourceId>,
    /// Light data storage buffer
    light_buffer: Option<ResourceId>,
    /// Maximum number of lights
    max_lights: u32,
    /// G-buffer resource IDs (set externally before graph compilation)
    pub gbuffer_albedo: Option<ResourceId>,
    pub gbuffer_normal: Option<ResourceId>,
    pub gbuffer_material: Option<ResourceId>,
    pub gbuffer_depth: Option<ResourceId>,
}

impl LightingPass {
    pub fn new(max_lights: u32) -> Self {
        Self {
            hdr_output: None,
            light_buffer: None,
            max_lights,
            gbuffer_albedo: None,
            gbuffer_normal: None,
            gbuffer_material: None,
            gbuffer_depth: None,
        }
    }

    pub fn hdr_output(&self) -> Option<ResourceId> {
        self.hdr_output
    }

    pub fn light_buffer(&self) -> Option<ResourceId> {
        self.light_buffer
    }

    /// Set G-buffer resource IDs (call before adding to graph)
    pub fn set_gbuffer_resources(
        &mut self,
        albedo: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        depth: ResourceId,
    ) {
        self.gbuffer_albedo = Some(albedo);
        self.gbuffer_normal = Some(normal);
        self.gbuffer_material = Some(material);
        self.gbuffer_depth = Some(depth);
    }
}

impl Default for LightingPass {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl RenderPass for LightingPass {
    fn name(&self) -> &str {
        "Deferred Lighting Pass"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        // Read G-buffer textures
        if let Some(albedo) = self.gbuffer_albedo {
            ctx.read(albedo, ResourceUsage::TextureRead);
        }
        if let Some(normal) = self.gbuffer_normal {
            ctx.read(normal, ResourceUsage::TextureRead);
        }
        if let Some(material) = self.gbuffer_material {
            ctx.read(material, ResourceUsage::TextureRead);
        }
        if let Some(depth) = self.gbuffer_depth {
            ctx.read(depth, ResourceUsage::TextureRead);
        }

        // Create light buffer (storage buffer for all lights)
        // GpuLightData is 64 bytes (4 Vec4s)
        let light_buffer_size = (self.max_lights as u64) * 64;
        let light_buffer = ctx.create_buffer(
            "light_buffer",
            BufferDescriptor {
                label: Some("Light Buffer".into()),
                size: light_buffer_size,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            },
        );
        self.light_buffer = Some(light_buffer);
        ctx.read(light_buffer, ResourceUsage::StorageBufferRead);

        // Create HDR output texture
        let hdr_output = ctx.create_texture_relative(
            "hdr_color",
            TextureSize::Relative {
                width_scale: 1.0,
                height_scale: 1.0,
            },
            TextureFormat::Rgba16Float,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        );
        self.hdr_output = Some(hdr_output);
        ctx.write(hdr_output, ResourceUsage::RenderTarget);
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        let width = ctx.width;
        let height = ctx.height;

        let hdr_view = self.hdr_output.and_then(|id| ctx.get_texture(id));

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        let Some(hdr_view) = hdr_view else {
            return;
        };

        // Begin lighting render pass
        backend.begin_render_pass(&RenderPassDescriptor {
            label: Some("Deferred Lighting Pass".into()),
            color_attachments: vec![ColorAttachment {
                view: hdr_view,
                resolve_target: None,
                load_op: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                store_op: StoreOp::Store,
            }],
            depth_stencil_attachment: None,
        });

        backend.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);

        // Note: Actual fullscreen quad rendering is done by the engine
        // This pass sets up the render target

        backend.end_render_pass();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Deferred lighting shader
pub const DEFERRED_LIGHTING_SHADER: &str = r#"
// Deferred lighting shader
// Performs PBR lighting using G-buffer data

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct LightData {
    // xyz = position, w = radius
    position_radius: vec4<f32>,
    // xyz = color, w = intensity
    color_intensity: vec4<f32>,
    // xyz = direction, w = light type (0=point, 1=spot, 2=directional)
    direction_type: vec4<f32>,
    // x = cos(inner_angle), y = cos(outer_angle), zw = unused
    spot_params: vec4<f32>,
}

struct LightingUniforms {
    light_count: u32,
    ambient: vec3<f32>,
}

// Fullscreen vertex output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// G-buffer textures
@group(0) @binding(0) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_material: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(4) var gbuffer_sampler: sampler;

// Lights
@group(1) @binding(0) var<storage, read> lights: array<LightData>;
@group(1) @binding(1) var<uniform> lighting: LightingUniforms;

// Camera
@group(2) @binding(0) var<uniform> camera: CameraUniforms;

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);

    return output;
}

// Reconstruct world position from depth
fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);

    // Transform to world space
    let world_pos = camera.inv_view * camera.inv_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Decode normal from G-buffer
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

// PBR lighting calculation
fn calculate_light(
    light: LightData,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let light_type = u32(light.direction_type.w);
    var light_dir: vec3<f32>;
    var attenuation: f32 = 1.0;

    if light_type == 0u {
        // Point light
        let light_vec = light.position_radius.xyz - world_pos;
        let distance = length(light_vec);
        light_dir = normalize(light_vec);

        let radius = light.position_radius.w;
        attenuation = max(0.0, 1.0 - (distance / radius));
        attenuation = attenuation * attenuation;
    } else if light_type == 1u {
        // Spot light
        let light_vec = light.position_radius.xyz - world_pos;
        let distance = length(light_vec);
        light_dir = normalize(light_vec);

        let radius = light.position_radius.w;
        attenuation = max(0.0, 1.0 - (distance / radius));
        attenuation = attenuation * attenuation;

        // Spot cone attenuation
        let spot_dir = normalize(light.direction_type.xyz);
        let cos_angle = dot(-light_dir, spot_dir);
        let inner_cos = light.spot_params.x;
        let outer_cos = light.spot_params.y;
        let spot_atten = saturate((cos_angle - outer_cos) / (inner_cos - outer_cos));
        attenuation = attenuation * spot_atten;
    } else {
        // Directional light
        light_dir = -normalize(light.direction_type.xyz);
    }

    let light_color = light.color_intensity.xyz;
    let intensity = light.color_intensity.w;

    // Simple Blinn-Phong shading (can be replaced with full PBR)
    let ndotl = max(dot(normal, light_dir), 0.0);

    // Diffuse
    let diffuse = albedo * (1.0 - metallic);

    // Specular
    let half_vec = normalize(light_dir + view_dir);
    let ndoth = max(dot(normal, half_vec), 0.0);
    let shininess = mix(16.0, 128.0, 1.0 - roughness);
    let spec_strength = pow(ndoth, shininess) * (1.0 - roughness);
    let spec_color = mix(vec3<f32>(0.04), albedo, metallic);
    let specular = spec_color * spec_strength;

    return (diffuse * ndotl + specular) * light_color * intensity * attenuation;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let pixel_coord = vec2<i32>(input.position.xy);

    // Sample G-buffer
    let albedo_sample = textureLoad(gbuffer_albedo, pixel_coord, 0);
    let normal_sample = textureLoad(gbuffer_normal, pixel_coord, 0);
    let material_sample = textureLoad(gbuffer_material, pixel_coord, 0);
    let depth = textureLoad(gbuffer_depth, pixel_coord, 0);

    // Early out for sky/background (depth == 1.0)
    if depth >= 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let albedo = albedo_sample.rgb;
    let normal = decode_normal(normal_sample.rgb);
    let metallic = material_sample.r;
    let roughness = material_sample.g;

    // Reconstruct world position
    let world_pos = reconstruct_world_position(uv, depth);

    // View direction
    let view_dir = normalize(camera.position.xyz - world_pos);

    // Accumulate lighting
    var color = lighting.ambient * albedo;

    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        color = color + calculate_light(
            lights[i],
            world_pos,
            normal,
            albedo,
            metallic,
            roughness,
            view_dir,
        );
    }

    return vec4<f32>(color, 1.0);
}
"#;

/// Fullscreen vertex shader (shared)
pub const FULLSCREEN_VERTEX_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate fullscreen triangle (3 vertices)
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);

    return output;
}
"#;
