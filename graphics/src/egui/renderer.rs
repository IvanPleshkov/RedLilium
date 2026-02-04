//! Egui renderer for RedLilium graphics.
//!
//! This module handles GPU resources and rendering for egui.

use std::collections::HashMap;
use std::sync::Arc;

use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};

use crate::graph::{ColorAttachment, GraphicsPass, LoadOp, RenderTargetConfig};
use crate::materials::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Material, MaterialDescriptor,
    MaterialInstance, ShaderSource, ShaderStage, ShaderStageFlags,
};
use crate::mesh::{
    IndexFormat, MeshDescriptor, VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic,
    VertexBufferLayout, VertexLayout,
};
use crate::resources::{Buffer, Sampler, Texture};
use crate::shader::ShaderComposer;
use crate::types::{
    AddressMode, BufferDescriptor, BufferUsage, FilterMode, SamplerDescriptor, TextureDescriptor,
    TextureFormat, TextureUsage,
};
use crate::{GraphicsDevice, SurfaceTexture};

/// WGSL shader source for egui rendering.
const EGUI_SHADER: &str = r#"
#import redlilium::egui::{EguiUniforms, EguiVertexInput, EguiVertexOutput, srgb_to_linear}

@group(0) @binding(0) var<uniform> uniforms: EguiUniforms;
@group(1) @binding(0) var egui_texture: texture_2d<f32>;
@group(1) @binding(1) var egui_sampler: sampler;

@vertex
fn vs_main(in: EguiVertexInput) -> EguiVertexOutput {
    var out: EguiVertexOutput;

    // Transform from screen space [0, screen_size] to clip space [-1, 1]
    let pos = vec2<f32>(
        2.0 * in.position.x / uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * in.position.y / uniforms.screen_size.y
    );

    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    out.color = in.color;

    return out;
}

@fragment
fn fs_main(in: EguiVertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(egui_texture, egui_sampler, in.tex_coords);

    // Convert vertex color from sRGB to linear
    let vertex_linear = vec4<f32>(srgb_to_linear(in.color.rgb), in.color.a);

    // Multiply with texture (already in linear space if font atlas)
    var color = vertex_linear * tex_color;

    // Pre-multiply alpha for proper blending
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
"#;

/// Egui vertex data matching egui's Vertex structure.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EguiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl From<&Vertex> for EguiVertex {
    fn from(v: &Vertex) -> Self {
        Self {
            pos: [v.pos.x, v.pos.y],
            uv: [v.uv.x, v.uv.y],
            color: [
                v.color.r() as f32 / 255.0,
                v.color.g() as f32 / 255.0,
                v.color.b() as f32 / 255.0,
                v.color.a() as f32 / 255.0,
            ],
        }
    }
}

/// Uniform buffer data for egui rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EguiUniforms {
    pub screen_size: [f32; 2],
    pub _padding: [f32; 2],
}

/// Manages GPU resources for egui rendering.
pub struct EguiRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    vertex_layout: Arc<VertexLayout>,
    uniform_buffer: Arc<Buffer>,
    sampler: Arc<Sampler>,
    textures: HashMap<TextureId, Arc<Texture>>,
    #[allow(dead_code)]
    uniform_binding_layout: Arc<BindingLayout>,
    #[allow(dead_code)]
    texture_binding_layout: Arc<BindingLayout>,
}

impl EguiRenderer {
    /// Create a new egui renderer.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        // Create vertex layout for egui vertices
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<EguiVertex>() as u32
                ))
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Position,
                    format: VertexAttributeFormat::Float2,
                    offset: 0,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::TexCoord0,
                    format: VertexAttributeFormat::Float2,
                    offset: 8,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Color,
                    format: VertexAttributeFormat::Float4,
                    offset: 16,
                    buffer_index: 0,
                })
                .with_label("egui_vertex_layout"),
        );

        // Create binding layouts
        let uniform_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("egui_uniform_bindings"),
        );

        let texture_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("egui_texture_bindings"),
        );

        // Compose shader with library imports
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_shader = shader_composer
            .compose(EGUI_SHADER, &[])
            .expect("Failed to compose egui shader");

        // Create material
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(uniform_binding_layout.clone())
                    .with_binding_layout(texture_binding_layout.clone())
                    .with_vertex_layout(vertex_layout.clone())
                    .with_blend_state(crate::materials::BlendState::premultiplied_alpha())
                    .with_label("egui_material"),
            )
            .expect("Failed to create egui material");

        // Create uniform buffer
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<EguiUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create egui uniform buffer");

        // Create sampler
        let sampler = device
            .create_sampler(&SamplerDescriptor {
                label: Some("egui_sampler".into()),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            })
            .expect("Failed to create egui sampler");

        Self {
            device,
            material,
            vertex_layout,
            uniform_buffer,
            sampler,
            textures: HashMap::new(),
            uniform_binding_layout,
            texture_binding_layout,
        }
    }

    /// Update screen size uniforms.
    pub fn update_screen_size(&self, width: u32, height: u32) {
        let uniforms = EguiUniforms {
            screen_size: [width as f32, height as f32],
            _padding: [0.0, 0.0],
        };
        self.device
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Process texture updates from egui.
    pub fn update_textures(&mut self, textures_delta: &TexturesDelta) {
        // Free textures that are no longer needed
        for id in &textures_delta.free {
            self.textures.remove(id);
        }

        // Set or update textures
        for (id, delta) in &textures_delta.set {
            self.set_texture(*id, delta);
        }
    }

    /// Set or update a texture.
    fn set_texture(&mut self, id: TextureId, delta: &ImageDelta) {
        let (width, height) = (delta.image.width() as u32, delta.image.height() as u32);

        // Convert image data to RGBA8
        let pixels: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                image.pixels.iter().flat_map(|c| c.to_array()).collect()
            }
        };

        if let Some(pos) = delta.pos {
            // Partial update - need to update a region of existing texture
            if let Some(texture) = self.textures.get(&id) {
                // For partial updates, we'd need a staging buffer and transfer pass
                // For simplicity, we recreate the texture (not ideal for performance)
                log::debug!("Partial texture update at {:?} - recreating texture", pos);
                // In a production implementation, you'd use a transfer pass here
                let _ = texture;
            }
        }

        // Create or recreate texture
        let texture = self
            .device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label(format!("egui_texture_{:?}", id)),
            )
            .expect("Failed to create egui texture");

        // Upload pixel data
        self.device.write_texture(&texture, &pixels);

        self.textures.insert(id, texture);
    }

    /// Create a graphics pass for rendering egui primitives.
    pub fn create_graphics_pass(
        &self,
        primitives: &[ClippedPrimitive],
        surface_texture: &SurfaceTexture,
        screen_width: u32,
        screen_height: u32,
    ) -> GraphicsPass {
        let mut pass = GraphicsPass::new("egui".into());

        // Set render target (draw on top of existing content)
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(surface_texture).with_load_op(LoadOp::Load),
            ),
        );

        // Create uniform binding group
        #[allow(clippy::arc_with_non_send_sync)]
        let uniform_binding =
            Arc::new(BindingGroup::new().with_buffer(0, self.uniform_buffer.clone()));

        // Process each primitive
        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }

                    // Get texture for this mesh
                    let texture = match self.textures.get(&mesh.texture_id) {
                        Some(t) => t.clone(),
                        None => {
                            log::warn!("Missing texture {:?}", mesh.texture_id);
                            continue;
                        }
                    };

                    // Convert vertices
                    let vertices: Vec<EguiVertex> =
                        mesh.vertices.iter().map(EguiVertex::from).collect();

                    // Create mesh
                    let gpu_mesh = self
                        .device
                        .create_mesh(
                            &MeshDescriptor::new(self.vertex_layout.clone())
                                .with_vertex_count(vertices.len() as u32)
                                .with_indices(IndexFormat::Uint32, mesh.indices.len() as u32)
                                .with_label("egui_mesh"),
                        )
                        .expect("Failed to create egui mesh");

                    // Upload vertex data
                    if let Some(vb) = gpu_mesh.vertex_buffer(0) {
                        self.device
                            .write_buffer(vb, 0, bytemuck::cast_slice(&vertices));
                    }

                    // Upload index data
                    if let Some(ib) = gpu_mesh.index_buffer() {
                        self.device
                            .write_buffer(ib, 0, bytemuck::cast_slice(&mesh.indices));
                    }

                    // Create texture binding group
                    #[allow(clippy::arc_with_non_send_sync)]
                    let texture_binding = Arc::new(
                        BindingGroup::new()
                            .with_texture(0, texture)
                            .with_sampler(1, self.sampler.clone()),
                    );

                    // Create material instance
                    let material_instance = Arc::new(
                        MaterialInstance::new(self.material.clone())
                            .with_binding_group(uniform_binding.clone())
                            .with_binding_group(texture_binding),
                    );

                    // Calculate scissor rect
                    let clip_min_x = clip_rect.min.x.round() as i32;
                    let clip_min_y = clip_rect.min.y.round() as i32;
                    let clip_max_x = clip_rect.max.x.round() as i32;
                    let clip_max_y = clip_rect.max.y.round() as i32;

                    let scissor_x = clip_min_x.max(0);
                    let scissor_y = clip_min_y.max(0);
                    let scissor_width = (clip_max_x - clip_min_x).max(0) as u32;
                    let scissor_height = (clip_max_y - clip_min_y).max(0) as u32;

                    // Clamp to screen bounds
                    let scissor_width =
                        scissor_width.min(screen_width.saturating_sub(scissor_x as u32));
                    let scissor_height =
                        scissor_height.min(screen_height.saturating_sub(scissor_y as u32));

                    if scissor_width > 0 && scissor_height > 0 {
                        pass.add_draw_with_scissor(
                            gpu_mesh,
                            material_instance,
                            crate::types::ScissorRect {
                                x: scissor_x,
                                y: scissor_y,
                                width: scissor_width,
                                height: scissor_height,
                            },
                        );
                    }
                }
                Primitive::Callback(_) => {
                    // Custom rendering callbacks are not supported yet
                    log::warn!("Egui render callbacks are not supported");
                }
            }
        }

        pass
    }
}
