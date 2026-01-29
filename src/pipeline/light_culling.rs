//! Light culling compute pass for Forward+

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use crate::scene::{GpuLightData, TileLightData};
use std::any::Any;

/// Light culling compute pass
pub struct LightCullingPass {
    tile_size: u32,
    max_lights: u32,
    light_buffer: Option<ResourceId>,
    tile_buffer: Option<ResourceId>,
}

impl LightCullingPass {
    pub fn new(tile_size: u32, max_lights: u32) -> Self {
        Self {
            tile_size,
            max_lights,
            light_buffer: None,
            tile_buffer: None,
        }
    }

    pub fn tile_count(screen_width: u32, screen_height: u32, tile_size: u32) -> (u32, u32) {
        let tiles_x = (screen_width + tile_size - 1) / tile_size;
        let tiles_y = (screen_height + tile_size - 1) / tile_size;
        (tiles_x, tiles_y)
    }
}

impl RenderPass for LightCullingPass {
    fn name(&self) -> &str {
        "Light Culling"
    }

    fn setup(&mut self, ctx: &mut PassSetupContext) {
        let (width, height) = ctx.screen_size();
        let (tiles_x, tiles_y) = Self::tile_count(width, height, self.tile_size);
        let total_tiles = tiles_x * tiles_y;

        let light_buffer_size =
            (self.max_lights as usize * std::mem::size_of::<GpuLightData>()) as u64;
        let light_buffer = ctx.create_buffer(
            "light_buffer",
            BufferDescriptor {
                label: Some("Light Buffer".into()),
                size: light_buffer_size.max(64),
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            },
        );
        self.light_buffer = Some(light_buffer);

        let tile_data_size = std::mem::size_of::<TileLightData>() as u64;
        let tile_buffer_size = total_tiles as u64 * tile_data_size;
        let tile_buffer = ctx.create_buffer(
            "tile_light_buffer",
            BufferDescriptor {
                label: Some("Tile Light Buffer".into()),
                size: tile_buffer_size.max(64),
                usage: BufferUsage::STORAGE,
                mapped_at_creation: false,
            },
        );
        self.tile_buffer = Some(tile_buffer);

        ctx.write(light_buffer, ResourceUsage::StorageBufferWrite);
        ctx.write(tile_buffer, ResourceUsage::StorageBufferWrite);
    }

    fn execute(&self, ctx: &mut PassExecuteContext) {
        let width = ctx.width;
        let height = ctx.height;
        let (tiles_x, tiles_y) = Self::tile_count(width, height, self.tile_size);

        let Some(backend) = ctx.backend::<WgpuBackend>() else {
            return;
        };

        backend.begin_compute_pass(Some("Light Culling"));

        let workgroups_x = (tiles_x + 7) / 8;
        let workgroups_y = (tiles_y + 7) / 8;
        backend.dispatch_compute(workgroups_x, workgroups_y, 1);

        backend.end_compute_pass();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub const LIGHT_CULLING_SHADER: &str = r#"
struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct Light {
    position_radius: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_type: vec4<f32>,
    spot_params: vec4<f32>,
}

struct TileData {
    light_count: u32,
    _padding: vec3<u32>,
    light_indices: array<u32, 256>,
}

struct CullingUniforms {
    tile_count: vec2<u32>,
    tile_size: u32,
    light_count: u32,
    screen_size: vec2<u32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> culling: CullingUniforms;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(3) var<storage, read_write> tiles: array<TileData>;
@group(0) @binding(4) var depth_texture: texture_depth_2d;

var<workgroup> visible_light_count: atomic<u32>;
var<workgroup> visible_light_indices: array<u32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tile_id = workgroup_id.x + workgroup_id.y * culling.tile_count.x;
    let local_idx = local_id.x + local_id.y * 16u;

    if local_idx == 0u {
        atomicStore(&visible_light_count, 0u);
    }
    workgroupBarrier();

    let lights_per_thread = (culling.light_count + 255u) / 256u;
    let start_light = local_idx * lights_per_thread;
    let end_light = min(start_light + lights_per_thread, culling.light_count);

    for (var i = start_light; i < end_light; i = i + 1u) {
        let light = lights[i];
        let pos = light.position_radius.xyz;
        let radius = light.position_radius.w;

        let view_pos = camera.view * vec4<f32>(pos, 1.0);
        if -view_pos.z - radius < camera.near_far.y && -view_pos.z + radius > camera.near_far.x {
            let idx = atomicAdd(&visible_light_count, 1u);
            if idx < 256u {
                visible_light_indices[idx] = i;
            }
        }
    }
    workgroupBarrier();

    if local_idx == 0u {
        let count = min(atomicLoad(&visible_light_count), 256u);
        tiles[tile_id].light_count = count;
        for (var i = 0u; i < count; i = i + 1u) {
            tiles[tile_id].light_indices[i] = visible_light_indices[i];
        }
    }
}
"#;
