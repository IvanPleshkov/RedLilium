//! Render pass definitions for the render graph

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::render_graph::resource::*;
use crate::scene::Scene;
use std::any::Any;

/// Unique identifier for a render pass
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(pub(crate) u32);

/// Context for setting up pass resources
pub struct PassSetupContext<'a> {
    pub(crate) resources: &'a mut Vec<VirtualResource>,
    pub(crate) inputs: &'a mut Vec<ResourceAccess>,
    pub(crate) outputs: &'a mut Vec<ResourceAccess>,
    pub(crate) next_resource_id: &'a mut u32,
    pub(crate) screen_width: u32,
    pub(crate) screen_height: u32,
}

impl<'a> PassSetupContext<'a> {
    /// Create a new texture resource
    pub fn create_texture(&mut self, name: &str, desc: TextureDescriptor) -> ResourceId {
        let id = ResourceId(*self.next_resource_id);
        *self.next_resource_id += 1;

        self.resources.push(VirtualResource::Texture(VirtualTexture {
            id,
            desc,
            name: name.to_string(),
        }));

        id
    }

    /// Create a texture with size relative to screen
    pub fn create_texture_relative(
        &mut self,
        name: &str,
        size: TextureSize,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> ResourceId {
        let (width, height) = size.resolve(self.screen_width, self.screen_height);

        self.create_texture(
            name,
            TextureDescriptor {
                label: Some(name.to_string()),
                width,
                height,
                depth: 1,
                mip_levels: 1,
                format,
                usage,
            },
        )
    }

    /// Create a new buffer resource
    pub fn create_buffer(&mut self, name: &str, desc: BufferDescriptor) -> ResourceId {
        let id = ResourceId(*self.next_resource_id);
        *self.next_resource_id += 1;

        self.resources.push(VirtualResource::Buffer(VirtualBuffer {
            id,
            desc,
            name: name.to_string(),
        }));

        id
    }

    /// Declare that this pass reads from a resource
    pub fn read(&mut self, resource: ResourceId, usage: ResourceUsage) {
        self.inputs.push(ResourceAccess { resource, usage });
    }

    /// Declare that this pass writes to a resource
    pub fn write(&mut self, resource: ResourceId, usage: ResourceUsage) {
        self.outputs.push(ResourceAccess { resource, usage });
    }

    /// Get screen dimensions
    pub fn screen_size(&self) -> (u32, u32) {
        (self.screen_width, self.screen_height)
    }
}

/// Context for executing a render pass
pub struct PassExecuteContext<'a> {
    pub backend: &'a mut dyn std::any::Any,
    pub scene: &'a Scene,
    pub width: u32,
    pub height: u32,
    pub resource_textures: &'a std::collections::HashMap<ResourceId, TextureViewHandle>,
    pub resource_buffers: &'a std::collections::HashMap<ResourceId, BufferHandle>,
}

impl<'a> PassExecuteContext<'a> {
    /// Get backend as concrete type
    pub fn backend<B: GraphicsBackend + 'static>(&mut self) -> Option<&mut B> {
        self.backend.downcast_mut::<B>()
    }

    /// Get a texture view handle for a resource
    pub fn get_texture(&self, resource: ResourceId) -> Option<TextureViewHandle> {
        self.resource_textures.get(&resource).copied()
    }

    /// Get a buffer handle for a resource
    pub fn get_buffer(&self, resource: ResourceId) -> Option<BufferHandle> {
        self.resource_buffers.get(&resource).copied()
    }
}

/// Trait for render passes
pub trait RenderPass: Send + Sync {
    /// Get the pass name for debugging
    fn name(&self) -> &str;

    /// Setup phase - declare resources and dependencies
    fn setup(&mut self, ctx: &mut PassSetupContext);

    /// Execute phase - record commands
    fn execute(&self, ctx: &mut PassExecuteContext);

    /// Allow downcasting
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Type of render pass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassType {
    /// Graphics render pass
    Graphics,
    /// Compute pass
    Compute,
    /// Transfer/copy pass
    Transfer,
}

/// Metadata about a pass in the graph
#[derive(Debug)]
pub struct PassNode {
    pub id: PassId,
    pub name: String,
    pub pass_type: PassType,
    pub inputs: Vec<ResourceAccess>,
    pub outputs: Vec<ResourceAccess>,
}

impl PassNode {
    pub fn reads_resource(&self, resource: ResourceId) -> bool {
        self.inputs.iter().any(|a| a.resource == resource)
    }

    pub fn writes_resource(&self, resource: ResourceId) -> bool {
        self.outputs.iter().any(|a| a.resource == resource)
    }
}
