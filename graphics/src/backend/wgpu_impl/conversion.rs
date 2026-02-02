//! Type conversions between RedLilium types and wgpu types.

use crate::mesh::{PrimitiveTopology, VertexAttributeFormat, VertexStepMode};
use crate::types::{
    AddressMode, BufferUsage, CompareFunction, FilterMode, TextureFormat, TextureUsage,
};

/// Convert BufferUsage flags to wgpu buffer usages.
pub fn convert_buffer_usage(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut result = wgpu::BufferUsages::empty();

    if usage.contains(BufferUsage::VERTEX) {
        result |= wgpu::BufferUsages::VERTEX;
    }
    if usage.contains(BufferUsage::INDEX) {
        result |= wgpu::BufferUsages::INDEX;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        result |= wgpu::BufferUsages::UNIFORM;
    }
    if usage.contains(BufferUsage::STORAGE) {
        result |= wgpu::BufferUsages::STORAGE;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        result |= wgpu::BufferUsages::INDIRECT;
    }
    if usage.contains(BufferUsage::COPY_SRC) {
        result |= wgpu::BufferUsages::COPY_SRC;
    }
    if usage.contains(BufferUsage::COPY_DST) {
        result |= wgpu::BufferUsages::COPY_DST;
    }
    if usage.contains(BufferUsage::MAP_READ) {
        result |= wgpu::BufferUsages::MAP_READ;
    }
    if usage.contains(BufferUsage::MAP_WRITE) {
        result |= wgpu::BufferUsages::MAP_WRITE;
    }

    result
}

/// Convert TextureFormat to wgpu format.
pub fn convert_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        // 8-bit formats
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        TextureFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        TextureFormat::R8Sint => wgpu::TextureFormat::R8Sint,

        // 16-bit formats
        TextureFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
        TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,

        // 32-bit formats
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,

        // 64-bit formats
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,

        // 128-bit formats
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,

        // Depth/stencil formats
        TextureFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
    }
}

/// Convert TextureUsage flags to wgpu texture usages.
pub fn convert_texture_usage(usage: TextureUsage) -> wgpu::TextureUsages {
    let mut result = wgpu::TextureUsages::empty();

    if usage.contains(TextureUsage::COPY_SRC) {
        result |= wgpu::TextureUsages::COPY_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        result |= wgpu::TextureUsages::COPY_DST;
    }
    if usage.contains(TextureUsage::TEXTURE_BINDING) {
        result |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) {
        result |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        result |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }

    result
}

/// Convert AddressMode to wgpu address mode.
pub fn convert_address_mode(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

/// Convert FilterMode to wgpu filter mode.
pub fn convert_filter_mode(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

/// Convert FilterMode to wgpu mipmap filter mode.
pub fn convert_mipmap_filter_mode(mode: FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

/// Convert CompareFunction to wgpu compare function.
pub fn convert_compare_function(func: CompareFunction) -> wgpu::CompareFunction {
    match func {
        CompareFunction::Never => wgpu::CompareFunction::Never,
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

/// Convert LoadOp to wgpu load op for color attachments.
pub fn convert_load_op(op: &crate::graph::LoadOp) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        crate::graph::LoadOp::Load => wgpu::LoadOp::Load,
        crate::graph::LoadOp::DontCare => wgpu::LoadOp::Load, // wgpu doesn't have DontCare for color
        crate::graph::LoadOp::Clear(clear_value) => {
            if let crate::types::ClearValue::Color { r, g, b, a } = clear_value {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: *r as f64,
                    g: *g as f64,
                    b: *b as f64,
                    a: *a as f64,
                })
            } else {
                wgpu::LoadOp::Load
            }
        }
    }
}

/// Convert LoadOp to wgpu load op for depth attachments.
pub fn convert_depth_load_op(op: &crate::graph::LoadOp) -> wgpu::LoadOp<f32> {
    match op {
        crate::graph::LoadOp::Load => wgpu::LoadOp::Load,
        crate::graph::LoadOp::DontCare => wgpu::LoadOp::Load, // wgpu doesn't have DontCare for depth
        crate::graph::LoadOp::Clear(clear_value) => {
            if let crate::types::ClearValue::Depth(depth) = clear_value {
                wgpu::LoadOp::Clear(*depth)
            } else if let crate::types::ClearValue::DepthStencil { depth, .. } = clear_value {
                wgpu::LoadOp::Clear(*depth)
            } else {
                wgpu::LoadOp::Load
            }
        }
    }
}

/// Convert StoreOp to wgpu store op.
pub fn convert_store_op(op: &crate::graph::StoreOp) -> wgpu::StoreOp {
    match op {
        crate::graph::StoreOp::Store => wgpu::StoreOp::Store,
        crate::graph::StoreOp::DontCare => wgpu::StoreOp::Discard,
    }
}

/// Convert VertexAttributeFormat to wgpu vertex format.
pub fn convert_vertex_format(format: VertexAttributeFormat) -> wgpu::VertexFormat {
    match format {
        VertexAttributeFormat::Float => wgpu::VertexFormat::Float32,
        VertexAttributeFormat::Float2 => wgpu::VertexFormat::Float32x2,
        VertexAttributeFormat::Float3 => wgpu::VertexFormat::Float32x3,
        VertexAttributeFormat::Float4 => wgpu::VertexFormat::Float32x4,
        VertexAttributeFormat::Int => wgpu::VertexFormat::Sint32,
        VertexAttributeFormat::Int2 => wgpu::VertexFormat::Sint32x2,
        VertexAttributeFormat::Int3 => wgpu::VertexFormat::Sint32x3,
        VertexAttributeFormat::Int4 => wgpu::VertexFormat::Sint32x4,
        VertexAttributeFormat::Uint => wgpu::VertexFormat::Uint32,
        VertexAttributeFormat::Uint2 => wgpu::VertexFormat::Uint32x2,
        VertexAttributeFormat::Uint3 => wgpu::VertexFormat::Uint32x3,
        VertexAttributeFormat::Uint4 => wgpu::VertexFormat::Uint32x4,
        VertexAttributeFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        VertexAttributeFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
    }
}

/// Convert PrimitiveTopology to wgpu primitive topology.
pub fn convert_topology(topology: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

/// Convert VertexStepMode to wgpu vertex step mode.
pub fn convert_step_mode(mode: VertexStepMode) -> wgpu::VertexStepMode {
    match mode {
        VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

/// Convert PresentMode to wgpu present mode.
pub fn convert_present_mode(mode: crate::swapchain::PresentMode) -> wgpu::PresentMode {
    match mode {
        crate::swapchain::PresentMode::Immediate => wgpu::PresentMode::Immediate,
        crate::swapchain::PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        crate::swapchain::PresentMode::Fifo => wgpu::PresentMode::Fifo,
        crate::swapchain::PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
    }
}

/// Convert ShaderStageFlags to wgpu shader stages.
pub fn convert_shader_stages(flags: crate::materials::ShaderStageFlags) -> wgpu::ShaderStages {
    let mut result = wgpu::ShaderStages::empty();

    if flags.contains(crate::materials::ShaderStageFlags::VERTEX) {
        result |= wgpu::ShaderStages::VERTEX;
    }
    if flags.contains(crate::materials::ShaderStageFlags::FRAGMENT) {
        result |= wgpu::ShaderStages::FRAGMENT;
    }
    if flags.contains(crate::materials::ShaderStageFlags::COMPUTE) {
        result |= wgpu::ShaderStages::COMPUTE;
    }

    result
}

/// Convert BindingType to wgpu binding type.
pub fn convert_binding_type(binding_type: crate::materials::BindingType) -> wgpu::BindingType {
    match binding_type {
        crate::materials::BindingType::UniformBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::materials::BindingType::StorageBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::materials::BindingType::Texture => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        crate::materials::BindingType::TextureCube => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        crate::materials::BindingType::Sampler => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        }
        crate::materials::BindingType::CombinedTextureSampler => {
            // wgpu doesn't have combined texture/sampler, use texture binding
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        }
    }
}
