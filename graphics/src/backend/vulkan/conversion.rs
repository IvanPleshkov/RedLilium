//! Type conversions between RedLilium types and Vulkan types.

use ash::vk;

use crate::types::{
    AddressMode, BufferUsage, CompareFunction, FilterMode, TextureFormat, TextureUsage,
};

/// Convert BufferUsage flags to Vulkan buffer usage flags.
pub fn convert_buffer_usage(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut result = vk::BufferUsageFlags::empty();

    if usage.contains(BufferUsage::VERTEX) {
        result |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        result |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        result |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::STORAGE) {
        result |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        result |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(BufferUsage::COPY_SRC) {
        result |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::COPY_DST) {
        result |= vk::BufferUsageFlags::TRANSFER_DST;
    }

    // MAP_READ and MAP_WRITE don't have direct Vulkan buffer usage equivalents
    // They affect memory allocation location instead

    result
}

/// Convert TextureFormat to Vulkan format.
pub fn convert_texture_format(format: TextureFormat) -> vk::Format {
    match format {
        // 8-bit formats
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::R8Snorm => vk::Format::R8_SNORM,
        TextureFormat::R8Uint => vk::Format::R8_UINT,
        TextureFormat::R8Sint => vk::Format::R8_SINT,

        // 16-bit formats
        TextureFormat::R16Unorm => vk::Format::R16_UNORM,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,

        // 32-bit formats
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,

        // 64-bit formats
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,

        // 128-bit formats
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,

        // Depth/stencil formats
        TextureFormat::Depth16Unorm => vk::Format::D16_UNORM,
        TextureFormat::Depth24Plus => vk::Format::D32_SFLOAT, // Vulkan doesn't have D24, use D32
        TextureFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

/// Convert TextureUsage flags to Vulkan image usage flags.
///
/// The format is needed to determine whether RENDER_ATTACHMENT should map to
/// COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT.
pub fn convert_texture_usage(usage: TextureUsage, format: TextureFormat) -> vk::ImageUsageFlags {
    let mut result = vk::ImageUsageFlags::empty();

    if usage.contains(TextureUsage::COPY_SRC) {
        result |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        result |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(TextureUsage::TEXTURE_BINDING) {
        result |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) {
        result |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        // Use the appropriate attachment type based on format
        if format.is_depth_stencil() {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
    }

    result
}

/// Convert FilterMode to Vulkan filter.
pub fn convert_filter_mode(mode: FilterMode) -> vk::Filter {
    match mode {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

/// Convert FilterMode to Vulkan mipmap filter mode.
pub fn convert_mipmap_filter_mode(mode: FilterMode) -> vk::SamplerMipmapMode {
    match mode {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// Convert AddressMode to Vulkan sampler address mode.
pub fn convert_address_mode(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

/// Convert CompareFunction to Vulkan compare op.
pub fn convert_compare_function(func: CompareFunction) -> vk::CompareOp {
    match func {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

/// Convert LoadOp to Vulkan attachment load op and clear value for color attachments.
pub fn convert_load_op_color(op: &crate::graph::LoadOp) -> (vk::AttachmentLoadOp, vk::ClearValue) {
    match op {
        crate::graph::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        crate::graph::LoadOp::DontCare => {
            (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default())
        }
        crate::graph::LoadOp::Clear(clear_value) => {
            if let crate::types::ClearValue::Color { r, g, b, a } = clear_value {
                (
                    vk::AttachmentLoadOp::CLEAR,
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [*r, *g, *b, *a],
                        },
                    },
                )
            } else {
                (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default())
            }
        }
    }
}

/// Convert LoadOp to Vulkan attachment load op and clear value for depth attachments.
pub fn convert_load_op_depth(op: &crate::graph::LoadOp) -> (vk::AttachmentLoadOp, vk::ClearValue) {
    match op {
        crate::graph::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        crate::graph::LoadOp::DontCare => {
            (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default())
        }
        crate::graph::LoadOp::Clear(clear_value) => {
            let depth = match clear_value {
                crate::types::ClearValue::Depth(d) => *d,
                crate::types::ClearValue::DepthStencil { depth, .. } => *depth,
                _ => 1.0,
            };
            let stencil = match clear_value {
                crate::types::ClearValue::Stencil(s) => *s,
                crate::types::ClearValue::DepthStencil { stencil, .. } => *stencil,
                _ => 0,
            };
            (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                },
            )
        }
    }
}

/// Convert StoreOp to Vulkan attachment store op.
pub fn convert_store_op(op: &crate::graph::StoreOp) -> vk::AttachmentStoreOp {
    match op {
        crate::graph::StoreOp::Store => vk::AttachmentStoreOp::STORE,
        crate::graph::StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

/// Convert PresentMode to Vulkan present mode.
pub fn convert_present_mode(mode: crate::swapchain::PresentMode) -> vk::PresentModeKHR {
    match mode {
        crate::swapchain::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        crate::swapchain::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        crate::swapchain::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        crate::swapchain::PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
    }
}
