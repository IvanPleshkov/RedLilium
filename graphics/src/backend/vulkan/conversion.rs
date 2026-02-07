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
        TextureFormat::Rgba10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TextureFormat::Bgra10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,

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

        // BC compressed formats
        TextureFormat::Bc1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc1RgbaUnormSrgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        TextureFormat::Bc2RgbaUnorm => vk::Format::BC2_UNORM_BLOCK,
        TextureFormat::Bc2RgbaUnormSrgb => vk::Format::BC2_SRGB_BLOCK,
        TextureFormat::Bc3RgbaUnorm => vk::Format::BC3_UNORM_BLOCK,
        TextureFormat::Bc3RgbaUnormSrgb => vk::Format::BC3_SRGB_BLOCK,
        TextureFormat::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        TextureFormat::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        TextureFormat::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        TextureFormat::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        TextureFormat::Bc6hRgbFloat => vk::Format::BC6H_SFLOAT_BLOCK,
        TextureFormat::Bc7RgbaUnorm => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Bc7RgbaUnormSrgb => vk::Format::BC7_SRGB_BLOCK,

        // ETC2/EAC compressed formats
        TextureFormat::Etc2Rgb8Unorm => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
        TextureFormat::Etc2Rgb8UnormSrgb => vk::Format::ETC2_R8G8B8_SRGB_BLOCK,
        TextureFormat::Etc2Rgb8A1Unorm => vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK,
        TextureFormat::Etc2Rgb8A1UnormSrgb => vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        TextureFormat::Etc2Rgba8Unorm => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
        TextureFormat::Etc2Rgba8UnormSrgb => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        TextureFormat::EacR11Unorm => vk::Format::EAC_R11_UNORM_BLOCK,
        TextureFormat::EacR11Snorm => vk::Format::EAC_R11_SNORM_BLOCK,
        TextureFormat::EacRg11Unorm => vk::Format::EAC_R11G11_UNORM_BLOCK,
        TextureFormat::EacRg11Snorm => vk::Format::EAC_R11G11_SNORM_BLOCK,

        // ASTC compressed formats
        TextureFormat::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
        TextureFormat::Astc4x4UnormSrgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
        TextureFormat::Astc5x4Unorm => vk::Format::ASTC_5X4_UNORM_BLOCK,
        TextureFormat::Astc5x4UnormSrgb => vk::Format::ASTC_5X4_SRGB_BLOCK,
        TextureFormat::Astc5x5Unorm => vk::Format::ASTC_5X5_UNORM_BLOCK,
        TextureFormat::Astc5x5UnormSrgb => vk::Format::ASTC_5X5_SRGB_BLOCK,
        TextureFormat::Astc6x5Unorm => vk::Format::ASTC_6X5_UNORM_BLOCK,
        TextureFormat::Astc6x5UnormSrgb => vk::Format::ASTC_6X5_SRGB_BLOCK,
        TextureFormat::Astc6x6Unorm => vk::Format::ASTC_6X6_UNORM_BLOCK,
        TextureFormat::Astc6x6UnormSrgb => vk::Format::ASTC_6X6_SRGB_BLOCK,
        TextureFormat::Astc8x5Unorm => vk::Format::ASTC_8X5_UNORM_BLOCK,
        TextureFormat::Astc8x5UnormSrgb => vk::Format::ASTC_8X5_SRGB_BLOCK,
        TextureFormat::Astc8x6Unorm => vk::Format::ASTC_8X6_UNORM_BLOCK,
        TextureFormat::Astc8x6UnormSrgb => vk::Format::ASTC_8X6_SRGB_BLOCK,
        TextureFormat::Astc8x8Unorm => vk::Format::ASTC_8X8_UNORM_BLOCK,
        TextureFormat::Astc8x8UnormSrgb => vk::Format::ASTC_8X8_SRGB_BLOCK,
        TextureFormat::Astc10x5Unorm => vk::Format::ASTC_10X5_UNORM_BLOCK,
        TextureFormat::Astc10x5UnormSrgb => vk::Format::ASTC_10X5_SRGB_BLOCK,
        TextureFormat::Astc10x6Unorm => vk::Format::ASTC_10X6_UNORM_BLOCK,
        TextureFormat::Astc10x6UnormSrgb => vk::Format::ASTC_10X6_SRGB_BLOCK,
        TextureFormat::Astc10x8Unorm => vk::Format::ASTC_10X8_UNORM_BLOCK,
        TextureFormat::Astc10x8UnormSrgb => vk::Format::ASTC_10X8_SRGB_BLOCK,
        TextureFormat::Astc10x10Unorm => vk::Format::ASTC_10X10_UNORM_BLOCK,
        TextureFormat::Astc10x10UnormSrgb => vk::Format::ASTC_10X10_SRGB_BLOCK,
        TextureFormat::Astc12x10Unorm => vk::Format::ASTC_12X10_UNORM_BLOCK,
        TextureFormat::Astc12x10UnormSrgb => vk::Format::ASTC_12X10_SRGB_BLOCK,
        TextureFormat::Astc12x12Unorm => vk::Format::ASTC_12X12_UNORM_BLOCK,
        TextureFormat::Astc12x12UnormSrgb => vk::Format::ASTC_12X12_SRGB_BLOCK,
        _ => vk::Format::R8G8B8A8_UNORM,
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

/// Convert LoadOp to Vulkan attachment load op and clear value for stencil attachments.
pub fn convert_load_op_stencil(
    op: &crate::graph::LoadOp,
) -> (vk::AttachmentLoadOp, vk::ClearValue) {
    match op {
        crate::graph::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        crate::graph::LoadOp::DontCare => {
            (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default())
        }
        crate::graph::LoadOp::Clear(clear_value) => {
            let stencil = match clear_value {
                crate::types::ClearValue::Stencil(s) => *s,
                crate::types::ClearValue::DepthStencil { stencil, .. } => *stencil,
                _ => 0,
            };
            (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil,
                    },
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

/// Convert BlendFactor to Vulkan blend factor.
pub fn convert_blend_factor(factor: crate::materials::BlendFactor) -> vk::BlendFactor {
    match factor {
        crate::materials::BlendFactor::Zero => vk::BlendFactor::ZERO,
        crate::materials::BlendFactor::One => vk::BlendFactor::ONE,
        crate::materials::BlendFactor::Src => vk::BlendFactor::SRC_COLOR,
        crate::materials::BlendFactor::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        crate::materials::BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        crate::materials::BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        crate::materials::BlendFactor::Dst => vk::BlendFactor::DST_COLOR,
        crate::materials::BlendFactor::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        crate::materials::BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        crate::materials::BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        crate::materials::BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        crate::materials::BlendFactor::Constant => vk::BlendFactor::CONSTANT_COLOR,
        crate::materials::BlendFactor::OneMinusConstant => {
            vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR
        }
    }
}

/// Convert BlendOperation to Vulkan blend op.
pub fn convert_blend_operation(op: crate::materials::BlendOperation) -> vk::BlendOp {
    match op {
        crate::materials::BlendOperation::Add => vk::BlendOp::ADD,
        crate::materials::BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        crate::materials::BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        crate::materials::BlendOperation::Min => vk::BlendOp::MIN,
        crate::materials::BlendOperation::Max => vk::BlendOp::MAX,
    }
}

/// Convert BlendComponent to Vulkan blend attachment state fields.
pub fn convert_blend_component(
    component: &crate::materials::BlendComponent,
) -> (vk::BlendFactor, vk::BlendFactor, vk::BlendOp) {
    (
        convert_blend_factor(component.src_factor),
        convert_blend_factor(component.dst_factor),
        convert_blend_operation(component.operation),
    )
}

/// Convert BlendState to Vulkan pipeline color blend attachment state.
pub fn convert_blend_state(
    state: &crate::materials::BlendState,
) -> vk::PipelineColorBlendAttachmentState {
    let (color_src, color_dst, color_op) = convert_blend_component(&state.color);
    let (alpha_src, alpha_dst, alpha_op) = convert_blend_component(&state.alpha);

    vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(color_src)
        .dst_color_blend_factor(color_dst)
        .color_blend_op(color_op)
        .src_alpha_blend_factor(alpha_src)
        .dst_alpha_blend_factor(alpha_dst)
        .alpha_blend_op(alpha_op)
        .color_write_mask(vk::ColorComponentFlags::RGBA)
}
