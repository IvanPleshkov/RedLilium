//! Common types shared between backends

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

/// Texture format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgba16Float,
    Rgba32Float,
    Depth32Float,
    Depth24PlusStencil8,
    R32Float,
    Rg32Float,
}

impl TextureFormat {
    pub fn is_depth(&self) -> bool {
        matches!(
            self,
            TextureFormat::Depth32Float | TextureFormat::Depth24PlusStencil8
        )
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Depth32Float
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::R32Float => 4,
            TextureFormat::Rgba16Float | TextureFormat::Rg32Float => 8,
            TextureFormat::Rgba32Float => 16,
        }
    }
}

/// Texture usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage(u32);

impl TextureUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const TEXTURE_BINDING: Self = Self(1 << 2);
    pub const STORAGE_BINDING: Self = Self(1 << 3);
    pub const RENDER_ATTACHMENT: Self = Self(1 << 4);

    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u32);

impl BufferUsage {
    pub const MAP_READ: Self = Self(1 << 0);
    pub const MAP_WRITE: Self = Self(1 << 1);
    pub const COPY_SRC: Self = Self(1 << 2);
    pub const COPY_DST: Self = Self(1 << 3);
    pub const INDEX: Self = Self(1 << 4);
    pub const VERTEX: Self = Self(1 << 5);
    pub const UNIFORM: Self = Self(1 << 6);
    pub const STORAGE: Self = Self(1 << 7);
    pub const INDIRECT: Self = Self(1 << 8);

    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn bits(&self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Texture descriptor
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub label: Option<String>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            width: 1,
            height: 1,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        }
    }
}

/// Buffer descriptor
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub label: Option<String>,
    pub size: u64,
    pub usage: BufferUsage,
    pub mapped_at_creation: bool,
}

/// Vertex attribute format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Sint32,
}

impl VertexFormat {
    pub fn size(&self) -> u64 {
        match self {
            VertexFormat::Float32 | VertexFormat::Uint32 | VertexFormat::Sint32 => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
        }
    }
}

/// Vertex attribute description
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u64,
}

/// Vertex buffer layout
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

/// Standard vertex with position, normal, UV, and tangent
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub tangent: Vec4,
}

impl Vertex {
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::Float32x3,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::Float32x3,
                    offset: 12,
                },
                VertexAttribute {
                    location: 2,
                    format: VertexFormat::Float32x2,
                    offset: 24,
                },
                VertexAttribute {
                    location: 3,
                    format: VertexFormat::Float32x4,
                    offset: 32,
                },
            ],
        }
    }
}

/// Camera uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view: Mat4,
    pub inv_proj: Mat4,
    pub position: Vec4, // w unused
    pub near_far: Vec4, // x=near, y=far, zw unused
}

/// Per-object uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ObjectUniform {
    pub model: Mat4,
    pub normal_matrix: Mat4,
}

/// Primitive topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

/// Front face winding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontFace {
    Ccw,
    Cw,
}

/// Cull mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// Compare function for depth/stencil
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Blend factor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
}

/// Blend operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Blend component state
#[derive(Debug, Clone, Copy)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        }
    }
}

/// Blend state
#[derive(Debug, Clone, Copy)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub fn alpha_blending() -> Self {
        Self {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        }
    }

    pub fn additive() -> Self {
        Self {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent::default(),
        }
    }
}

/// Shader stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// Filter mode for samplers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Address mode for samplers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Sampler descriptor
#[derive(Debug, Clone)]
pub struct SamplerDescriptor {
    pub label: Option<String>,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub compare: Option<CompareFunction>,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            compare: None,
        }
    }
}
