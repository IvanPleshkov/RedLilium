//! wgpu backend implementation

use crate::backend::traits::*;
use crate::backend::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Buffered render pass command
#[derive(Clone)]
enum RenderCommand {
    SetPipeline(RenderPipelineHandle),
    SetBindGroup { index: u32, bind_group: BindGroupHandle },
    SetVertexBuffer { slot: u32, buffer: BufferHandle, offset: u64 },
    SetIndexBuffer { buffer: BufferHandle, offset: u64, format: IndexFormat },
    SetViewport { x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32 },
    SetScissorRect { x: u32, y: u32, width: u32, height: u32 },
    Draw { vertices: std::ops::Range<u32>, instances: std::ops::Range<u32> },
    DrawIndexed { indices: std::ops::Range<u32>, base_vertex: i32, instances: std::ops::Range<u32> },
}

/// Buffered compute pass command
#[derive(Clone)]
enum ComputeCommand {
    SetPipeline(ComputePipelineHandle),
    SetBindGroup { index: u32, bind_group: BindGroupHandle },
    Dispatch { x: u32, y: u32, z: u32 },
}

/// Pending render pass with buffered commands
struct PendingRenderPass {
    descriptor: RenderPassDescriptor,
    commands: Vec<RenderCommand>,
}

/// Pending compute pass with buffered commands
struct PendingComputePass {
    label: Option<String>,
    commands: Vec<ComputeCommand>,
}

/// wgpu backend implementation
pub struct WgpuBackend {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    current_texture: Option<wgpu::SurfaceTexture>,
    current_view_id: u64,  // ID used to identify the swapchain view handle

    // Resource storage
    buffers: HashMap<u64, wgpu::Buffer>,
    textures: HashMap<u64, wgpu::Texture>,
    texture_views: HashMap<u64, wgpu::TextureView>,
    samplers: HashMap<u64, wgpu::Sampler>,
    bind_group_layouts: HashMap<u64, wgpu::BindGroupLayout>,
    bind_groups: HashMap<u64, wgpu::BindGroup>,
    render_pipelines: HashMap<u64, wgpu::RenderPipeline>,
    compute_pipelines: HashMap<u64, wgpu::ComputePipeline>,

    // Handle counters
    next_buffer_id: u64,
    next_texture_id: u64,
    next_view_id: u64,
    next_sampler_id: u64,
    next_layout_id: u64,
    next_bind_group_id: u64,
    next_render_pipeline_id: u64,
    next_compute_pipeline_id: u64,

    // Command encoding
    encoder: Option<wgpu::CommandEncoder>,

    // Pending passes - commands are buffered here and executed on end_*_pass
    pending_render_pass: Option<PendingRenderPass>,
    pending_compute_pass: Option<PendingComputePass>,
}

impl WgpuBackend {
    fn convert_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
        match format {
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
            TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
            TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
            TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        }
    }

    fn convert_texture_format_back(format: wgpu::TextureFormat) -> TextureFormat {
        match format {
            wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float => TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Depth32Float => TextureFormat::Depth32Float,
            wgpu::TextureFormat::Depth24PlusStencil8 => TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::R32Float => TextureFormat::R32Float,
            wgpu::TextureFormat::Rg32Float => TextureFormat::Rg32Float,
            _ => TextureFormat::Rgba8Unorm,
        }
    }

    fn convert_buffer_usage(usage: BufferUsage) -> wgpu::BufferUsages {
        let mut result = wgpu::BufferUsages::empty();
        if usage.contains(BufferUsage::MAP_READ) {
            result |= wgpu::BufferUsages::MAP_READ;
        }
        if usage.contains(BufferUsage::MAP_WRITE) {
            result |= wgpu::BufferUsages::MAP_WRITE;
        }
        if usage.contains(BufferUsage::COPY_SRC) {
            result |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.contains(BufferUsage::COPY_DST) {
            result |= wgpu::BufferUsages::COPY_DST;
        }
        if usage.contains(BufferUsage::INDEX) {
            result |= wgpu::BufferUsages::INDEX;
        }
        if usage.contains(BufferUsage::VERTEX) {
            result |= wgpu::BufferUsages::VERTEX;
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
        result
    }

    fn convert_texture_usage(usage: TextureUsage) -> wgpu::TextureUsages {
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

    fn convert_vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
        match format {
            VertexFormat::Float32 => wgpu::VertexFormat::Float32,
            VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
            VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
            VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
            VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
            VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
        }
    }

    fn convert_compare_function(func: CompareFunction) -> wgpu::CompareFunction {
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

    fn convert_blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
        match factor {
            BlendFactor::Zero => wgpu::BlendFactor::Zero,
            BlendFactor::One => wgpu::BlendFactor::One,
            BlendFactor::Src => wgpu::BlendFactor::Src,
            BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
            BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
            BlendFactor::Dst => wgpu::BlendFactor::Dst,
            BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
            BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
            BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        }
    }

    fn convert_blend_operation(op: BlendOperation) -> wgpu::BlendOperation {
        match op {
            BlendOperation::Add => wgpu::BlendOperation::Add,
            BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
            BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOperation::Min => wgpu::BlendOperation::Min,
            BlendOperation::Max => wgpu::BlendOperation::Max,
        }
    }

    fn convert_filter_mode(mode: FilterMode) -> wgpu::FilterMode {
        match mode {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }

    fn convert_address_mode(mode: AddressMode) -> wgpu::AddressMode {
        match mode {
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }

}

impl WgpuBackend {
    /// Async initialization - used directly on web, wrapped by `new` on native
    pub async fn new_async(window: Arc<winit::window::Window>, vsync: bool) -> BackendResult<Self> {
        // Platform-specific initialization
        #[cfg(target_arch = "wasm32")]
        let (instance, surface, adapter, device, queue) = Self::init_web(window.clone()).await?;

        #[cfg(not(target_arch = "wasm32"))]
        let (instance, surface, adapter, device, queue) = Self::init_native(window.clone()).await?;

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        // Clamp to device limits while maintaining aspect ratio
        let max_size = device.limits().max_texture_dimension_2d;
        let (clamped_width, clamped_height) = if size.width > max_size || size.height > max_size {
            // Calculate scale factor to fit within max_size while maintaining aspect ratio
            let scale = (max_size as f32 / size.width as f32).min(max_size as f32 / size.height as f32);
            let new_width = ((size.width as f32 * scale) as u32).max(1);
            let new_height = ((size.height as f32 * scale) as u32).max(1);
            (new_width, new_height)
        } else {
            (size.width.max(1), size.height.max(1))
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: clamped_width,
            height: clamped_height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        Ok(Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
            current_texture: None,
            current_view_id: 0,
            buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
            samplers: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            bind_groups: HashMap::new(),
            render_pipelines: HashMap::new(),
            compute_pipelines: HashMap::new(),
            next_buffer_id: 1,
            next_texture_id: 1,
            next_view_id: 1,
            next_sampler_id: 1,
            next_layout_id: 1,
            next_bind_group_id: 1,
            next_render_pipeline_id: 1,
            next_compute_pipeline_id: 1,
            encoder: None,
            pending_render_pass: None,
            pending_compute_pass: None,
        })
    }

    /// Web-specific initialization with WebGL2 default and WebGPU fallback
    #[cfg(target_arch = "wasm32")]
    async fn init_web(
        window: Arc<winit::window::Window>,
    ) -> BackendResult<(
        wgpu::Instance,
        wgpu::Surface<'static>,
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
    )> {
        use crate::web::console_log;

        // Try WebGL2 first (more compatible, default)
        console_log("Trying WebGL2 backend...");
        if let Ok(result) = Self::try_init_backend(
            window.clone(),
            wgpu::Backends::GL,
            wgpu::Limits::downlevel_webgl2_defaults(),
            "WebGL2",
        ).await {
            console_log("WebGL2 backend initialized successfully");
            return Ok(result);
        }

        // Fall back to WebGPU if WebGL2 fails
        console_log("WebGL2 failed, trying WebGPU backend...");
        if let Ok(result) = Self::try_init_backend(
            window.clone(),
            wgpu::Backends::BROWSER_WEBGPU,
            wgpu::Limits::default(),
            "WebGPU",
        ).await {
            console_log("WebGPU backend initialized successfully");
            return Ok(result);
        }

        Err(BackendError::InitializationFailed(
            "Neither WebGL2 nor WebGPU backends could be initialized".into()
        ))
    }

    /// Try to initialize a specific backend
    #[cfg(target_arch = "wasm32")]
    async fn try_init_backend(
        window: Arc<winit::window::Window>,
        backends: wgpu::Backends,
        limits: wgpu::Limits,
        backend_name: &str,
    ) -> BackendResult<(
        wgpu::Instance,
        wgpu::Surface<'static>,
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
    )> {
        use crate::web::console_log;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| BackendError::SurfaceCreationFailed(e.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                BackendError::InitializationFailed(format!("No {} adapter found", backend_name))
            })?;

        let adapter_info = adapter.get_info();
        console_log(&format!(
            "Found adapter: {} ({:?} backend)",
            adapter_info.name,
            adapter_info.backend
        ));

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Graphics Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .map_err(|e| BackendError::DeviceCreationFailed(e.to_string()))?;

        console_log(&format!(
            "Device created with max texture size: {}",
            device.limits().max_texture_dimension_2d
        ));

        Ok((instance, surface, adapter, device, queue))
    }

    /// Native initialization
    #[cfg(not(target_arch = "wasm32"))]
    async fn init_native(
        window: Arc<winit::window::Window>,
    ) -> BackendResult<(
        wgpu::Instance,
        wgpu::Surface<'static>,
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
    )> {
        // On Windows, try Vulkan first to avoid D3D12 debug layer validation errors
        let backends = if std::env::var("WGPU_BACKEND").is_ok() {
            wgpu::Backends::all()
        } else {
            #[cfg(target_os = "windows")]
            {
                wgpu::Backends::VULKAN
            }
            #[cfg(not(target_os = "windows"))]
            {
                wgpu::Backends::all()
            }
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| BackendError::SurfaceCreationFailed(e.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await;

        // If no adapter found with preferred backend, try with all backends
        let (instance, surface, adapter) = if adapter.is_none() && backends != wgpu::Backends::all() {
            log::warn!("Preferred backend not available, falling back to all backends");
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let surface = instance
                .create_surface(window.clone())
                .map_err(|e| BackendError::SurfaceCreationFailed(e.to_string()))?;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| {
                    BackendError::InitializationFailed("No suitable adapter found".into())
                })?;
            (instance, surface, adapter)
        } else {
            let adapter = adapter.ok_or_else(|| {
                BackendError::InitializationFailed("No suitable adapter found".into())
            })?;
            (instance, surface, adapter)
        };

        let adapter_info = adapter.get_info();
        log::info!(
            "Selected GPU: {} ({:?} backend)",
            adapter_info.name,
            adapter_info.backend
        );
        println!(
            "Selected GPU: {} ({:?} backend)",
            adapter_info.name,
            adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Graphics Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| BackendError::DeviceCreationFailed(e.to_string()))?;

        Ok((instance, surface, adapter, device, queue))
    }
}

impl GraphicsBackend for WgpuBackend {
    #[cfg(not(target_arch = "wasm32"))]
    fn new(window: Arc<winit::window::Window>, vsync: bool) -> BackendResult<Self> {
        pollster::block_on(Self::new_async(window, vsync))
    }

    #[cfg(target_arch = "wasm32")]
    fn new(_window: Arc<winit::window::Window>, _vsync: bool) -> BackendResult<Self> {
        // On web, use new_async instead
        Err(BackendError::InitializationFailed(
            "Use WgpuBackend::new_async() on web platform".into()
        ))
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // Clamp to device limits while maintaining aspect ratio
            let max_size = self.device.limits().max_texture_dimension_2d;
            let (clamped_width, clamped_height) = if width > max_size || height > max_size {
                // Calculate scale factor to fit within max_size while maintaining aspect ratio
                let scale = (max_size as f32 / width as f32).min(max_size as f32 / height as f32);
                let new_width = ((width as f32 * scale) as u32).max(1);
                let new_height = ((height as f32 * scale) as u32).max(1);
                (new_width, new_height)
            } else {
                (width, height)
            };

            self.surface_config.width = clamped_width;
            self.surface_config.height = clamped_height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    fn begin_frame(&mut self) -> BackendResult<FrameContext> {
        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| match e {
                wgpu::SurfaceError::Lost => BackendError::SurfaceLost,
                wgpu::SurfaceError::OutOfMemory => BackendError::OutOfMemory,
                _ => BackendError::AcquireImageFailed(e.to_string()),
            })?;

        // Use a unique ID for the swapchain view - we'll create the view on demand
        let view_id = self.next_view_id;
        self.next_view_id += 1;
        self.current_view_id = view_id;
        // Don't create view here - create it fresh when needed

        let width = self.surface_config.width;
        let height = self.surface_config.height;

        self.current_texture = Some(output);
        self.encoder = Some(
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Frame Encoder"),
                }),
        );

        Ok(FrameContext {
            swapchain_view: TextureViewHandle(view_id),
            width,
            height,
        })
    }

    fn end_frame(&mut self) -> BackendResult<()> {
        // Submit any pending commands
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Present the swapchain
        if let Some(texture) = self.current_texture.take() {
            texture.present();
        }

        Ok(())
    }

    fn swapchain_format(&self) -> TextureFormat {
        Self::convert_texture_format_back(self.surface_config.format)
    }

    fn create_buffer(&mut self, desc: &BufferDescriptor) -> BackendResult<BufferHandle> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label.as_deref(),
            size: desc.size,
            usage: Self::convert_buffer_usage(desc.usage),
            mapped_at_creation: desc.mapped_at_creation,
        });

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(id, buffer);

        Ok(BufferHandle(id))
    }

    fn create_buffer_init(
        &mut self,
        desc: &BufferDescriptor,
        data: &[u8],
    ) -> BackendResult<BufferHandle> {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: desc.label.as_deref(),
            contents: data,
            usage: Self::convert_buffer_usage(desc.usage),
        });

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(id, buffer);

        Ok(BufferHandle(id))
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) {
        if let Some(buf) = self.buffers.get(&buffer.0) {
            self.queue.write_buffer(buf, offset, data);
        }
    }

    fn create_texture(&mut self, desc: &TextureDescriptor) -> BackendResult<TextureHandle> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label.as_deref(),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.depth,
            },
            mip_level_count: desc.mip_levels,
            sample_count: 1,
            dimension: if desc.depth > 1 {
                wgpu::TextureDimension::D3
            } else {
                wgpu::TextureDimension::D2
            },
            format: Self::convert_texture_format(desc.format),
            usage: Self::convert_texture_usage(desc.usage),
            view_formats: &[],
        });

        let id = self.next_texture_id;
        self.next_texture_id += 1;
        self.textures.insert(id, texture);

        Ok(TextureHandle(id))
    }

    fn create_texture_view(&mut self, texture: TextureHandle) -> BackendResult<TextureViewHandle> {
        let tex = self
            .textures
            .get(&texture.0)
            .ok_or_else(|| BackendError::TextureCreationFailed("Texture not found".into()))?;

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let id = self.next_view_id;
        self.next_view_id += 1;
        self.texture_views.insert(id, view);

        Ok(TextureViewHandle(id))
    }

    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) {
        if let Some(tex) = self.textures.get(&texture.0) {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn create_sampler(&mut self, desc: &SamplerDescriptor) -> BackendResult<SamplerHandle> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label.as_deref(),
            address_mode_u: Self::convert_address_mode(desc.address_mode_u),
            address_mode_v: Self::convert_address_mode(desc.address_mode_v),
            address_mode_w: Self::convert_address_mode(desc.address_mode_w),
            mag_filter: Self::convert_filter_mode(desc.mag_filter),
            min_filter: Self::convert_filter_mode(desc.min_filter),
            mipmap_filter: Self::convert_filter_mode(desc.mipmap_filter),
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: desc.compare.map(Self::convert_compare_function),
            anisotropy_clamp: 1,
            border_color: None,
        });

        let id = self.next_sampler_id;
        self.next_sampler_id += 1;
        self.samplers.insert(id, sampler);

        Ok(SamplerHandle(id))
    }

    fn create_bind_group_layout(
        &mut self,
        entries: &[BindGroupLayoutEntry],
    ) -> BackendResult<BindGroupLayoutHandle> {
        let wgpu_entries: Vec<wgpu::BindGroupLayoutEntry> = entries
            .iter()
            .map(|e| {
                let ty = match &e.ty {
                    BindingType::UniformBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::StorageBuffer { read_only } => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::Texture { sample_type } => wgpu::BindingType::Texture {
                        sample_type: match sample_type {
                            TextureSampleType::Float { filterable } => {
                                wgpu::TextureSampleType::Float { filterable: *filterable }
                            }
                            TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
                            TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
                            TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    BindingType::StorageTexture { format } => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: Self::convert_texture_format(*format),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    BindingType::Sampler { comparison } => wgpu::BindingType::Sampler(
                        if *comparison {
                            wgpu::SamplerBindingType::Comparison
                        } else {
                            wgpu::SamplerBindingType::Filtering
                        },
                    ),
                };

                let mut visibility = wgpu::ShaderStages::empty();
                if e.visibility.contains(ShaderStageFlags::VERTEX) {
                    visibility |= wgpu::ShaderStages::VERTEX;
                }
                if e.visibility.contains(ShaderStageFlags::FRAGMENT) {
                    visibility |= wgpu::ShaderStages::FRAGMENT;
                }
                if e.visibility.contains(ShaderStageFlags::COMPUTE) {
                    visibility |= wgpu::ShaderStages::COMPUTE;
                }

                wgpu::BindGroupLayoutEntry {
                    binding: e.binding,
                    visibility,
                    ty,
                    count: None,
                }
            })
            .collect();

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &wgpu_entries,
            });

        let id = self.next_layout_id;
        self.next_layout_id += 1;
        self.bind_group_layouts.insert(id, layout);

        Ok(BindGroupLayoutHandle(id))
    }

    fn create_bind_group(
        &mut self,
        layout: BindGroupLayoutHandle,
        entries: &[(u32, BindGroupEntry)],
    ) -> BackendResult<BindGroupHandle> {
        let layout_ref = self
            .bind_group_layouts
            .get(&layout.0)
            .ok_or_else(|| BackendError::PipelineCreationFailed("Layout not found".into()))?;

        let wgpu_entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .filter_map(|(binding, entry)| {
                let resource = match entry {
                    BindGroupEntry::Buffer { buffer, offset, size } => {
                        let buf = self.buffers.get(&buffer.0)?;
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: buf,
                            offset: *offset,
                            size: size.and_then(std::num::NonZeroU64::new),
                        })
                    }
                    BindGroupEntry::Texture(view) => {
                        let v = self.texture_views.get(&view.0)?;
                        wgpu::BindingResource::TextureView(v)
                    }
                    BindGroupEntry::Sampler(sampler) => {
                        let s = self.samplers.get(&sampler.0)?;
                        wgpu::BindingResource::Sampler(s)
                    }
                    BindGroupEntry::StorageTexture(view) => {
                        let v = self.texture_views.get(&view.0)?;
                        wgpu::BindingResource::TextureView(v)
                    }
                };

                Some(wgpu::BindGroupEntry {
                    binding: *binding,
                    resource,
                })
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: layout_ref,
            entries: &wgpu_entries,
        });

        let id = self.next_bind_group_id;
        self.next_bind_group_id += 1;
        self.bind_groups.insert(id, bind_group);

        Ok(BindGroupHandle(id))
    }

    fn create_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor,
    ) -> BackendResult<RenderPipelineHandle> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(desc.vertex_shader.as_str().into()),
            });

        let layouts: Vec<&wgpu::BindGroupLayout> = desc
            .bind_group_layouts
            .iter()
            .filter_map(|h| self.bind_group_layouts.get(&h.0))
            .collect();

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layouts,
                push_constant_ranges: &[],
            });

        // Build vertex buffer layouts with proper lifetimes
        let vertex_attrs: Vec<Vec<wgpu::VertexAttribute>> = desc
            .vertex_layouts
            .iter()
            .map(|layout| {
                layout
                    .attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: Self::convert_vertex_format(a.format),
                        offset: a.offset,
                        shader_location: a.location,
                    })
                    .collect()
            })
            .collect();

        let vertex_buffers: Vec<wgpu::VertexBufferLayout> = desc
            .vertex_layouts
            .iter()
            .zip(vertex_attrs.iter())
            .map(|(layout, attrs)| wgpu::VertexBufferLayout {
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
                    VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
                },
                attributes: attrs,
            })
            .collect();

        let color_targets: Vec<Option<wgpu::ColorTargetState>> = desc
            .color_targets
            .iter()
            .map(|target| {
                Some(wgpu::ColorTargetState {
                    format: Self::convert_texture_format(target.format),
                    blend: target.blend.as_ref().map(|b| wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: Self::convert_blend_factor(b.color.src_factor),
                            dst_factor: Self::convert_blend_factor(b.color.dst_factor),
                            operation: Self::convert_blend_operation(b.color.operation),
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: Self::convert_blend_factor(b.alpha.src_factor),
                            dst_factor: Self::convert_blend_factor(b.alpha.dst_factor),
                            operation: Self::convert_blend_operation(b.alpha.operation),
                        },
                    }),
                    write_mask: wgpu::ColorWrites::from_bits_truncate(target.write_mask.0),
                })
            })
            .collect();

        let primitive = wgpu::PrimitiveState {
            topology: match desc.primitive_topology {
                PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
                PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
                PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
                PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
            },
            strip_index_format: None,
            front_face: match desc.front_face {
                FrontFace::Ccw => wgpu::FrontFace::Ccw,
                FrontFace::Cw => wgpu::FrontFace::Cw,
            },
            cull_mode: match desc.cull_mode {
                CullMode::None => None,
                CullMode::Front => Some(wgpu::Face::Front),
                CullMode::Back => Some(wgpu::Face::Back),
            },
            ..Default::default()
        };

        let depth_stencil = desc.depth_stencil.as_ref().map(|ds| wgpu::DepthStencilState {
            format: Self::convert_texture_format(ds.format),
            depth_write_enabled: ds.depth_write_enabled,
            depth_compare: Self::convert_compare_function(ds.depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: desc.label.as_deref(),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &vertex_buffers,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: desc.fragment_shader.as_ref().map(|_| wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive,
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let id = self.next_render_pipeline_id;
        self.next_render_pipeline_id += 1;
        self.render_pipelines.insert(id, pipeline);

        Ok(RenderPipelineHandle(id))
    }

    fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor,
    ) -> BackendResult<ComputePipelineHandle> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(desc.shader.as_str().into()),
            });

        let layouts: Vec<&wgpu::BindGroupLayout> = desc
            .bind_group_layouts
            .iter()
            .filter_map(|h| self.bind_group_layouts.get(&h.0))
            .collect();

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layouts,
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: desc.label.as_deref(),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: &desc.entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let id = self.next_compute_pipeline_id;
        self.next_compute_pipeline_id += 1;
        self.compute_pipelines.insert(id, pipeline);

        Ok(ComputePipelineHandle(id))
    }

    fn begin_render_pass(&mut self, desc: &RenderPassDescriptor) {
        // Store the descriptor for later execution
        self.pending_render_pass = Some(PendingRenderPass {
            descriptor: desc.clone(),
            commands: Vec::new(),
        });
    }

    fn end_render_pass(&mut self) {
        // Take the pending pass and encoder temporarily
        let Some(pending) = self.pending_render_pass.take() else {
            return;
        };

        let Some(mut encoder) = self.encoder.take() else {
            return;
        };

        // Create swapchain view if needed - scope it to be dropped before encoder is used
        let swapchain_view: Option<wgpu::TextureView> = self.current_texture.as_ref().map(|tex| {
            tex.texture.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let current_view_id = self.current_view_id;

        // Execute render pass in its own scope
        {
            // Build color attachments
            let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = pending
                .descriptor
                .color_attachments
                .iter()
                .filter_map(|att| {
                    let view = if att.view.0 == current_view_id {
                        swapchain_view.as_ref()?
                    } else {
                        self.texture_views.get(&att.view.0)?
                    };
                    Some(Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: match &att.load_op {
                                LoadOp::Clear(color) => wgpu::LoadOp::Clear(wgpu::Color {
                                    r: color[0] as f64,
                                    g: color[1] as f64,
                                    b: color[2] as f64,
                                    a: color[3] as f64,
                                }),
                                LoadOp::Load => wgpu::LoadOp::Load,
                            },
                            store: match att.store_op {
                                StoreOp::Store => wgpu::StoreOp::Store,
                                StoreOp::Discard => wgpu::StoreOp::Discard,
                            },
                        },
                    }))
                })
                .collect();

            // Build depth attachment
            let depth_attachment = pending.descriptor.depth_stencil_attachment.as_ref().and_then(|att| {
                let view = if att.view.0 == current_view_id {
                    swapchain_view.as_ref()?
                } else {
                    self.texture_views.get(&att.view.0)?
                };
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: match &att.depth_load_op {
                            LoadOp::Clear(_) => wgpu::LoadOp::Clear(att.depth_clear_value),
                            LoadOp::Load => wgpu::LoadOp::Load,
                        },
                        store: match att.depth_store_op {
                            StoreOp::Store => wgpu::StoreOp::Store,
                            StoreOp::Discard => wgpu::StoreOp::Discard,
                        },
                    }),
                    stencil_ops: None,
                })
            });

            // Create render pass and execute all buffered commands
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: pending.descriptor.label.as_deref(),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: depth_attachment,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                // Execute buffered commands
                for cmd in &pending.commands {
                    match cmd {
                        RenderCommand::SetPipeline(handle) => {
                            if let Some(pipeline) = self.render_pipelines.get(&handle.0) {
                                render_pass.set_pipeline(pipeline);
                            }
                        }
                        RenderCommand::SetBindGroup { index, bind_group } => {
                            if let Some(bg) = self.bind_groups.get(&bind_group.0) {
                                render_pass.set_bind_group(*index, bg, &[]);
                            }
                        }
                        RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                            if let Some(buf) = self.buffers.get(&buffer.0) {
                                render_pass.set_vertex_buffer(*slot, buf.slice(*offset..));
                            }
                        }
                        RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                            if let Some(buf) = self.buffers.get(&buffer.0) {
                                let wgpu_format = match format {
                                    IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                                    IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                                };
                                render_pass.set_index_buffer(buf.slice(*offset..), wgpu_format);
                            }
                        }
                        RenderCommand::SetViewport { x, y, width, height, min_depth, max_depth } => {
                            render_pass.set_viewport(*x, *y, *width, *height, *min_depth, *max_depth);
                        }
                        RenderCommand::SetScissorRect { x, y, width, height } => {
                            render_pass.set_scissor_rect(*x, *y, *width, *height);
                        }
                        RenderCommand::Draw { vertices, instances } => {
                            render_pass.draw(vertices.clone(), instances.clone());
                        }
                        RenderCommand::DrawIndexed { indices, base_vertex, instances } => {
                            render_pass.draw_indexed(indices.clone(), *base_vertex, instances.clone());
                        }
                    }
                }
                // render_pass is dropped here, ending the pass with proper state transitions
            }
        }

        // Put encoder back
        self.encoder = Some(encoder);
    }

    fn begin_compute_pass(&mut self, label: Option<&str>) {
        self.pending_compute_pass = Some(PendingComputePass {
            label: label.map(|s| s.to_string()),
            commands: Vec::new(),
        });
    }

    fn end_compute_pass(&mut self) {
        let Some(pending) = self.pending_compute_pass.take() else {
            return;
        };

        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };

        let compute_pipelines = &self.compute_pipelines;
        let bind_groups = &self.bind_groups;

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: pending.label.as_deref(),
                timestamp_writes: None,
            });

            for cmd in &pending.commands {
                match cmd {
                    ComputeCommand::SetPipeline(handle) => {
                        if let Some(pipeline) = compute_pipelines.get(&handle.0) {
                            compute_pass.set_pipeline(pipeline);
                        }
                    }
                    ComputeCommand::SetBindGroup { index, bind_group } => {
                        if let Some(bg) = bind_groups.get(&bind_group.0) {
                            compute_pass.set_bind_group(*index, bg, &[]);
                        }
                    }
                    ComputeCommand::Dispatch { x, y, z } => {
                        compute_pass.dispatch_workgroups(*x, *y, *z);
                    }
                }
            }
        }
    }

    fn set_render_pipeline(&mut self, pipeline: RenderPipelineHandle) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetPipeline(pipeline));
        }
    }

    fn set_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(ComputeCommand::SetPipeline(pipeline));
        }
    }

    fn set_bind_group(&mut self, index: u32, bind_group: BindGroupHandle) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetBindGroup { index, bind_group });
        } else if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(ComputeCommand::SetBindGroup { index, bind_group });
        }
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetVertexBuffer { slot, buffer, offset });
        }
    }

    fn set_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetIndexBuffer { buffer, offset, format });
        }
    }

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetViewport { x, y, width, height, min_depth, max_depth });
        }
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::SetScissorRect { x, y, width, height });
        }
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::Draw { vertices, instances });
        }
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(RenderCommand::DrawIndexed { indices, base_vertex, instances });
        }
    }

    fn dispatch_compute(&mut self, x: u32, y: u32, z: u32) {
        if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(ComputeCommand::Dispatch { x, y, z });
        }
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        self.buffers.remove(&buffer.0);
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        self.textures.remove(&texture.0);
    }
}

// Additional methods for egui integration and external rendering
impl WgpuBackend {
    /// Get reference to the wgpu device (for egui-wgpu Renderer creation)
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to the wgpu queue (for egui-wgpu buffer updates)
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the surface format as wgpu type (for egui-wgpu Renderer creation)
    pub fn wgpu_surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// Get mutable reference to the command encoder (for egui-wgpu buffer updates)
    /// Only valid during a frame (between begin_frame and end_frame)
    pub fn encoder_mut(&mut self) -> Option<&mut wgpu::CommandEncoder> {
        self.encoder.as_mut()
    }

    /// Get device, queue, and encoder together for operations that need all three.
    /// This avoids borrow checker issues when calling external libraries like egui.
    pub fn device_queue_encoder(&mut self) -> (&wgpu::Device, &wgpu::Queue, Option<&mut wgpu::CommandEncoder>) {
        (&self.device, &self.queue, self.encoder.as_mut())
    }

    /// Get handle to the current swapchain view for external rendering (e.g., egui).
    /// Returns None if not within a frame (between begin_frame and end_frame).
    pub fn current_swapchain_view(&self) -> Option<TextureViewHandle> {
        if self.current_texture.is_some() {
            Some(TextureViewHandle(self.current_view_id))
        } else {
            None
        }
    }

    /// Render egui to the swapchain. This method handles the render pass creation
    /// internally to avoid lifetime issues with closures.
    pub fn render_egui(
        &mut self,
        renderer: &egui_wgpu::Renderer,
        paint_jobs: &[egui::ClippedPrimitive],
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        swapchain_view: TextureViewHandle,
    ) {
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };

        // Create swapchain view
        let Some(swapchain_texture_view) = self.current_texture.as_ref().map(|tex| {
            tex.texture.create_view(&wgpu::TextureViewDescriptor::default())
        }) else {
            return;
        };

        let current_view_id = self.current_view_id;

        // Determine which view to use
        let view = if swapchain_view.0 == current_view_id {
            &swapchain_texture_view
        } else if let Some(v) = self.texture_views.get(&swapchain_view.0) {
            v
        } else {
            return;
        };

        // Create render pass for egui
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve existing content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            renderer.render(&mut render_pass, paint_jobs, screen_descriptor);
        }
    }

    /// Execute a callback with direct access to a wgpu render pass.
    /// This is specifically designed for egui integration which requires
    /// a mutable RenderPass reference.
    ///
    /// # Arguments
    /// * `desc` - Render pass descriptor
    /// * `callback` - Function that receives the mutable render pass
    pub fn with_render_pass<F>(&mut self, desc: &RenderPassDescriptor, callback: F)
    where
        F: FnOnce(&mut wgpu::RenderPass<'_>),
    {
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };

        // Create swapchain view if needed
        let swapchain_view: Option<wgpu::TextureView> = self.current_texture.as_ref().map(|tex| {
            tex.texture.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let current_view_id = self.current_view_id;

        // Build color attachments
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = desc
            .color_attachments
            .iter()
            .filter_map(|att| {
                let view = if att.view.0 == current_view_id {
                    swapchain_view.as_ref()?
                } else {
                    self.texture_views.get(&att.view.0)?
                };
                Some(Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match &att.load_op {
                            LoadOp::Clear(color) => wgpu::LoadOp::Clear(wgpu::Color {
                                r: color[0] as f64,
                                g: color[1] as f64,
                                b: color[2] as f64,
                                a: color[3] as f64,
                            }),
                            LoadOp::Load => wgpu::LoadOp::Load,
                        },
                        store: match att.store_op {
                            StoreOp::Store => wgpu::StoreOp::Store,
                            StoreOp::Discard => wgpu::StoreOp::Discard,
                        },
                    },
                }))
            })
            .collect();

        // Build depth attachment (if needed)
        let depth_attachment = desc.depth_stencil_attachment.as_ref().and_then(|att| {
            let view = if att.view.0 == current_view_id {
                swapchain_view.as_ref()?
            } else {
                self.texture_views.get(&att.view.0)?
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: match &att.depth_load_op {
                        LoadOp::Clear(_) => wgpu::LoadOp::Clear(att.depth_clear_value),
                        LoadOp::Load => wgpu::LoadOp::Load,
                    },
                    store: match att.depth_store_op {
                        StoreOp::Store => wgpu::StoreOp::Store,
                        StoreOp::Discard => wgpu::StoreOp::Discard,
                    },
                }),
                stencil_ops: None,
            })
        });

        // Create render pass and invoke callback
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: desc.label.as_deref(),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            callback(&mut render_pass);
        }
    }
}
