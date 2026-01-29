//! Vulkan egui integration using egui-ash-renderer
//!
//! Provides egui rendering support for the Vulkan backend.

use ash::vk;
use egui_ash_renderer::{Options, Renderer};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::sync::{Arc, Mutex};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::backend::vulkan::VulkanBackend;

/// Vulkan-specific egui integration
pub struct VulkanEguiIntegration {
    /// egui context
    ctx: egui::Context,
    /// egui-winit state for input handling
    winit_state: egui_winit::State,
    /// egui-ash renderer (must be dropped before allocator)
    renderer: Option<Renderer>,
    /// Allocator owned by this integration (required by egui-ash-renderer)
    /// Note: This allocator uses cloned device/instance handles from VulkanBackend.
    /// It MUST be dropped before VulkanBackend destroys the actual Vulkan device.
    allocator: Option<Arc<Mutex<Allocator>>>,
    /// Cached paint jobs
    paint_jobs: Vec<egui::ClippedPrimitive>,
    /// Cached textures delta
    textures_delta: egui::TexturesDelta,
    /// Scale factor for input coordinates
    input_scale: f32,
}

impl VulkanEguiIntegration {
    /// Create a new Vulkan egui integration
    pub fn new(
        backend: &VulkanBackend,
        window: &Window,
    ) -> Self {
        let ctx = egui::Context::default();

        // Create egui-winit state
        let viewport_id = egui::ViewportId::ROOT;
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
        );

        // Create a separate allocator for egui (requires std::sync::Mutex)
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: backend.instance().clone(),
            device: backend.device().clone(),
            physical_device: backend.physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .expect("Failed to create egui allocator");
        let allocator = Arc::new(Mutex::new(allocator));

        // Create egui-ash renderer
        let renderer = Renderer::with_gpu_allocator(
            allocator.clone(),
            backend.device().clone(),
            backend.egui_render_pass(),
            Options {
                srgb_framebuffer: true,
                ..Default::default()
            },
        )
        .expect("Failed to create egui-ash renderer");

        Self {
            ctx,
            winit_state,
            renderer: Some(renderer),
            allocator: Some(allocator),
            paint_jobs: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            input_scale: 1.0,
        }
    }

    /// Destroy GPU resources. Must be called before VulkanBackend is dropped.
    /// This ensures the renderer and allocator are cleaned up while the Vulkan device is still valid.
    pub fn destroy(&mut self, backend: &VulkanBackend) {
        unsafe {
            // Wait for GPU to finish all operations
            let _ = backend.device().device_wait_idle();
        }

        // Drop renderer first (it uses the allocator)
        self.renderer = None;

        // Then drop allocator
        self.allocator = None;
    }

    /// Set the scale factor for input coordinates
    pub fn set_surface_scale(
        &mut self,
        window_width: u32,
        window_height: u32,
        surface_width: u32,
        surface_height: u32,
    ) {
        let scale_x = surface_width as f32 / window_width as f32;
        let scale_y = surface_height as f32 / window_height as f32;
        self.input_scale = scale_x.min(scale_y);
    }

    /// Handle a winit window event
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let scaled_event = if self.input_scale != 1.0 {
            match event {
                WindowEvent::CursorMoved {
                    device_id,
                    position,
                } => {
                    let scaled_pos = winit::dpi::PhysicalPosition::new(
                        position.x * self.input_scale as f64,
                        position.y * self.input_scale as f64,
                    );
                    Some(WindowEvent::CursorMoved {
                        device_id: *device_id,
                        position: scaled_pos,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        let event_to_use = scaled_event.as_ref().unwrap_or(event);
        let response = self.winit_state.on_window_event(window, event_to_use);
        response.consumed
    }

    /// Begin a new egui frame
    pub fn begin_frame(&mut self, window: &Window) {
        let mut raw_input = self.winit_state.take_egui_input(window);

        if self.input_scale != 1.0 {
            if let Some(rect) = &mut raw_input.screen_rect {
                rect.max.x *= self.input_scale;
                rect.max.y *= self.input_scale;
            }
        }

        self.ctx.begin_frame(raw_input);
    }

    /// End the egui frame
    pub fn end_frame(&mut self, window: &Window) {
        let full_output = self.ctx.end_frame();

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        self.paint_jobs = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        self.textures_delta = full_output.textures_delta;
    }

    /// Render egui to a Vulkan command buffer
    ///
    /// # Safety
    /// The command buffer must be in a recording state.
    ///
    /// Note: This method creates a temporary framebuffer for each render call.
    /// For better performance, framebuffers should be cached per swapchain image.
    pub unsafe fn render(
        &mut self,
        backend: &VulkanBackend,
        command_buffer: vk::CommandBuffer,
        swapchain_image_view: vk::ImageView,
        screen_width: u32,
        screen_height: u32,
    ) {
        let device = backend.device();
        let render_pass = backend.egui_render_pass();

        // Create a temporary framebuffer for this swapchain image
        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: 1,
            p_attachments: &swapchain_image_view,
            width: screen_width,
            height: screen_height,
            layers: 1,
            ..Default::default()
        };
        let framebuffer = device
            .create_framebuffer(&framebuffer_info, None)
            .expect("Failed to create egui framebuffer");

        // Clear to a dark background color (matches the wgpu demo)
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.15, 1.0], // Dark blue-gray
            },
        }];

        // Begin render pass - the render pass handles layout transition from UNDEFINED
        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: screen_width,
                    height: screen_height,
                },
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        device.cmd_begin_render_pass(command_buffer, &render_pass_begin, vk::SubpassContents::INLINE);

        if let Some(ref mut renderer) = self.renderer {
            // Update textures
            renderer
                .set_textures(
                    backend.graphics_queue(),
                    backend.command_pool(),
                    self.textures_delta.set.drain(..).collect::<Vec<_>>().as_slice(),
                )
                .expect("Failed to set egui textures");

            // Render egui
            renderer
                .cmd_draw(
                    command_buffer,
                    vk::Extent2D {
                        width: screen_width,
                        height: screen_height,
                    },
                    self.ctx.pixels_per_point(),
                    &self.paint_jobs,
                )
                .expect("Failed to draw egui");
        }

        // End render pass - transitions to PRESENT_SRC_KHR
        device.cmd_end_render_pass(command_buffer);

        // Destroy temporary framebuffer
        // Note: This is safe because the command buffer hasn't been submitted yet
        device.destroy_framebuffer(framebuffer, None);

        // Free old textures
        if let Some(ref mut renderer) = self.renderer {
            renderer
                .free_textures(self.textures_delta.free.drain(..).collect::<Vec<_>>().as_slice())
                .expect("Failed to free egui textures");
        }
    }

    /// Get the egui context
    pub fn context(&self) -> &egui::Context {
        &self.ctx
    }

    /// Check if egui wants keyboard input
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    /// Check if egui wants pointer input
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }
}

impl Drop for VulkanEguiIntegration {
    fn drop(&mut self) {
        // If destroy() wasn't called, the renderer and allocator will be dropped here.
        // This may cause issues if VulkanBackend was already dropped.
        // Always call destroy() before dropping VulkanBackend!
        if self.renderer.is_some() || self.allocator.is_some() {
            log::warn!("VulkanEguiIntegration::destroy() was not called before drop. This may cause issues.");
        }
    }
}
