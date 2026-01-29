//! wgpu egui integration
//!
//! Provides egui rendering support for the wgpu backend.

use egui::ViewportId;
use egui_wgpu::ScreenDescriptor;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::backend::traits::TextureViewHandle;
use crate::backend::wgpu_backend::WgpuBackend;

/// wgpu-specific egui integration
pub struct WgpuEguiIntegration {
    /// egui context (shared state for UI)
    ctx: egui::Context,
    /// egui-winit state for input handling
    winit_state: egui_winit::State,
    /// egui-wgpu renderer for drawing
    renderer: egui_wgpu::Renderer,
    /// Cached paint jobs from last frame
    paint_jobs: Vec<egui::ClippedPrimitive>,
    /// Cached textures delta
    textures_delta: egui::TexturesDelta,
    /// Scale factor for input coordinates (window size / surface size)
    input_scale: f32,
}

impl WgpuEguiIntegration {
    /// Create a new egui integration instance
    pub fn new(backend: &WgpuBackend, window: &Window) -> Self {
        let ctx = egui::Context::default();

        let viewport_id = ViewportId::ROOT;
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
        );

        let renderer = egui_wgpu::Renderer::new(
            backend.device(),
            backend.wgpu_surface_format(),
            None,
            1,
        );

        Self {
            ctx,
            winit_state,
            renderer,
            paint_jobs: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            input_scale: 1.0,
        }
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

    /// Render egui
    pub fn render(
        &mut self,
        backend: &mut WgpuBackend,
        swapchain_view: TextureViewHandle,
        screen_width: u32,
        screen_height: u32,
    ) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [screen_width, screen_height],
            pixels_per_point: self.ctx.pixels_per_point(),
        };

        let (device, queue, encoder) = backend.device_queue_encoder();

        for (id, image_delta) in &self.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image_delta);
        }

        if let Some(encoder) = encoder {
            self.renderer.update_buffers(
                device,
                queue,
                encoder,
                &self.paint_jobs,
                &screen_descriptor,
            );
        }

        backend.render_egui(
            &self.renderer,
            &self.paint_jobs,
            &screen_descriptor,
            swapchain_view,
        );

        for id in &self.textures_delta.free {
            self.renderer.free_texture(id);
        }

        self.textures_delta = egui::TexturesDelta::default();
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
