//! Window management using winit

use std::sync::Arc;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::{Window as WinitWindow, WindowBuilder},
};

/// Wrapper around winit window with additional state
pub struct Window {
    window: Arc<WinitWindow>,
    width: u32,
    height: u32,
    resized: bool,
    close_requested: bool,
}

impl Window {
    /// Create a new window with the given title and dimensions
    pub fn new(event_loop: &EventLoop<()>, title: &str, width: u32, height: u32) -> Self {
        let window = Arc::new(
            WindowBuilder::new()
                .with_title(title)
                .with_inner_size(PhysicalSize::new(width, height))
                .build(event_loop)
                .expect("Failed to create window"),
        );

        Self {
            window,
            width,
            height,
            resized: false,
            close_requested: false,
        }
    }

    /// Get the raw window for backend initialization
    pub fn window(&self) -> &WinitWindow {
        &self.window
    }

    /// Get arc reference to window
    pub fn window_arc(&self) -> Arc<WinitWindow> {
        Arc::clone(&self.window)
    }

    /// Get current window dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if window was resized since last frame
    pub fn was_resized(&self) -> bool {
        self.resized
    }

    /// Clear the resize flag
    pub fn clear_resize_flag(&mut self) {
        self.resized = false;
    }

    /// Check if close was requested
    pub fn should_close(&self) -> bool {
        self.close_requested
    }

    /// Handle window events
    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                self.width = size.width;
                self.height = size.height;
                self.resized = true;
            }
            WindowEvent::CloseRequested => {
                self.close_requested = true;
            }
            _ => {}
        }
    }

    /// Request a redraw
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

/// Run the application with a callback
pub fn run<F>(title: &str, width: u32, height: u32, mut callback: F)
where
    F: FnMut(&mut Window) + 'static,
{
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut window = Window::new(&event_loop, title, width, height);

    event_loop
        .run(move |event, elwt: &EventLoopWindowTarget<()>| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => {
                    window.handle_event(&event);

                    if let WindowEvent::CloseRequested = event {
                        elwt.exit();
                    }
                }
                Event::AboutToWait => {
                    callback(&mut window);
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("Event loop failed");
}
