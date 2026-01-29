//! Web-specific functionality for running the graphics engine in a browser.
//!
//! This module provides utilities for initializing and running the engine
//! in a WebAssembly environment using WebGPU.

use wasm_bindgen::prelude::*;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::platform::web::WindowExtWebSys;
use winit::window::WindowBuilder;

/// Get the browser window dimensions
pub fn get_window_size() -> (u32, u32) {
    let window = web_sys::window().expect("no global window exists");
    let width = window.inner_width().unwrap().as_f64().unwrap() as u32;
    let height = window.inner_height().unwrap().as_f64().unwrap() as u32;
    (width.max(100), height.max(100))
}

/// Set up the canvas element for rendering
pub fn setup_canvas(window: &winit::window::Window, canvas_id: &str) -> web_sys::HtmlCanvasElement {
    let canvas = window.canvas().expect("Couldn't get canvas from window");

    // Get the document and find the container
    let web_window = web_sys::window().expect("no global window exists");
    let document = web_window.document().expect("no document exists");

    // Try to find existing container, or use body
    let container = document
        .get_element_by_id(canvas_id)
        .unwrap_or_else(|| document.body().unwrap().into());

    // Append canvas to container
    container
        .append_child(&canvas)
        .expect("Couldn't append canvas to container");

    // Get device pixel ratio for proper scaling
    let dpr = web_window.device_pixel_ratio();

    // Get container dimensions
    let container_width = web_window.inner_width().unwrap().as_f64().unwrap();
    let container_height = web_window.inner_height().unwrap().as_f64().unwrap();

    // Set canvas actual size (resolution)
    let canvas_width = (container_width * dpr) as u32;
    let canvas_height = (container_height * dpr) as u32;
    canvas.set_width(canvas_width);
    canvas.set_height(canvas_height);

    // Set canvas CSS size (display size)
    let style = canvas.style();
    style.set_property("width", &format!("{}px", container_width as u32)).unwrap();
    style.set_property("height", &format!("{}px", container_height as u32)).unwrap();
    style.set_property("display", "block").unwrap();

    console_log(&format!(
        "Canvas setup: {}x{} (CSS: {}x{}, DPR: {})",
        canvas_width, canvas_height,
        container_width as u32, container_height as u32,
        dpr
    ));

    canvas
}

/// Create a winit window configured for web
pub fn create_web_window(event_loop: &EventLoop<()>, title: &str) -> winit::window::Window {
    let (width, height) = get_window_size();

    WindowBuilder::new()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(width, height))
        .build(event_loop)
        .expect("Failed to create window")
}

/// Spawn a future on the browser's event loop
pub fn spawn_local<F>(future: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn console_log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

/// Log an error to the browser console
#[wasm_bindgen]
pub fn console_error(msg: &str) {
    web_sys::console::error_1(&msg.into());
}
