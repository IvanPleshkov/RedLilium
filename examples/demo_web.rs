//! Web demo application for the graphics engine
//!
//! This is the WebAssembly entry point for the demo.
//! Build with: cargo build --example demo_web --target wasm32-unknown-unknown --no-default-features --features web

#![cfg(target_arch = "wasm32")]

use glam::Vec3;
use graphics_engine::{
    init_web_logging,
    resources::{Material, Mesh},
    scene::{
        Camera, CameraController, CameraInput, DirectionalLight, FreeFlyController, MainCamera,
        MeshRenderer, OrbitController, PointLight, Transform,
    },
    web::{console_log, setup_canvas, spawn_local},
    BackendType, EguiIntegration, Engine, EngineConfig,
};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    platform::web::EventLoopExtWebSys,
    window::WindowBuilder,
};

/// Application state for input handling
struct AppState {
    camera_input: CameraInput,
    free_fly: FreeFlyController,
    orbit: OrbitController,
    active_controller: usize,
    cursor_grabbed: bool,
    egui: Option<EguiIntegration>,
    show_debug_ui: bool,
    frame_times: VecDeque<f32>,
    fps: f32,
    last_frame_time: f64,
    // UI showcase state
    ui_slider_value: f32,
    ui_checkbox: bool,
    ui_radio_selection: usize,
    ui_text_input: String,
    ui_color: [f32; 3],
    ui_dropdown_selection: usize,
    ui_click_count: u32,
}

impl AppState {
    fn new() -> Self {
        Self {
            camera_input: CameraInput::new(),
            free_fly: FreeFlyController::default().with_speed(8.0),
            orbit: OrbitController::new(Vec3::ZERO, 15.0).with_angles(45.0, 30.0),
            active_controller: 0,
            cursor_grabbed: false,
            egui: None,
            show_debug_ui: true,
            frame_times: VecDeque::with_capacity(60),
            fps: 0.0,
            last_frame_time: 0.0,
            ui_slider_value: 0.5,
            ui_checkbox: false,
            ui_radio_selection: 0,
            ui_text_input: String::from("Hello egui!"),
            ui_color: [0.2, 0.6, 1.0],
            ui_dropdown_selection: 0,
            ui_click_count: 0,
        }
    }

    fn active_controller_name(&self) -> &'static str {
        match self.active_controller {
            0 => "FreeFly",
            1 => "Orbit",
            _ => "Unknown",
        }
    }

    fn switch_controller(&mut self) {
        self.active_controller = (self.active_controller + 1) % 2;
        console_log(&format!("Camera mode: {}", self.active_controller_name()));
    }

    fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        match self.active_controller {
            0 => self.free_fly.update(camera, &self.camera_input, dt),
            1 => self.orbit.update(camera, &self.camera_input, dt),
            _ => {}
        }
    }

    fn update_fps(&mut self, dt: f32) {
        if self.frame_times.len() >= 60 {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);

        if !self.frame_times.is_empty() {
            let avg_dt: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
            self.fps = 1.0 / avg_dt;
        }
    }
}

fn setup_scene(engine: &mut Engine) {
    let cube_id = engine.add_mesh(Mesh::cube());
    let sphere_id = engine.add_mesh(Mesh::sphere(32, 16));
    let plane_id = engine.add_mesh(Mesh::plane(20.0, 20.0, 10));

    let gold_id = engine.add_material(Material::gold());
    let silver_id = engine.add_material(Material::silver());
    let plastic_red_id = engine.add_material(Material::plastic(Vec3::new(0.8, 0.2, 0.2)));
    let plastic_green_id = engine.add_material(Material::plastic(Vec3::new(0.2, 0.8, 0.2)));
    let floor_id = engine.add_material(
        Material::new("floor")
            .with_base_color(glam::Vec4::new(0.5, 0.5, 0.5, 1.0))
            .with_roughness(0.8),
    );

    // Setup camera (update the existing MainCamera entity)
    {
        let mut query = engine.world_mut().query::<(&mut Camera, &MainCamera)>();
        for (mut camera, _) in query.iter_mut(engine.world_mut()) {
            camera.position = Vec3::new(5.0, 5.0, 10.0);
            camera.look_at(Vec3::ZERO);
        }
    }

    // Add floor
    engine.world_mut().spawn((
        MeshRenderer::new(plane_id, floor_id),
        Transform::from_position(Vec3::new(0.0, -1.0, 0.0)),
    ));

    // Add cubes
    engine.world_mut().spawn((
        MeshRenderer::new(cube_id, gold_id),
        Transform::from_position(Vec3::new(-2.0, 0.0, 0.0)),
    ));

    engine.world_mut().spawn((
        MeshRenderer::new(cube_id, silver_id),
        Transform::from_position(Vec3::new(2.0, 0.0, 0.0)),
    ));

    // Add spheres
    engine.world_mut().spawn((
        MeshRenderer::new(sphere_id, plastic_red_id),
        Transform::from_position(Vec3::new(0.0, 0.5, 2.0)),
    ));

    engine.world_mut().spawn((
        MeshRenderer::new(sphere_id, plastic_green_id),
        Transform::from_position(Vec3::new(0.0, 0.5, -2.0)),
    ));

    // Add lights
    // Main directional light (sun)
    engine.world_mut().spawn(
        DirectionalLight::new(Vec3::new(-0.5, -1.0, -0.3), Vec3::new(1.0, 0.95, 0.9), 2.0),
    );

    // Point lights for local illumination
    engine.world_mut().spawn((
        Transform::from_position(Vec3::new(3.0, 2.0, 3.0)),
        PointLight::new(Vec3::new(1.0, 0.7, 0.4), 10.0, 15.0),
    ));
    engine.world_mut().spawn((
        Transform::from_position(Vec3::new(-3.0, 2.0, 3.0)),
        PointLight::new(Vec3::new(0.4, 0.7, 1.0), 10.0, 15.0),
    ));
    engine.world_mut().spawn((
        Transform::from_position(Vec3::new(0.0, 3.0, -3.0)),
        PointLight::new(Vec3::new(0.7, 1.0, 0.7), 8.0, 12.0),
    ));

    // Add more lights to demonstrate Forward+
    for i in 0..10 {
        let angle = (i as f32 / 10.0) * std::f32::consts::TAU;
        let radius = 6.0;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        let color = Vec3::new(
            (i as f32 * 0.1).sin().abs(),
            (i as f32 * 0.15 + 1.0).sin().abs(),
            (i as f32 * 0.2 + 2.0).sin().abs(),
        );
        engine.world_mut().spawn((
            Transform::from_position(Vec3::new(x, 0.5, z)),
            PointLight::new(color, 3.0, 5.0),
        ));
    }
}

/// Main entry point for web
#[wasm_bindgen(start)]
pub fn main() {
    init_web_logging();
    console_log("Starting Graphics Engine Demo (Web)");

    // Spawn the async main function
    spawn_local(async_main());
}

async fn async_main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Get window size from browser
    let web_window = web_sys::window().expect("no global window");
    let width = web_window.inner_width().unwrap().as_f64().unwrap() as u32;
    let height = web_window.inner_height().unwrap().as_f64().unwrap() as u32;

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Graphics Engine Demo - Web")
            .with_inner_size(PhysicalSize::new(width.max(800), height.max(600)))
            .build(&event_loop)
            .expect("Failed to create window"),
    );

    // Set up canvas in DOM
    let canvas = setup_canvas(&window, "canvas-container");

    // Get actual canvas dimensions after setup
    let canvas_width = canvas.width();
    let canvas_height = canvas.height();
    console_log(&format!("Creating engine with size: {}x{}", canvas_width, canvas_height));

    let config = EngineConfig {
        title: "Graphics Engine Demo".to_string(),
        width: canvas_width,
        height: canvas_height,
        backend: BackendType::Wgpu,
        vsync: true,
        tile_size: 16,
        max_lights: 1024,
    };

    // Create engine asynchronously
    console_log("Creating engine...");
    let mut engine = match Engine::new_async(Arc::clone(&window), config).await {
        Ok(e) => {
            console_log("Engine created successfully");
            e
        }
        Err(e) => {
            console_log(&format!("Failed to create engine: {:?}", e));
            return;
        }
    };

    // Ensure engine size matches canvas
    engine.resize(canvas_width, canvas_height);

    setup_scene(&mut engine);

    let mut state = AppState::new();

    // Initialize egui
    if let Some(wgpu_backend) = engine.backend().as_wgpu() {
        state.egui = Some(EguiIntegration::new(wgpu_backend, &window));
        console_log("egui debug UI initialized (press F1 to toggle)");
    }

    // Sync controllers with initial camera position
    {
        let mut query = engine.world_mut().query::<(&Camera, &MainCamera)>();
        if let Some((camera, _)) = query.iter(engine.world()).next() {
            state.free_fly.sync_with_camera(camera);
            state.orbit.sync_with_camera(camera);
        }
    }

    // Count objects and lights
    let object_count = engine.world_mut().query::<&MeshRenderer>().iter(engine.world()).count();
    let light_count = engine.world_mut().query::<&PointLight>().iter(engine.world()).count()
        + engine.world_mut().query::<&DirectionalLight>().iter(engine.world()).count();

    console_log(&format!(
        "Scene setup complete: {} objects, {} lights",
        object_count,
        light_count
    ));

    // Use Rc<RefCell> for shared mutable state in the web event loop
    let engine = Rc::new(RefCell::new(engine));
    let state = Rc::new(RefCell::new(state));
    let window = Rc::new(window);

    // Get initial timestamp
    let performance = web_sys::window().unwrap().performance().unwrap();

    // Run event loop (web-style, non-blocking)
    event_loop.spawn(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        let mut engine = engine.borrow_mut();
        let mut state = state.borrow_mut();

        match event {
            Event::WindowEvent { event, .. } => {
                // Pass events to egui first
                let egui_consumed = if let Some(ref mut egui) = state.egui {
                    egui.on_window_event(&window, &event)
                } else {
                    false
                };

                if !egui_consumed {
                    handle_window_event(&event, &mut state, &mut engine, &window, elwt);
                } else {
                    match &event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(size) => engine.resize(size.width, size.height),
                        WindowEvent::RedrawRequested => {
                            render_frame(&mut engine, &mut state, &window, &performance);
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                // Calculate delta time
                let now = performance.now();
                let dt = ((now - state.last_frame_time) / 1000.0) as f32;
                state.last_frame_time = now;

                if dt > 0.0 && dt < 1.0 {
                    state.update_fps(dt);

                    let egui_wants_keyboard = state.egui.as_ref()
                        .map(|e| e.wants_keyboard_input())
                        .unwrap_or(false);

                    if !egui_wants_keyboard {
                        // Get mutable camera from ECS
                        let mut query = engine.world_mut().query::<(&mut Camera, &MainCamera)>();
                        for (mut camera, _) in query.iter_mut(engine.world_mut()) {
                            state.update_camera(&mut camera, dt);
                        }
                    }
                }

                state.camera_input.reset_deltas();
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn handle_window_event(
    event: &WindowEvent,
    state: &mut AppState,
    engine: &mut Engine,
    window: &winit::window::Window,
    elwt: &winit::event_loop::ActiveEventLoop,
) {
    match event {
        WindowEvent::CloseRequested => {
            elwt.exit();
        }
        WindowEvent::Resized(size) => {
            engine.resize(size.width, size.height);
        }
        WindowEvent::RedrawRequested => {
            let performance = web_sys::window().unwrap().performance().unwrap();
            render_frame(engine, state, window, &performance);
        }
        WindowEvent::KeyboardInput { event, .. } => {
            let pressed = event.state == ElementState::Pressed;

            if let PhysicalKey::Code(key) = event.physical_key {
                match key {
                    KeyCode::Tab if pressed && !event.repeat => {
                        state.switch_controller();
                    }
                    KeyCode::F1 if pressed && !event.repeat => {
                        state.show_debug_ui = !state.show_debug_ui;
                    }
                    KeyCode::KeyW => state.camera_input.forward = pressed,
                    KeyCode::KeyS => state.camera_input.backward = pressed,
                    KeyCode::KeyA => state.camera_input.left = pressed,
                    KeyCode::KeyD => state.camera_input.right = pressed,
                    KeyCode::KeyQ | KeyCode::ControlLeft => state.camera_input.down = pressed,
                    KeyCode::KeyE | KeyCode::Space => state.camera_input.up = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        state.camera_input.sprint = pressed
                    }
                    _ => {}
                }
            }
        }
        WindowEvent::MouseInput { state: btn_state, button, .. } => {
            if *button == MouseButton::Right {
                let pressed = *btn_state == ElementState::Pressed;
                state.camera_input.mouse_look_active = pressed;
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let scroll = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
            };
            state.camera_input.scroll_delta += scroll;
        }
        WindowEvent::CursorMoved { position, .. } => {
            // Track mouse movement for camera control on web
            static mut LAST_POS: Option<(f64, f64)> = None;
            unsafe {
                if state.camera_input.mouse_look_active {
                    if let Some((lx, ly)) = LAST_POS {
                        state.camera_input.mouse_delta.x += (position.x - lx) as f32;
                        state.camera_input.mouse_delta.y += (position.y - ly) as f32;
                    }
                }
                LAST_POS = Some((position.x, position.y));
            }
        }
        WindowEvent::Focused(false) => {
            state.camera_input = CameraInput::new();
        }
        _ => {}
    }
}

static mut FRAME_COUNT: u32 = 0;

fn render_frame(
    engine: &mut Engine,
    state: &mut AppState,
    window: &winit::window::Window,
    _performance: &web_sys::Performance,
) {
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT == 1 || FRAME_COUNT % 60 == 0 {
            console_log(&format!("render_frame called (frame {})", FRAME_COUNT));
        }
    }

    // Build egui UI
    if state.show_debug_ui && state.egui.is_some() {
        let fps = state.fps;
        let controller_name = state.active_controller_name();
        let object_count = engine.world_mut().query::<&MeshRenderer>().iter(engine.world()).count();
        let light_count = engine.world_mut().query::<&PointLight>().iter(engine.world()).count()
            + engine.world_mut().query::<&DirectionalLight>().iter(engine.world()).count();
        let cam_pos = {
            let mut query = engine.world_mut().query::<(&Camera, &MainCamera)>();
            query.iter(engine.world())
                .next()
                .map(|(c, _)| c.position)
                .unwrap_or(Vec3::ZERO)
        };

        let mut slider_value = state.ui_slider_value;
        let mut checkbox = state.ui_checkbox;
        let mut radio_selection = state.ui_radio_selection;
        let mut text_input = state.ui_text_input.clone();
        let mut color = state.ui_color;
        let mut dropdown_selection = state.ui_dropdown_selection;
        let mut click_count = state.ui_click_count;

        let egui = state.egui.as_mut().unwrap();
        egui.begin_frame(window);

        egui::Window::new("Debug")
            .default_pos([10.0, 10.0])
            .default_size([220.0, 300.0])
            .show(egui.context(), |ui| {
                ui.heading("Performance");
                ui.label(format!("FPS: {:.1}", fps));
                ui.label(format!("Frame time: {:.2} ms", if fps > 0.0 { 1000.0 / fps } else { 0.0 }));
                ui.separator();

                ui.heading("Scene");
                ui.label(format!("Objects: {}", object_count));
                ui.label(format!("Lights: {}", light_count));
                ui.separator();

                ui.heading("Camera");
                ui.label(format!("Mode: {}", controller_name));
                ui.label(format!("Position: ({:.1}, {:.1}, {:.1})", cam_pos.x, cam_pos.y, cam_pos.z));
                ui.separator();

                ui.heading("Controls");
                ui.label("WASD - Move");
                ui.label("Q/E - Up/Down");
                ui.label("RMB + Mouse - Look");
                ui.label("Tab - Switch camera");
                ui.label("F1 - Toggle this UI");
            });

        egui::Window::new("UI Showcase")
            .default_pos([240.0, 10.0])
            .default_size([250.0, 400.0])
            .show(egui.context(), |ui| {
                ui.heading("Basic Elements");

                ui.horizontal(|ui| {
                    if ui.button("Click me!").clicked() {
                        click_count += 1;
                    }
                    ui.label(format!("Clicked: {} times", click_count));
                });
                ui.separator();

                ui.label("Slider:");
                ui.add(egui::Slider::new(&mut slider_value, 0.0..=1.0).text("value"));
                ui.add(egui::ProgressBar::new(slider_value).show_percentage());
                ui.separator();

                ui.checkbox(&mut checkbox, "Enable feature");
                ui.separator();

                ui.label("Radio buttons:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut radio_selection, 0, "A");
                    ui.radio_value(&mut radio_selection, 1, "B");
                    ui.radio_value(&mut radio_selection, 2, "C");
                });
                ui.separator();

                ui.label("Text input:");
                ui.text_edit_singleline(&mut text_input);
                ui.separator();

                ui.label("Color picker:");
                ui.color_edit_button_rgb(&mut color);
                ui.separator();

                let options = ["First", "Second", "Third", "Fourth"];
                egui::ComboBox::from_label("Dropdown")
                    .selected_text(options[dropdown_selection])
                    .show_ui(ui, |ui| {
                        for (i, option) in options.iter().enumerate() {
                            ui.selectable_value(&mut dropdown_selection, i, *option);
                        }
                    });
            });

        egui.end_frame(window);

        state.ui_slider_value = slider_value;
        state.ui_checkbox = checkbox;
        state.ui_radio_selection = radio_selection;
        state.ui_text_input = text_input;
        state.ui_color = color;
        state.ui_dropdown_selection = dropdown_selection;
        state.ui_click_count = click_count;
    }

    // Render scene
    if let Err(e) = engine.render_scene() {
        console_log(&format!("Render error: {:?}", e));
        return;
    }

    // Render egui overlay
    if state.show_debug_ui {
        if let Some(ref mut egui) = state.egui {
            if let Some(wgpu_backend) = engine.backend_mut().as_wgpu_mut() {
                let size = window.inner_size();
                if let Some(swapchain_view) = wgpu_backend.current_swapchain_view() {
                    egui.render(wgpu_backend, swapchain_view, size.width, size.height);
                } else {
                    unsafe {
                        if FRAME_COUNT <= 5 {
                            console_log("Warning: No swapchain view available for egui");
                        }
                    }
                }
            }
        }
    }

    // Present
    if let Err(e) = engine.end_frame() {
        console_log(&format!("Present error: {:?}", e));
    }
}
