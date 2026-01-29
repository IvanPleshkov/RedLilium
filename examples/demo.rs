//! Demo application showcasing the graphics engine
//!
//! Run with:
//!   cargo run --example demo
//!   cargo run --example demo -- --backend vulkan
//!
//! Controls:
//!   WASD     - Move camera
//!   QE       - Move up/down
//!   Shift    - Sprint (2x speed)
//!   Mouse    - Look around (hold right mouse button)
//!   Scroll   - Adjust speed (FreeFly) or zoom (Orbit)
//!   Tab      - Switch camera mode (FreeFly / Orbit)
//!   F1       - Toggle debug UI (wgpu only)
//!   Escape   - Exit

use glam::Vec3;
use egui;
use graphics_engine::{
    backend::GraphicsBackend,
    resources::{Material, Mesh},
    scene::{
        Camera, CameraController, CameraInput, FreeFlyController, MainCamera, MeshRenderer,
        OrbitController, PointLight, DirectionalLight, Transform,
    },
    BackendType, WgpuEguiIntegration, VulkanEguiIntegration, Engine, EngineConfig,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

/// egui integration wrapper for either backend
enum EguiBackend {
    Wgpu(WgpuEguiIntegration),
    Vulkan(VulkanEguiIntegration),
}

/// Application state for input handling
struct AppState {
    camera_input: CameraInput,
    free_fly: FreeFlyController,
    orbit: OrbitController,
    active_controller: usize, // 0 = FreeFly, 1 = Orbit
    last_frame: Instant,
    cursor_grabbed: bool,
    /// egui integration
    egui: Option<EguiBackend>,
    /// Whether debug UI is visible
    show_debug_ui: bool,
    /// Frame time history for averaging
    frame_times: VecDeque<f32>,
    /// Current FPS (averaged)
    fps: f32,
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
            last_frame: Instant::now(),
            cursor_grabbed: false,
            egui: None,
            show_debug_ui: true,
            frame_times: VecDeque::with_capacity(60),
            fps: 0.0,
            // UI showcase defaults
            ui_slider_value: 0.5,
            ui_checkbox: false,
            ui_radio_selection: 0,
            ui_text_input: String::from("Hello egui!"),
            ui_color: [0.2, 0.6, 1.0],
            ui_dropdown_selection: 0,
            ui_click_count: 0,
        }
    }

    fn update_fps(&mut self, dt: f32) {
        // Keep last 60 frame times for averaging
        if self.frame_times.len() >= 60 {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);

        // Calculate average FPS
        if !self.frame_times.is_empty() {
            let avg_dt: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
            self.fps = 1.0 / avg_dt;
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
        println!("Camera mode: {}", self.active_controller_name());
    }

    fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        match self.active_controller {
            0 => self.free_fly.update(camera, &self.camera_input, dt),
            1 => self.orbit.update(camera, &self.camera_input, dt),
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    // Parse command line args for backend selection
    let args: Vec<String> = std::env::args().collect();
    let backend_type = if args.len() > 2 && args[1] == "--backend" {
        match args[2].to_lowercase().as_str() {
            "vulkan" | "vk" => BackendType::Vulkan,
            _ => BackendType::Wgpu,
        }
    } else {
        BackendType::Wgpu
    };

    println!("Starting Graphics Engine Demo");
    println!("Backend: {:?}", backend_type);
    println!();
    println!("Controls:");
    println!("  WASD       - Move camera");
    println!("  Q/E        - Move up/down");
    println!("  Shift      - Sprint (2x speed)");
    println!("  Right Mouse - Look around");
    println!("  Scroll     - Adjust speed/zoom");
    println!("  Tab        - Switch camera mode");
    println!("  F1         - Toggle debug UI");
    println!("  Escape     - Exit");
    println!();

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Graphics Engine Demo - FreeFly Camera")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .build(&event_loop)
            .expect("Failed to create window"),
    );

    // Create engine
    let config = EngineConfig {
        title: "Graphics Engine Demo".to_string(),
        width: 1280,
        height: 720,
        backend: backend_type,
        vsync: true,
        tile_size: 16,
        max_lights: 1024,
    };

    let mut engine = match Engine::new(Arc::clone(&window), config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create engine: {:?}", e);
            return;
        }
    };

    // Setup scene
    setup_scene(&mut engine);

    // Initialize app state
    let mut state = AppState::new();

    // Initialize egui for the selected backend
    match backend_type {
        BackendType::Wgpu => {
            if let Some(wgpu_backend) = engine.backend().as_wgpu() {
                state.egui = Some(EguiBackend::Wgpu(WgpuEguiIntegration::new(wgpu_backend, &window)));
                println!("egui debug UI initialized (wgpu backend, press F1 to toggle)");
            }
        }
        BackendType::Vulkan => {
            if let Some(vulkan_backend) = engine.backend().as_vulkan() {
                state.egui = Some(EguiBackend::Vulkan(VulkanEguiIntegration::new(vulkan_backend, &window)));
                println!("egui debug UI initialized (Vulkan backend, press F1 to toggle)");
            }
        }
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

    println!("Scene setup complete:");
    println!("  Objects: {}", object_count);
    println!("  Lights: {}", light_count);
    println!("  Camera mode: {}", state.active_controller_name());
    println!();

    // Run event loop
    let window_clone = Arc::clone(&window);
    event_loop
        .run(move |event, elwt: &EventLoopWindowTarget<()>| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => {
                    // Pass events to egui first
                    let egui_consumed = match &mut state.egui {
                        Some(EguiBackend::Wgpu(egui)) => egui.on_window_event(&window_clone, &event),
                        Some(EguiBackend::Vulkan(egui)) => egui.on_window_event(&window_clone, &event),
                        None => false,
                    };

                    // Only handle event if egui didn't consume it
                    if !egui_consumed {
                        handle_window_event(
                            &event,
                            &mut state,
                            &mut engine,
                            &window_clone,
                            elwt,
                        );
                    } else {
                        // Still need to handle certain events even if egui consumed them
                        match &event {
                            WindowEvent::CloseRequested => elwt.exit(),
                            WindowEvent::Resized(size) => engine.resize(size.width, size.height),
                            WindowEvent::RedrawRequested => {
                                render_frame(&mut engine, &mut state, &window_clone);
                            }
                            _ => {}
                        }
                    }
                }
                Event::DeviceEvent { event, .. } => {
                    // Don't process mouse motion if egui wants pointer input
                    let egui_wants_pointer = match &state.egui {
                        Some(EguiBackend::Wgpu(egui)) => egui.wants_pointer_input(),
                        Some(EguiBackend::Vulkan(egui)) => egui.wants_pointer_input(),
                        None => false,
                    };

                    if !egui_wants_pointer {
                        handle_device_event(&event, &mut state);
                    }
                }
                Event::LoopExiting => {
                    // IMPORTANT: Destroy egui GPU resources before the engine is dropped
                    // The egui integration holds a clone of the Vulkan device handle,
                    // which becomes invalid when VulkanBackend is dropped
                    if let Some(EguiBackend::Vulkan(ref mut egui)) = state.egui {
                        if let Some(vulkan_backend) = engine.backend().as_vulkan() {
                            egui.destroy(vulkan_backend);
                        }
                    }
                    state.egui = None;
                }
                Event::AboutToWait => {
                    // Calculate delta time
                    let now = Instant::now();
                    let dt = (now - state.last_frame).as_secs_f32();
                    state.last_frame = now;

                    // Update FPS counter
                    state.update_fps(dt);

                    // Update camera (skip if egui wants keyboard)
                    let egui_wants_keyboard = match &state.egui {
                        Some(EguiBackend::Wgpu(egui)) => egui.wants_keyboard_input(),
                        Some(EguiBackend::Vulkan(egui)) => egui.wants_keyboard_input(),
                        None => false,
                    };

                    if !egui_wants_keyboard {
                        // Get mutable camera from ECS
                        let mut query = engine.world_mut().query::<(&mut Camera, &MainCamera)>();
                        for (mut camera, _) in query.iter_mut(engine.world_mut()) {
                            state.update_camera(&mut camera, dt);
                        }
                    }

                    // Reset per-frame input deltas
                    state.camera_input.reset_deltas();

                    // Request redraw
                    window_clone.request_redraw();
                }
                _ => {}
            }
        })
        .expect("Event loop failed");
}

/// Build egui UI content (shared between backends)
fn build_egui_ui(
    ctx: &egui::Context,
    fps: f32,
    controller_name: &str,
    object_count: usize,
    light_count: usize,
    cam_pos: Vec3,
    slider_value: &mut f32,
    checkbox: &mut bool,
    radio_selection: &mut usize,
    text_input: &mut String,
    color: &mut [f32; 3],
    dropdown_selection: &mut usize,
    click_count: &mut u32,
) {
    // Build debug UI
    egui::Window::new("Debug")
        .default_pos([10.0, 10.0])
        .default_size([220.0, 300.0])
        .show(ctx, |ui| {
            // Performance section
            ui.heading("Performance");
            ui.label(format!("FPS: {:.1}", fps));
            ui.label(format!(
                "Frame time: {:.2} ms",
                if fps > 0.0 { 1000.0 / fps } else { 0.0 }
            ));
            ui.separator();

            // Scene info
            ui.heading("Scene");
            ui.label(format!("Objects: {}", object_count));
            ui.label(format!("Lights: {}", light_count));
            ui.separator();

            // Camera info
            ui.heading("Camera");
            ui.label(format!("Mode: {}", controller_name));
            ui.label(format!(
                "Position: ({:.1}, {:.1}, {:.1})",
                cam_pos.x, cam_pos.y, cam_pos.z
            ));
            ui.separator();

            // Controls hint
            ui.heading("Controls");
            ui.label("WASD - Move");
            ui.label("Q/E - Up/Down");
            ui.label("RMB + Mouse - Look");
            ui.label("Tab - Switch camera");
            ui.label("F1 - Toggle this UI");
        });

    // UI Elements Showcase window
    egui::Window::new("UI Showcase")
        .default_pos([240.0, 10.0])
        .default_size([250.0, 400.0])
        .show(ctx, |ui| {
            ui.heading("Basic Elements");

            // Button
            ui.horizontal(|ui| {
                if ui.button("Click me!").clicked() {
                    *click_count += 1;
                }
                ui.label(format!("Clicked: {} times", *click_count));
            });

            ui.separator();

            // Slider
            ui.label("Slider:");
            ui.add(egui::Slider::new(slider_value, 0.0..=1.0).text("value"));

            // Progress bar using slider value
            ui.label("Progress bar:");
            ui.add(egui::ProgressBar::new(*slider_value).show_percentage());

            ui.separator();

            // Checkbox
            ui.checkbox(checkbox, "Enable feature");

            ui.separator();

            // Radio buttons
            ui.label("Radio buttons:");
            ui.horizontal(|ui| {
                ui.radio_value(radio_selection, 0, "Option A");
                ui.radio_value(radio_selection, 1, "Option B");
                ui.radio_value(radio_selection, 2, "Option C");
            });

            ui.separator();

            // Text input
            ui.label("Text input:");
            ui.text_edit_singleline(text_input);

            ui.separator();

            // Color picker
            ui.label("Color picker:");
            ui.color_edit_button_rgb(color);
            ui.horizontal(|ui| {
                ui.label("Preview:");
                let color32 = egui::Color32::from_rgb(
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                );
                ui.colored_label(color32, "Sample Text");
            });

            ui.separator();

            // Dropdown (ComboBox)
            ui.label("Dropdown:");
            let options = ["First", "Second", "Third", "Fourth"];
            egui::ComboBox::from_label("Select")
                .selected_text(options[*dropdown_selection])
                .show_ui(ui, |ui| {
                    for (i, option) in options.iter().enumerate() {
                        ui.selectable_value(dropdown_selection, i, *option);
                    }
                });

            ui.separator();

            // Collapsing header
            ui.collapsing("Collapsible Section", |ui| {
                ui.label("This content is hidden by default.");
                ui.label("Click the header to expand/collapse.");
                ui.horizontal(|ui| {
                    ui.label("Nested content:");
                    ui.monospace("code style");
                });
            });

            ui.separator();

            // Hyperlink
            ui.hyperlink_to("egui documentation", "https://docs.rs/egui");
        });
}

/// Render a frame with egui overlay
fn render_frame(engine: &mut Engine, state: &mut AppState, window: &winit::window::Window) {
    // Begin egui frame if available and visible
    if state.show_debug_ui && state.egui.is_some() {
        // Extract data needed for UI before borrowing egui mutably
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

        // Extract mutable showcase state
        let mut slider_value = state.ui_slider_value;
        let mut checkbox = state.ui_checkbox;
        let mut radio_selection = state.ui_radio_selection;
        let mut text_input = state.ui_text_input.clone();
        let mut color = state.ui_color;
        let mut dropdown_selection = state.ui_dropdown_selection;
        let mut click_count = state.ui_click_count;

        // Begin frame, build UI, end frame
        match &mut state.egui {
            Some(EguiBackend::Wgpu(egui)) => {
                egui.begin_frame(window);
                build_egui_ui(
                    egui.context(), fps, controller_name, object_count, light_count, cam_pos,
                    &mut slider_value, &mut checkbox, &mut radio_selection, &mut text_input,
                    &mut color, &mut dropdown_selection, &mut click_count,
                );
                egui.end_frame(window);
            }
            Some(EguiBackend::Vulkan(egui)) => {
                egui.begin_frame(window);
                build_egui_ui(
                    egui.context(), fps, controller_name, object_count, light_count, cam_pos,
                    &mut slider_value, &mut checkbox, &mut radio_selection, &mut text_input,
                    &mut color, &mut dropdown_selection, &mut click_count,
                );
                egui.end_frame(window);
            }
            None => {}
        }

        // Write back modified showcase state
        state.ui_slider_value = slider_value;
        state.ui_checkbox = checkbox;
        state.ui_radio_selection = radio_selection;
        state.ui_text_input = text_input;
        state.ui_color = color;
        state.ui_dropdown_selection = dropdown_selection;
        state.ui_click_count = click_count;
    }

    // Render main scene (without presenting)
    if let Err(e) = engine.render_scene() {
        eprintln!("Render error: {:?}", e);
        return;
    }

    // Render egui overlay (before presenting)
    if state.show_debug_ui {
        match &mut state.egui {
            Some(EguiBackend::Wgpu(egui)) => {
                if let Some(wgpu_backend) = engine.backend_mut().as_wgpu_mut() {
                    // Use actual surface size (may be clamped by device limits)
                    let (width, height) = wgpu_backend.surface_size();
                    if let Some(swapchain_view) = wgpu_backend.current_swapchain_view() {
                        egui.render(wgpu_backend, swapchain_view, width, height);
                    }
                }
            }
            Some(EguiBackend::Vulkan(egui)) => {
                if let Some(vulkan_backend) = engine.backend_mut().as_vulkan_mut() {
                    let (width, height) = vulkan_backend.surface_size();
                    let command_buffer = vulkan_backend.command_buffer();
                    let swapchain_view = vulkan_backend.current_swapchain_image_view();
                    unsafe {
                        egui.render(vulkan_backend, command_buffer, swapchain_view, width, height);
                    }
                }
            }
            None => {}
        }
    }

    // Present the frame
    if let Err(e) = engine.end_frame() {
        eprintln!("Present error: {:?}", e);
    }
}


fn handle_window_event(
    event: &WindowEvent,
    state: &mut AppState,
    engine: &mut Engine,
    window: &winit::window::Window,
    elwt: &EventLoopWindowTarget<()>,
) {
    match event {
        WindowEvent::CloseRequested => {
            println!("Close requested, shutting down...");
            elwt.exit();
        }
        WindowEvent::Resized(size) => {
            engine.resize(size.width, size.height);
        }
        WindowEvent::RedrawRequested => {
            render_frame(engine, state, window);
        }
        WindowEvent::KeyboardInput { event, .. } => {
            let pressed = event.state == ElementState::Pressed;

            if let PhysicalKey::Code(key) = event.physical_key {
                match key {
                    KeyCode::Escape => {
                        elwt.exit();
                    }
                    KeyCode::Tab if pressed && !event.repeat => {
                        state.switch_controller();
                        let title = format!(
                            "Graphics Engine Demo - {} Camera",
                            state.active_controller_name()
                        );
                        window.set_title(&title);
                    }
                    KeyCode::F1 if pressed && !event.repeat => {
                        state.show_debug_ui = !state.show_debug_ui;
                        println!(
                            "Debug UI: {}",
                            if state.show_debug_ui { "visible" } else { "hidden" }
                        );
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

                // Grab/release cursor
                if pressed && !state.cursor_grabbed {
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    window.set_cursor_visible(false);
                    state.cursor_grabbed = true;
                } else if !pressed && state.cursor_grabbed {
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    state.cursor_grabbed = false;
                }
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let scroll = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
            };
            state.camera_input.scroll_delta += scroll;
        }
        WindowEvent::Focused(false) => {
            // Release all keys when window loses focus
            state.camera_input = CameraInput::new();
            if state.cursor_grabbed {
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                window.set_cursor_visible(true);
                state.cursor_grabbed = false;
            }
        }
        _ => {}
    }
}

fn handle_device_event(event: &DeviceEvent, state: &mut AppState) {
    if let DeviceEvent::MouseMotion { delta } = event {
        if state.camera_input.mouse_look_active {
            state.camera_input.mouse_delta.x += delta.0 as f32;
            state.camera_input.mouse_delta.y += delta.1 as f32;
        }
    }
}

fn setup_scene(engine: &mut Engine) {
    // Add meshes
    let cube_id = engine.add_mesh(Mesh::cube());
    let sphere_id = engine.add_mesh(Mesh::sphere(32, 16));
    let plane_id = engine.add_mesh(Mesh::plane(20.0, 20.0, 10));

    // Add materials
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
