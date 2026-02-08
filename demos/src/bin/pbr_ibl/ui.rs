//! Egui UI implementation for the PBR demo.

use redlilium_graphics::egui::{EguiApp, egui};

use crate::{GRID_SIZE, SPHERE_SPACING};

/// UI state shared between the demo and egui.
#[derive(Clone)]
pub struct PbrUiState {
    /// Base color RGB (0.0-1.0)
    pub base_color: [f32; 3],
    /// Skybox MIP level (0-7)
    pub skybox_mip_level: f32,
    /// Camera auto-rotation enabled
    pub auto_rotate: bool,
    /// Camera zoom level
    pub camera_distance: f32,
    /// Whether the UI is visible
    pub ui_visible: bool,
    /// Grid spacing
    pub sphere_spacing: f32,
    /// Whether to show the info panel
    pub show_info: bool,
    /// Whether to render wireframe
    pub show_wireframe: bool,
    /// Whether to show the G-buffer preview panel
    pub show_gbuffer: bool,
    /// Texture IDs for G-buffer visualization
    pub gbuffer_albedo_id: Option<egui::TextureId>,
    pub gbuffer_normal_id: Option<egui::TextureId>,
    pub gbuffer_position_id: Option<egui::TextureId>,
}

impl Default for PbrUiState {
    fn default() -> Self {
        Self {
            base_color: [0.9, 0.1, 0.1],
            skybox_mip_level: 0.0,
            auto_rotate: true,
            camera_distance: 8.0,
            ui_visible: true,
            sphere_spacing: SPHERE_SPACING,
            show_wireframe: false,
            show_info: true,
            show_gbuffer: true,
            gbuffer_albedo_id: None,
            gbuffer_normal_id: None,
            gbuffer_position_id: None,
        }
    }
}

/// Egui application for the PBR demo UI.
pub struct PbrUi {
    state: PbrUiState,
    state_changed: bool,
}

impl Default for PbrUi {
    fn default() -> Self {
        Self::new()
    }
}

impl PbrUi {
    pub fn new() -> Self {
        Self {
            state: PbrUiState::default(),
            state_changed: true,
        }
    }

    pub fn state(&self) -> &PbrUiState {
        &self.state
    }

    pub fn take_state_changed(&mut self) -> bool {
        let changed = self.state_changed;
        self.state_changed = false;
        changed
    }

    pub fn set_camera_distance(&mut self, distance: f32) {
        self.state.camera_distance = distance;
    }

    pub fn toggle_visibility(&mut self) {
        self.state.ui_visible = !self.state.ui_visible;
    }

    pub fn set_gbuffer_texture_ids(
        &mut self,
        albedo: Option<egui::TextureId>,
        normal: Option<egui::TextureId>,
        position: Option<egui::TextureId>,
    ) {
        self.state.gbuffer_albedo_id = albedo;
        self.state.gbuffer_normal_id = normal;
        self.state.gbuffer_position_id = position;
    }
}

impl EguiApp for PbrUi {
    fn setup(&mut self, ctx: &egui::Context) {
        // Configure egui style
        let mut style = (*ctx.style()).clone();
        style.visuals.window_corner_radius = egui::CornerRadius::same(8);
        style.spacing.slider_width = 200.0;
        ctx.set_style(style);
    }

    fn update(&mut self, ctx: &egui::Context) {
        if !self.state.ui_visible {
            return;
        }

        egui::Window::new("PBR Controls")
            .default_pos([10.0, 10.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Material");
                ui.separator();

                // Base color picker
                ui.horizontal(|ui| {
                    ui.label("Base Color:");
                    if ui
                        .color_edit_button_rgb(&mut self.state.base_color)
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Environment");
                ui.separator();

                // Skybox mip level slider
                ui.horizontal(|ui| {
                    ui.label("Skybox Blur:");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.state.skybox_mip_level, 0.0..=7.0)
                                .step_by(0.5),
                        )
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Camera");
                ui.separator();

                // Auto-rotate checkbox
                if ui
                    .checkbox(&mut self.state.auto_rotate, "Auto Rotate")
                    .changed()
                {
                    self.state_changed = true;
                }

                // Camera distance slider
                ui.horizontal(|ui| {
                    ui.label("Distance:");
                    if ui
                        .add(egui::Slider::new(
                            &mut self.state.camera_distance,
                            2.0..=20.0,
                        ))
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Grid");
                ui.separator();

                // Sphere spacing slider
                ui.horizontal(|ui| {
                    ui.label("Spacing:");
                    if ui
                        .add(egui::Slider::new(&mut self.state.sphere_spacing, 1.0..=3.0))
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Debug");
                ui.separator();

                // Wireframe checkbox
                if ui
                    .checkbox(&mut self.state.show_wireframe, "Wireframe")
                    .changed()
                {
                    self.state_changed = true;
                }

                // Show info checkbox
                ui.checkbox(&mut self.state.show_info, "Show Info Panel");

                // Show G-buffer checkbox
                ui.checkbox(&mut self.state.show_gbuffer, "Show G-Buffer");
            });

        // Info panel
        if self.state.show_info {
            egui::Window::new("Info")
                .default_pos([10.0, 400.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Deferred PBR IBL Demo");
                    ui.separator();
                    ui.label(format!("Grid: {}x{} spheres", GRID_SIZE, GRID_SIZE));
                    ui.label("Rows: Roughness (top=smooth)");
                    ui.label("Cols: Metallic (left=dielectric)");
                    ui.separator();
                    ui.label("Deferred Rendering Pipeline:");
                    ui.label("  1. G-Buffer Pass (MRT)");
                    ui.label("  2. Skybox Pass");
                    ui.label("  3. Resolve/Lighting Pass");
                    ui.separator();
                    ui.label("Controls:");
                    ui.label("  H: Toggle UI");
                    ui.label("  LMB Drag: Rotate camera");
                    ui.label("  Scroll: Zoom");
                });
        }

        // G-buffer preview window showing all render targets
        if self.state.show_gbuffer {
            egui::Window::new("G-Buffer")
                .default_pos([ctx.available_rect().right() - 420.0, 10.0])
                .default_width(400.0)
                .resizable(true)
                .show(ctx, |ui| {
                    let preview_size = egui::vec2(120.0, 90.0);

                    ui.horizontal(|ui| {
                        // Albedo
                        ui.vertical(|ui| {
                            ui.label("Albedo");
                            if let Some(id) = self.state.gbuffer_albedo_id {
                                ui.image(egui::load::SizedTexture::new(id, preview_size));
                            } else {
                                ui.label("(not available)");
                            }
                        });

                        // Normal + Metallic
                        ui.vertical(|ui| {
                            ui.label("Normal+Metal");
                            if let Some(id) = self.state.gbuffer_normal_id {
                                ui.image(egui::load::SizedTexture::new(id, preview_size));
                            } else {
                                ui.label("(not available)");
                            }
                        });

                        // Position + Roughness
                        ui.vertical(|ui| {
                            ui.label("Pos+Rough");
                            if let Some(id) = self.state.gbuffer_position_id {
                                ui.image(egui::load::SizedTexture::new(id, preview_size));
                            } else {
                                ui.label("(not available)");
                            }
                        });
                    });
                });
        }
    }
}
