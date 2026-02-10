//! Egui UI for the physics demo — demo selector, physics controls.

use redlilium_graphics::egui::{EguiApp, egui};

/// Whether the active scene is 3D or 2D.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    ThreeD,
    TwoD,
}

/// Shared UI state between the demo app and the egui panel.
pub struct PhysicsUi {
    pub visible: bool,
    pub paused: bool,
    pub active_dim: Dimension,
    pub active_index: usize,
    pub scene_names_3d: Vec<String>,
    pub scene_names_2d: Vec<String>,
    pub gravity_scale: f32,
    pub time_scale: f32,
    pub body_count: usize,
    pub collider_count: usize,

    // Signals consumed by the app each frame
    scene_changed: bool,
    reset_requested: bool,
}

impl PhysicsUi {
    pub fn new() -> Self {
        Self {
            visible: true,
            paused: false,
            active_dim: Dimension::ThreeD,
            active_index: 0,
            scene_names_3d: Vec::new(),
            scene_names_2d: Vec::new(),
            gravity_scale: 1.0,
            time_scale: 1.0,
            body_count: 0,
            collider_count: 0,
            scene_changed: false,
            reset_requested: false,
        }
    }

    pub fn take_scene_changed(&mut self) -> bool {
        std::mem::take(&mut self.scene_changed)
    }

    pub fn take_reset_requested(&mut self) -> bool {
        std::mem::take(&mut self.reset_requested)
    }

    pub fn toggle_visibility(&mut self) {
        self.visible = !self.visible;
    }
}

impl EguiApp for PhysicsUi {
    fn setup(&mut self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals.window_corner_radius = egui::CornerRadius::same(8);
        style.spacing.slider_width = 180.0;
        ctx.set_style(style);
    }

    fn update(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }

        egui::Window::new("Physics Demo")
            .default_pos([10.0, 10.0])
            .default_width(240.0)
            .resizable(false)
            .show(ctx, |ui| {
                // ---- Dimension tabs ----
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.active_dim == Dimension::ThreeD, "3D")
                        .clicked()
                        && self.active_dim != Dimension::ThreeD
                    {
                        self.active_dim = Dimension::ThreeD;
                        self.active_index = 0;
                        self.scene_changed = true;
                    }
                    if ui
                        .selectable_label(self.active_dim == Dimension::TwoD, "2D")
                        .clicked()
                        && self.active_dim != Dimension::TwoD
                    {
                        self.active_dim = Dimension::TwoD;
                        self.active_index = 0;
                        self.scene_changed = true;
                    }
                });

                ui.separator();

                // ---- Scene list ----
                let names = match self.active_dim {
                    Dimension::ThreeD => &self.scene_names_3d,
                    Dimension::TwoD => &self.scene_names_2d,
                };
                for (i, name) in names.iter().enumerate() {
                    if ui
                        .selectable_label(i == self.active_index, name.as_str())
                        .clicked()
                        && i != self.active_index
                    {
                        self.active_index = i;
                        self.scene_changed = true;
                    }
                }

                ui.separator();

                // ---- Controls ----
                ui.heading("Controls");

                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "Resume" } else { "Pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                    }
                    if ui.button("Reset").clicked() {
                        self.reset_requested = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Gravity:");
                    ui.add(egui::Slider::new(&mut self.gravity_scale, 0.0..=3.0).step_by(0.1));
                });

                ui.horizontal(|ui| {
                    ui.label("Speed:");
                    ui.add(egui::Slider::new(&mut self.time_scale, 0.0..=3.0).step_by(0.1));
                });

                ui.separator();

                // ---- Stats ----
                ui.label(format!("Bodies: {}", self.body_count));
                ui.label(format!("Colliders: {}", self.collider_count));

                ui.separator();
                ui.small("LMB: Orbit | Scroll: Zoom | H: Toggle UI | Space: Pause");
            });
    }
}
