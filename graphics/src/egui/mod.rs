//! # Egui Integration
//!
//! This module provides integration with the [egui](https://github.com/emilk/egui) UI library.
//!
//! ## Overview
//!
//! - [`EguiApp`] - Trait for applications that use egui
//! - [`EguiController`] - Controller that manages egui context, resources, and rendering
//!
//! ## Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use redlilium_graphics::egui::{EguiApp, EguiController};
//!
//! struct MyUi {
//!     show_window: bool,
//!     value: f32,
//! }
//!
//! impl EguiApp for MyUi {
//!     fn update(&mut self, ctx: &egui::Context) {
//!         egui::Window::new("My Window")
//!             .open(&mut self.show_window)
//!             .show(ctx, |ui| {
//!                 ui.add(egui::Slider::new(&mut self.value, 0.0..=100.0).text("Value"));
//!             });
//!     }
//! }
//!
//! // In your app:
//! let ui = Arc::new(std::sync::RwLock::new(MyUi {
//!     show_window: true,
//!     value: 50.0,
//! }));
//! let controller = EguiController::new(device.clone(), ui);
//! ```

mod controller;
mod input;
mod renderer;

pub use controller::EguiController;
pub use egui;

use std::sync::Arc;

/// Trait for applications that use egui.
///
/// Implement this trait to define your UI logic. The `update` method is called
/// every frame to build the UI using egui's immediate mode API.
///
/// # Example
///
/// ```ignore
/// use redlilium_graphics::egui::EguiApp;
///
/// struct MyUi {
///     counter: i32,
/// }
///
/// impl EguiApp for MyUi {
///     fn update(&mut self, ctx: &egui::Context) {
///         egui::CentralPanel::default().show(ctx, |ui| {
///             ui.heading("My Application");
///             if ui.button("Increment").clicked() {
///                 self.counter += 1;
///             }
///             ui.label(format!("Counter: {}", self.counter));
///         });
///     }
/// }
/// ```
pub trait EguiApp: Send + Sync {
    /// Called every frame to build the UI.
    ///
    /// Use the provided `egui::Context` to create windows, panels, and widgets.
    fn update(&mut self, ctx: &egui::Context);

    /// Called when the UI should be set up for the first time.
    ///
    /// Override this to configure egui's style, fonts, etc.
    fn setup(&mut self, _ctx: &egui::Context) {}
}

/// Type alias for an Arc-wrapped EguiApp with interior mutability.
pub type ArcEguiApp = Arc<std::sync::RwLock<dyn EguiApp>>;
