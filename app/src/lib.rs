//! # RedLilium App
//!
//! Application framework for creating windowed applications with RedLilium graphics.
//!
//! This crate provides a generic `App` struct that handles window creation, event loops,
//! and graphics initialization. It is designed to be used with custom handlers for
//! window events and draw requests.
//!
//! ## Overview
//!
//! - [`AppHandler`] - Trait for handling window events and draw requests
//! - [`AppArgs`] - Trait for parsing command line arguments
//! - [`App`] - Main application struct that manages the window and graphics
//!
//! ## Example
//!
//! ```ignore
//! use redlilium_app::{App, AppHandler, AppArgs, DefaultAppArgs, AppContext, DrawContext};
//!
//! struct MyApp;
//!
//! impl AppHandler for MyApp {
//!     fn on_init(&mut self, ctx: &mut AppContext) {
//!         // Initialize resources
//!     }
//!
//!     fn on_draw(&mut self, ctx: &mut DrawContext) {
//!         // Render frame
//!     }
//! }
//!
//! fn main() {
//!     let args = DefaultAppArgs::parse();
//!     App::run(MyApp, args);
//! }
//! ```

mod app;
mod args;
mod context;
mod handler;

pub use app::App;
pub use args::{AppArgs, DefaultAppArgs, WindowMode};
pub use context::{AppContext, DrawContext};
pub use handler::AppHandler;

/// App library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the app subsystem.
///
/// This should be called before using any app functionality.
pub fn init() {
    log::info!("RedLilium App v{} initialized", VERSION);
}
