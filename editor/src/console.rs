use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::log_capture::LogBuffer;

/// Console panel that displays captured log entries and panic information.
pub struct ConsolePanel {
    buffer: Arc<Mutex<LogBuffer>>,
    /// Reference time for displaying elapsed seconds.
    start_time: Instant,
    /// Minimum log level to display (inclusive).
    min_level: log::Level,
    /// Text filter (case-insensitive substring match).
    filter_text: String,
    /// Whether to auto-scroll to the bottom (pin to the newest line).
    auto_scroll: bool,
}

impl ConsolePanel {
    pub fn new(buffer: Arc<Mutex<LogBuffer>>) -> Self {
        Self {
            buffer,
            start_time: Instant::now(),
            min_level: log::Level::Trace,
            filter_text: String::new(),
            auto_scroll: true,
        }
    }

    /// Display the console panel with optional panic info banner.
    pub fn show(&mut self, ui: &mut egui::Ui, panic_info: Option<redlilium_ecs::PanicInfo>) {
        // Display panic banner if active
        if let Some(ref info) = panic_info {
            ui.painter().rect_filled(
                ui.available_rect_before_wrap(),
                egui::CornerRadius::default(),
                crate::theme::ERROR.gamma_multiply(0.2),
            );
            ui.vertical(|ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("⚠ GAME PANIC")
                            .size(14.0)
                            .color(crate::theme::ERROR)
                            .strong(),
                    );
                });
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("System: {}", info.system))
                            .color(crate::theme::TEXT_PRIMARY),
                    );
                });
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("Schedule: {}", info.schedule))
                            .color(crate::theme::TEXT_SECONDARY),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("Tick: {}", info.tick))
                            .color(crate::theme::TEXT_SECONDARY),
                    );
                });
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("Error: {}", info.message))
                            .monospace()
                            .color(crate::theme::TEXT_PRIMARY),
                    );
                    ui.add_space(4.0);
                });
                ui.add_space(4.0);
            });
            ui.separator();
        }

        // Top toolbar
        ui.horizontal(|ui| {
            // Level filter buttons
            for &level in &[
                log::Level::Error,
                log::Level::Warn,
                log::Level::Info,
                log::Level::Debug,
                log::Level::Trace,
            ] {
                let selected = self.min_level >= level;
                let label = egui::RichText::new(level_label(level)).color(level_color(level));
                if ui.selectable_label(selected, label).clicked() {
                    self.min_level = level;
                }
            }

            ui.separator();

            // Text filter
            ui.label("Filter:");
            ui.text_edit_singleline(&mut self.filter_text);

            ui.separator();

            // Clear button
            if ui.button("Clear").clicked()
                && let Ok(mut buf) = self.buffer.lock()
            {
                buf.clear();
            }
        });

        ui.separator();

        // Log entries
        let buf = self.buffer.lock();
        let Ok(buf) = buf else { return };

        let filter_lower = self.filter_text.to_lowercase();
        let entries: Vec<_> = buf
            .entries()
            .iter()
            .filter(|e| e.level <= self.min_level)
            .filter(|e| {
                filter_lower.is_empty()
                    || e.message.to_lowercase().contains(&filter_lower)
                    || e.target.to_lowercase().contains(&filter_lower)
            })
            .collect();

        let row_height = ui.text_style_height(&egui::TextStyle::Monospace) + 2.0;
        let total_rows = entries.len();

        let output = egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(self.auto_scroll)
            .show_rows(ui, row_height, total_rows, |ui, row_range| {
                for row in row_range {
                    let entry = entries[row];
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;

                        // Elapsed time
                        let elapsed = entry.timestamp.duration_since(self.start_time);
                        let secs = elapsed.as_secs();
                        let millis = elapsed.subsec_millis();
                        let time_str = format!("{secs:>5}.{millis:03}");
                        let time_text = egui::RichText::new(time_str)
                            .color(crate::theme::TEXT_MUTED)
                            .monospace();
                        ui.label(time_text);

                        // Level label
                        let level_text = egui::RichText::new(level_label(entry.level))
                            .color(level_color(entry.level))
                            .monospace();
                        ui.label(level_text);

                        // Target in subdued color
                        let target_text = egui::RichText::new(&entry.target)
                            .color(crate::theme::TEXT_SECONDARY)
                            .monospace();
                        ui.label(target_text);

                        // Message
                        let msg_text = egui::RichText::new(&entry.message).monospace();
                        ui.label(msg_text);
                    });
                }
            });

        // Drive auto-scroll from the user's scroll position/gesture. While
        // `auto_scroll` is set, `stick_to_bottom` keeps the view pinned to the
        // newest line; a scroll gesture over the console releases that pin so
        // the user can read history, and returning to the bottom re-engages it.
        let max_offset = (output.content_size.y - output.inner_rect.height()).max(0.0);
        let at_bottom = output.state.offset.y >= max_offset - 2.0;
        let pointer_over = ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| output.inner_rect.contains(p));
        if self.auto_scroll {
            // A scroll gesture over the area means the user wants to leave the
            // tail. (Sign-agnostic: any gesture releases; if they were merely
            // nudging at the bottom, the `at_bottom` branch re-engages next frame.)
            if pointer_over && ui.input(|i| i.raw_scroll_delta.y != 0.0) {
                self.auto_scroll = false;
            }
        } else if at_bottom {
            self.auto_scroll = true;
        }
    }
}

fn level_label(level: log::Level) -> &'static str {
    match level {
        log::Level::Error => "ERR",
        log::Level::Warn => "WRN",
        log::Level::Info => "INF",
        log::Level::Debug => "DBG",
        log::Level::Trace => "TRC",
    }
}

fn level_color(level: log::Level) -> egui::Color32 {
    match level {
        log::Level::Error => crate::theme::ERROR,
        log::Level::Warn => crate::theme::WARNING,
        log::Level::Info => crate::theme::SUCCESS,
        log::Level::Debug => crate::theme::INFO,
        log::Level::Trace => crate::theme::TEXT_MUTED,
    }
}
