use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::log_capture::LogBuffer;

/// Console panel that displays captured log entries.
pub struct ConsolePanel {
    buffer: Arc<Mutex<LogBuffer>>,
    /// Reference time for displaying elapsed seconds.
    start_time: Instant,
    /// Minimum log level to display (inclusive).
    min_level: log::Level,
    /// Text filter (case-insensitive substring match).
    filter_text: String,
    /// Whether to auto-scroll to the bottom.
    auto_scroll: bool,
    /// Number of entries last frame (to detect new entries).
    last_count: usize,
}

impl ConsolePanel {
    pub fn new(buffer: Arc<Mutex<LogBuffer>>) -> Self {
        Self {
            buffer,
            start_time: Instant::now(),
            min_level: log::Level::Trace,
            filter_text: String::new(),
            auto_scroll: true,
            last_count: 0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
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

        let new_entries = buf.entries().len() != self.last_count;
        self.last_count = buf.entries().len();

        let row_height = ui.text_style_height(&egui::TextStyle::Monospace) + 2.0;
        let total_rows = entries.len();

        egui::ScrollArea::vertical()
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

        // If the user scrolls up, disable auto-scroll; new entries re-enable it
        if new_entries {
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
