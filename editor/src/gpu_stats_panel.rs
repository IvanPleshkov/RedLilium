//! Read-only GPU timings window (#95).
//!
//! Displays per-pass GPU time from the Vulkan timestamp queries, surfaced as a
//! plain [`FrameGpuTimings`] resource. No [`EditAction`](redlilium_core) — this
//! panel only reads. Numbers are a few frames stale (that is how the timestamps
//! are read back) and smoothed with a 60-frame EMA so they are readable.

use std::collections::HashMap;

use redlilium_graphics::FrameGpuTimings;

/// Exponential moving-average window, in frames.
const EMA_FRAMES: f32 = 60.0;

/// Panel state: the rolling averages. The latest raw timings live in an ECS
/// resource; only the smoothing accumulators are owned here.
pub struct GpuStatsPanel {
    /// EMA smoothing factor (`2 / (N + 1)` for an N-frame window).
    alpha: f32,
    /// EMA of the whole-frame total (sum of submit totals).
    frame_total_ema: f32,
    /// EMA of each submit/pass duration, keyed by a stable `queue|label` string.
    ema: HashMap<String, f32>,
    /// False until the first sample seeds the averages (avoids a slow ramp
    /// from zero).
    seeded: bool,
}

impl Default for GpuStatsPanel {
    fn default() -> Self {
        Self {
            alpha: 2.0 / (EMA_FRAMES + 1.0),
            frame_total_ema: 0.0,
            ema: HashMap::new(),
            seeded: false,
        }
    }
}

impl GpuStatsPanel {
    /// Update `slot` toward `sample` and return the smoothed value.
    fn smooth(&mut self, key: String, sample: f32, seeded: bool) -> f32 {
        let entry = self.ema.entry(key).or_insert(sample);
        if seeded {
            *entry += self.alpha * (sample - *entry);
        } else {
            *entry = sample;
        }
        *entry
    }

    /// Draw the panel body. `supported` is
    /// [`DeviceCapabilities::gpu_timestamps`](redlilium_graphics::DeviceCapabilities);
    /// when false the panel degrades to an "unavailable" message.
    pub fn show(&mut self, ui: &mut egui::Ui, timings: &FrameGpuTimings, supported: bool) {
        if !supported {
            ui.label(
                egui::RichText::new("GPU timings unavailable on this backend")
                    .monospace()
                    .color(crate::theme::TEXT_MUTED),
            );
            ui.label(
                egui::RichText::new(
                    "(requires a Vulkan device with timestamp queries; wgpu/dummy report none)",
                )
                .monospace()
                .color(crate::theme::TEXT_MUTED),
            );
            return;
        }

        // Advance the averages from this frame's sample before rendering.
        let was_seeded = self.seeded;
        let frame_total = timings.frame_total_ms();
        if was_seeded {
            self.frame_total_ema += self.alpha * (frame_total - self.frame_total_ema);
        } else {
            self.frame_total_ema = frame_total;
        }

        // Frame-total header row.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(
                egui::RichText::new("Frame GPU total")
                    .monospace()
                    .color(crate::theme::TEXT_PRIMARY),
            );
            ui.label(
                egui::RichText::new(format!("{frame_total:>7.3} ms"))
                    .monospace()
                    .color(crate::theme::WARNING),
            );
            ui.label(
                egui::RichText::new(format!("(avg {:>7.3})", self.frame_total_ema))
                    .monospace()
                    .color(crate::theme::TEXT_MUTED),
            );
        });
        ui.separator();

        if timings.is_empty() {
            ui.label(
                egui::RichText::new("waiting for timing data…")
                    .monospace()
                    .color(crate::theme::TEXT_MUTED),
            );
            self.seeded = true;
            return;
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (si, submit) in timings.submits.iter().enumerate() {
                    let queue = format!("{:?}", submit.queue);
                    // Submit header row.
                    let total_avg = self.smooth(
                        format!("{si}|{queue}|<submit>"),
                        submit.total_ms,
                        was_seeded,
                    );
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.label(
                            egui::RichText::new(format!("▸ {queue}"))
                                .monospace()
                                .color(crate::theme::INFO),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:>7.3} ms", submit.total_ms))
                                .monospace()
                                .color(crate::theme::TEXT_SECONDARY),
                        );
                        ui.label(
                            egui::RichText::new(format!("(avg {total_avg:>7.3})"))
                                .monospace()
                                .color(crate::theme::TEXT_MUTED),
                        );
                    });

                    // Per-pass rows.
                    for (pi, (name, ms)) in submit.passes.iter().enumerate() {
                        let avg = self.smooth(format!("{si}|{queue}|{pi}|{name}"), *ms, was_seeded);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            ui.add_space(16.0);
                            ui.label(
                                egui::RichText::new(format!("{name:<24}"))
                                    .monospace()
                                    .color(crate::theme::TEXT_SECONDARY),
                            );
                            ui.label(
                                egui::RichText::new(format!("{ms:>7.3} ms"))
                                    .monospace()
                                    .color(crate::theme::TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(format!("(avg {avg:>7.3})"))
                                    .monospace()
                                    .color(crate::theme::TEXT_MUTED),
                            );
                        });
                    }
                }
            });

        self.seeded = true;
    }
}
