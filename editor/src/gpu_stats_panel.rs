//! Read-only GPU stats window (#95, #98).
//!
//! Two sections, both plain resource readers — no [`EditAction`](redlilium_core):
//! per-pass GPU timings from the Vulkan timestamp queries ([`FrameGpuTimings`],
//! #95), smoothed with a 60-frame EMA, and GPU-memory figures ([`GpuMemoryStats`],
//! #98): driver heap budgets (`VK_EXT_memory_budget`), allocator totals, and the
//! engine's live-resource counts. All numbers are a few frames stale (that is
//! how they are sampled).

use std::collections::HashMap;

use redlilium_graphics::{FrameGpuTimings, GpuMemoryStats};

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

    /// Draw the panel body: a "GPU Timings" section (#95) and a "Memory"
    /// section (#98), each collapsible. `timings_supported` is
    /// [`DeviceCapabilities::gpu_timestamps`](redlilium_graphics::DeviceCapabilities)
    /// and `memory_budget` is
    /// [`DeviceCapabilities::memory_budget`](redlilium_graphics::DeviceCapabilities);
    /// each drives its section's graceful degradation.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        timings: &FrameGpuTimings,
        timings_supported: bool,
        memory: &GpuMemoryStats,
        memory_budget: bool,
    ) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new("GPU Timings")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.show_timings(ui, timings, timings_supported);
                    });
                ui.separator();
                egui::CollapsingHeader::new("Memory")
                    .default_open(true)
                    .show(ui, |ui| {
                        Self::show_memory(ui, memory, memory_budget);
                    });
            });
    }

    /// The per-pass GPU timings section (#95). `supported` is
    /// [`DeviceCapabilities::gpu_timestamps`](redlilium_graphics::DeviceCapabilities);
    /// when false it degrades to an "unavailable" message.
    fn show_timings(&mut self, ui: &mut egui::Ui, timings: &FrameGpuTimings, supported: bool) {
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

        self.seeded = true;
    }

    /// The GPU-memory section (#98): driver heap budgets (DEVICE_LOCAL first),
    /// allocator totals, and engine resource counts. `memory_budget` is false
    /// when `VK_EXT_memory_budget` is unavailable — heaps then show sizes only,
    /// no fabricated budget. Static (no smoothing); the values already move
    /// slowly and are meaningful raw.
    fn show_memory(ui: &mut egui::Ui, memory: &GpuMemoryStats, memory_budget: bool) {
        // Heaps: DEVICE_LOCAL first, then by index.
        let mut order: Vec<usize> = (0..memory.heaps.len()).collect();
        order.sort_by_key(|&i| {
            let h = &memory.heaps[i];
            (!h.device_local, h.index)
        });

        if memory.heaps.is_empty() {
            ui.label(
                egui::RichText::new("no driver heap data on this backend")
                    .monospace()
                    .color(crate::theme::TEXT_MUTED),
            );
        }
        for &i in &order {
            let h = &memory.heaps[i];
            let tag = if h.device_local {
                "DEVICE_LOCAL"
            } else {
                "host"
            };
            match (h.usage, h.budget) {
                (Some(usage), Some(budget)) if budget > 0 => {
                    let frac = (usage as f32 / budget as f32).clamp(0.0, 1.0);
                    let text = format!(
                        "{} / {} ({:.0}%)",
                        fmt_bytes(usage),
                        fmt_bytes(budget),
                        frac * 100.0
                    );
                    ui.label(
                        egui::RichText::new(format!("heap {} · {tag}", h.index))
                            .monospace()
                            .color(crate::theme::TEXT_SECONDARY),
                    );
                    // Fill the available width, but never an infinite/NaN
                    // width — egui's hit-test unwraps `None` on a non-finite
                    // interact rect (egui 0.33 hit_test.rs:364).
                    let bar_width = ui.available_width().max(1.0);
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(egui::RichText::new(text).monospace())
                            .desired_width(bar_width),
                    );
                }
                _ => {
                    // No budget data — show the heap size only, no fake budget.
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.label(
                            egui::RichText::new(format!("heap {} · {tag}", h.index))
                                .monospace()
                                .color(crate::theme::TEXT_SECONDARY),
                        );
                        ui.label(
                            egui::RichText::new(fmt_bytes(h.size))
                                .monospace()
                                .color(crate::theme::TEXT_PRIMARY),
                        );
                    });
                }
            }
        }
        if !memory.heaps.is_empty() && !memory_budget {
            ui.label(
                egui::RichText::new("(budget/usage unavailable — VK_EXT_memory_budget off)")
                    .monospace()
                    .color(crate::theme::TEXT_MUTED),
            );
        }

        ui.separator();

        // Allocator totals (our own footprint, distinct from driver usage).
        let row = |ui: &mut egui::Ui, label: &str, value: String| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new(format!("{label:<20}"))
                        .monospace()
                        .color(crate::theme::TEXT_SECONDARY),
                );
                ui.label(
                    egui::RichText::new(value)
                        .monospace()
                        .color(crate::theme::TEXT_PRIMARY),
                );
            });
        };
        row(ui, "allocator used", fmt_bytes(memory.allocator_allocated));
        row(
            ui,
            "allocator reserved",
            fmt_bytes(memory.allocator_reserved),
        );
        row(ui, "allocations", memory.allocation_count.to_string());

        ui.separator();

        // Engine live-resource counts.
        let r = &memory.resources;
        row(ui, "buffers", r.buffers.to_string());
        row(ui, "textures", r.textures.to_string());
        row(ui, "meshes", r.meshes.to_string());
        row(ui, "materials", r.materials.to_string());
        row(ui, "samplers", r.samplers.to_string());
    }
}

/// Human-readable byte count (B / KiB / MiB / GiB), three significant-ish
/// digits. Used only for display in the stats panel.
fn fmt_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}
