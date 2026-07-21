//! Frame pacing: turning a CPU-sampled frame delta into the interval the frame
//! will actually be displayed for.
//!
//! `run_frame` consumes "how long this frame will be shown". A raw
//! `Instant::now()` delta is only an *estimator* of that quantity: under a
//! blocking present (FIFO/vsync) the CPU wakes from acquire/fence at
//! scheduler-jittered instants, while the pixels land on vsync boundaries.
//! Measured on a 60 Hz panel with the renderer running just under refresh, the
//! deltas looked like
//!
//! ```text
//! 16.1, 16.9, 16.7, 17.1, 18.5, 18.8, 29.2   (sum 133.3 ms)
//! ```
//!
//! — seven CPU frames whose total is exactly eight vsync intervals: six frames
//! shown for one interval, one shown for two, smeared by up to ±13%. Feeding
//! those numbers to a simulation makes it advance 13% too far on one frame and
//! 12% too little on the next, which reads as the world subtly speeding up and
//! slowing down. Fixed-timestep stepping and render interpolation cannot help:
//! they make motion exact in *simulation* time, and it is the map from
//! simulation time to *display* time that is wrong.
//!
//! [`FramePacer`] recovers the real interval. When a delta sits close to a
//! multiple of the display period it is snapped to that multiple, and the
//! rounding residue is carried and bled back so the paced clock cannot drift
//! away from the wall clock. When deltas are *not* clustered on refresh
//! multiples — a variable-refresh display, a compositor that does not block,
//! a genuine hitch — nothing is snapped and the raw delta passes through.

use std::collections::VecDeque;

/// How far from a refresh multiple a delta may sit and still be read as that
/// multiple, as a fraction of the multiple. Wide enough for the ±13% smear
/// observed under FIFO, narrow enough that a genuinely unquantized cadence
/// never snaps (a delta halfway between one and two intervals passes through).
const SNAP_TOLERANCE: f32 = 0.2;

/// Most the carried rounding residue may shift a single frame, seconds.
const MAX_BLEED: f32 = 0.001;

/// Ceiling on the carried residue, seconds — a backstop so a pathological run
/// cannot bank unbounded time and then spend it.
const MAX_RESIDUE: f32 = 0.1;

/// Deltas kept for the fallback period estimate.
const HISTORY: usize = 120;

/// Plausible display periods: 240 Hz down to 20 Hz. Anything outside is a
/// stall, a suspended process, or a bogus report — never a refresh rate.
const MIN_PERIOD: f32 = 1.0 / 240.0;
const MAX_PERIOD: f32 = 1.0 / 20.0;

/// Estimates the display interval a frame occupies from the raw CPU delta.
///
/// See the module docs for why this is an estimator of the same quantity the
/// raw delta estimates, rather than a smoothing knob.
pub struct FramePacer {
    /// Recent raw deltas, for the fallback period estimate.
    history: VecDeque<f32>,
    /// Display period reported by the windowing system, when it reports one.
    hint: Option<f32>,
    /// Carried residue (`Σraw − Σpaced`), bled back a bounded amount per frame
    /// so snapping cannot drift the simulation clock off the wall clock.
    residue: f32,
}

impl Default for FramePacer {
    fn default() -> Self {
        Self::new()
    }
}

impl FramePacer {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY),
            hint: None,
            residue: 0.0,
        }
    }

    /// Record the display period the windowing system reports, in seconds.
    /// Implausible values are ignored, which also covers "no monitor".
    pub fn set_display_period(&mut self, period: Option<f32>) {
        self.hint = period.filter(|p| (MIN_PERIOD..=MAX_PERIOD).contains(p));
    }

    /// Best estimate of the display period: the reported one, else the median
    /// of recent deltas (most frames occupy a single interval, so the median
    /// lands on one). `None` until there is enough history to guess.
    fn period(&self) -> Option<f32> {
        if let Some(hint) = self.hint {
            return Some(hint);
        }
        if self.history.len() < HISTORY / 4 {
            return None;
        }
        let mut sorted: Vec<f32> = self.history.iter().copied().collect();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        (MIN_PERIOD..=MAX_PERIOD)
            .contains(&median)
            .then_some(median)
    }

    /// Convert a raw frame delta into the interval this frame is displayed for.
    pub fn pace(&mut self, raw: f32) -> f32 {
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(raw);

        let Some(period) = self.period() else {
            return raw;
        };

        let multiple = (raw / period).round().max(1.0);
        let snapped = multiple * period;
        if (raw - snapped).abs() > snapped * SNAP_TOLERANCE {
            // Not a quantized cadence: no honest multiple to attribute this
            // frame to, so leave the measurement alone.
            return raw;
        }

        self.residue = (self.residue + raw - snapped).clamp(-MAX_RESIDUE, MAX_RESIDUE);
        let bleed = self.residue.clamp(-MAX_BLEED, MAX_BLEED);
        self.residue -= bleed;
        snapped + bleed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cadence measured on a 60 Hz panel while the artifact was visible:
    /// seven CPU frames totalling exactly eight vsync intervals.
    const MEASURED_MS: [f32; 7] = [16.1, 16.9, 16.7, 17.1, 18.5, 18.8, 29.2];

    #[test]
    fn snaps_a_measured_fifo_cadence_to_whole_vsync_intervals() {
        let period = 1.0 / 60.0f32;
        let mut pacer = FramePacer::new();
        pacer.set_display_period(Some(period));

        let (mut raw_total, mut paced_total) = (0.0f32, 0.0f32);
        for _ in 0..10 {
            for ms in MEASURED_MS {
                let raw = ms / 1000.0;
                let paced = pacer.pace(raw);
                raw_total += raw;
                paced_total += paced;

                let multiple = (paced / period).round();
                assert!(
                    multiple == 1.0 || multiple == 2.0,
                    "{ms}ms became {multiple} intervals"
                );
                assert!(
                    (paced - multiple * period).abs() <= MAX_BLEED + 1e-6,
                    "{ms}ms paced to {paced}, not a whole interval"
                );
            }
        }
        // The paced clock must track the wall clock: snapping redistributes
        // time between frames, it must not create or destroy any.
        assert!(
            (paced_total - raw_total).abs() < 5e-3,
            "paced {paced_total}s drifted from raw {raw_total}s"
        );
    }

    #[test]
    fn leaves_an_unquantized_cadence_alone() {
        let mut pacer = FramePacer::new();
        pacer.set_display_period(Some(1.0 / 60.0));
        // Halfway between one and two intervals: no multiple is defensible, so
        // the measurement passes through. Variable-refresh displays live here,
        // and there the raw delta already *is* display time.
        let raw = 0.025;
        assert_eq!(pacer.pace(raw), raw);
    }

    #[test]
    fn passes_through_until_a_period_is_known() {
        let mut pacer = FramePacer::new();
        // No hint and no history yet — nothing to snap against.
        assert_eq!(pacer.pace(0.0181), 0.0181);
    }

    #[test]
    fn recovers_the_period_from_history_without_a_hint() {
        let mut pacer = FramePacer::new();
        let period = 1.0 / 60.0f32;
        for _ in 0..HISTORY {
            pacer.pace(period);
        }
        // A smeared frame now snaps to one interval on the median estimate.
        let paced = pacer.pace(0.0188);
        assert!(
            (paced - period).abs() <= MAX_BLEED + 1e-6,
            "expected ~{period}, got {paced}"
        );
    }
}
