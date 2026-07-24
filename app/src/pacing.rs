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
//!
//! One more distortion needs the present mode to untangle
//! ([`FramePacer::set_vsync`]): the swapchain keeps a queue of
//! `frames_in_flight` images, so right after a long frame the queue has room
//! and acquire returns *instantly* for a frame or two — CPU deltas of a few
//! milliseconds on a 60 Hz display (`33, 4.7, 4.7, 29, 16.7, …`). Those
//! sub-period deltas are the queue refilling, not short displays: under a
//! blocking present every one of those frames is still shown for at least a
//! whole interval. Passing them through freezes the simulation for a frame
//! while the display advances a full interval — a visible stutter around
//! every hitch. With vsync declared, sub-period deltas are attributed one
//! whole interval and the difference is banked in the residue.

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
    /// Whether the present blocks on the display (FIFO). Enables the
    /// queue-refill correction: no frame can be shown for less than one
    /// interval, so sub-period deltas snap up instead of passing through.
    vsync: bool,
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
            vsync: false,
        }
    }

    /// Record the display period the windowing system reports, in seconds.
    /// Implausible values are ignored, which also covers "no monitor".
    pub fn set_display_period(&mut self, period: Option<f32>) {
        self.hint = period.filter(|p| (MIN_PERIOD..=MAX_PERIOD).contains(p));
    }

    /// Declare whether the present blocks on the display (FIFO/vsync).
    ///
    /// Under a blocking present no frame is displayed for less than one whole
    /// interval, so a sub-period CPU delta can only be the present queue
    /// refilling after a long frame — [`pace`](Self::pace) then attributes it
    /// one interval instead of passing it through. Safe with adaptive sync:
    /// VRR stretches intervals beyond the period, never below it.
    pub fn set_vsync(&mut self, vsync: bool) {
        self.vsync = vsync;
    }

    /// The current display-period estimate, if any — for diagnostics.
    pub fn display_period(&self) -> Option<f32> {
        self.period()
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
            // A sub-period delta under a blocking present is the queue
            // refilling after a long frame (see module docs): the frame is
            // still shown for one whole interval, so attribute that interval
            // (`raw < period` implies `multiple == 1`) and bank the
            // difference. Anything else is not a quantized cadence — no
            // honest multiple to attribute the frame to, so leave the
            // measurement alone.
            if !(self.vsync && raw < period) {
                return raw;
            }
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
    fn attributes_queue_refill_bursts_a_whole_interval_under_vsync() {
        let period = 1.0 / 60.0f32;
        let mut pacer = FramePacer::new();
        pacer.set_display_period(Some(period));
        pacer.set_vsync(true);

        // Steady, then a hitch drains the present queue and two frames refill
        // it near-instantly — the cadence FIFO produces around any hitch.
        for _ in 0..30 {
            pacer.pace(period);
        }
        pacer.pace(2.0 * period);
        for _ in 0..2 {
            let paced = pacer.pace(0.3 * period);
            assert!(
                paced >= period - MAX_BLEED - 1e-6,
                "refill frame paced to {paced}, below one display interval"
            );
            assert!(paced <= period + MAX_BLEED + 1e-6);
        }
    }

    #[test]
    fn passes_queue_refill_deltas_through_without_vsync() {
        let mut pacer = FramePacer::new();
        pacer.set_display_period(Some(1.0 / 60.0));
        // No blocking present declared: a sub-period delta may be a genuine
        // short display (immediate mode), so it must not be inflated.
        let raw = 0.0047;
        assert_eq!(pacer.pace(raw), raw);
    }

    #[test]
    fn repays_the_refill_attribution_so_the_clock_tracks_wall_time() {
        let period = 1.0 / 60.0f32;
        let mut pacer = FramePacer::new();
        pacer.set_display_period(Some(period));
        pacer.set_vsync(true);

        let (mut raw_total, mut paced_total) = (0.0f32, 0.0f32);
        let mut feed = |pacer: &mut FramePacer, raw: f32| {
            raw_total += raw;
            paced_total += pacer.pace(raw);
        };
        feed(&mut pacer, 2.0 * period);
        feed(&mut pacer, 0.3 * period);
        feed(&mut pacer, 0.3 * period);
        // The refill over-attribution is banked in the residue and bled back
        // a bounded amount per frame; a second of steady frames repays it.
        for _ in 0..60 {
            feed(&mut pacer, period);
        }
        assert!(
            (paced_total - raw_total).abs() < 1e-3,
            "paced {paced_total}s drifted from raw {raw_total}s"
        );
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
