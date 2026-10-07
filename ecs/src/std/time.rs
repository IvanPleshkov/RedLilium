//! Dual-clock time management: RealTime (host clock) and GameTime (game clock).
//!
//! **RealTime** advances every ticked frame. Use it for UI animations,
//! profiling, and editor-side logic.
//!
//! **GameTime** supports slow-motion / fast-forward via
//! [`set_scale`](GameTime::set_scale) and drives the fixed simulation clock.
//!
//! Both clocks are advanced by `Schedules::run_frame` every time it is
//! called, each with its own scale (`RealTime` implicitly 1.0, `GameTime` via
//! [`GameTime::scale`]). Scale zero freezes fixed simulation and its render
//! interpolation while input, per-frame schedules, and real-time UI continue.
//! A host may instead stop calling `run_frame` to freeze the entire world
//! (for example, an editor play session). Neither clock touches the world's
//! change-detection tick — they are plain time resources.

/// Real (wall-clock) time that advances every ticked frame.
///
/// Use this for UI, profiling, and slow-motion envelopes that must expire even
/// at game scale zero. When present, [`crate::Schedules::run_frame`] advances
/// this resource after `PreUpdate` and state transitions, before `FixedUpdate`.
/// `Update` and `PostUpdate` therefore see the current frame's real time.
#[derive(Debug, Clone, Copy)]
pub struct RealTime {
    elapsed: f64,
    delta: f64,
}

impl RealTime {
    /// Create a new RealTime at 0.
    pub fn new() -> Self {
        Self {
            elapsed: 0.0,
            delta: 0.0,
        }
    }

    /// Advance by a frame delta. Call once per frame before game systems run.
    pub fn advance(&mut self, dt: f64) {
        self.delta = dt;
        self.elapsed += dt;
    }

    /// Frame delta (wall-clock time since last frame).
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Total elapsed time since startup.
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }
}

impl Default for RealTime {
    fn default() -> Self {
        Self::new()
    }
}

/// Game time that supports slow-motion / fast-forward.
///
/// `Schedules::run_frame` advances GameTime every time it runs, using the
/// clock's own [`scale`](GameTime::scale) (set 0.5 for half-speed slow-mo,
/// 0.0 to freeze while still ticking, etc.). Freezing gameplay time entirely
/// is the host's call: either set the scale to 0, or — for an editor play
/// session — simply stop calling `run_frame` on that world.
///
/// The same scaled frame delta feeds [`crate::FixedUpdate`]. Its timestep
/// ([`crate::Time::fixed_delta`]) remains constant in game seconds: at 0.25×,
/// four times fewer steps execute. [`crate::Time::fixed_alpha`] follows that
/// clock too. Use `Time::delta()` in fixed gameplay systems, `GameTime::delta()`
/// in variable-step gameplay, and [`RealTime`] for unscaled UI and timers.
/// `GameTime::delta()` is a frame delta even inside `FixedUpdate`; do not use
/// it as the per-step delta.
#[derive(Debug, Clone, Copy)]
pub struct GameTime {
    elapsed: f64,
    delta: f64,
    /// Time multiplier applied by `run_frame` while the game simulates
    /// (1.0 = normal, 0.5 = half-speed, 0.0 = frozen).
    scale: f64,
    /// Kahan compensation term: carries the low-order bits lost when a small
    /// scaled delta is added to a large `elapsed`, so long slow-mo sessions
    /// accumulate no summation drift.
    compensation: f64,
}

impl GameTime {
    /// Create a new GameTime at 0, normal speed.
    pub fn new() -> Self {
        Self {
            elapsed: 0.0,
            delta: 0.0,
            scale: 1.0,
            compensation: 0.0,
        }
    }

    /// Advance game time by a scaled delta.
    ///
    /// `dt` is the frame delta (e.g., 1/60 for a 60fps frame).
    /// `scale` is the effective multiplier for this tick (0.0 = frozen,
    /// 1.0 = normal, 0.5 = half-speed). `Schedules::run_frame` passes
    /// [`self.scale`](GameTime::scale) on every call.
    ///
    /// Uses Kahan compensated summation so tiny scaled deltas added to a
    /// large `elapsed` do not lose precision over long sessions.
    pub fn tick(&mut self, dt: f64, scale: f64) {
        let scaled_delta = dt * scale;
        self.delta = scaled_delta;

        // A frozen tick must leave `elapsed` bit-identical — don't let it
        // flush the pending compensation remainder.
        if scaled_delta == 0.0 {
            return;
        }

        let y = scaled_delta - self.compensation;
        let t = self.elapsed + y;
        self.compensation = (t - self.elapsed) - y;
        self.elapsed = t;
    }

    /// Frame delta (game time advanced this frame).
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Total elapsed game time.
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }

    /// The configured time multiplier (1.0 = normal speed).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the time multiplier for slow-mo / fast-forward (1.0 = normal).
    ///
    /// [`crate::Schedules::run_frame`] samples it after `PreUpdate` and state
    /// transitions. Changes there affect this frame; changes in `FixedUpdate`,
    /// `Update`, or `PostUpdate` affect the next frame. All fixed steps within
    /// one frame use that single sample. Restoring 1.0 never catches up time
    /// skipped while slowed or frozen, and preserves the fractional fixed step.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is negative or non-finite. Reverse time is unsupported.
    pub fn set_scale(&mut self, scale: f64) {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "Game time scale must be finite and non-negative"
        );
        self.scale = scale;
    }

    /// Reset elapsed/delta to 0. Keeps the configured scale. Useful for a
    /// host that wants to zero the clock at the start of a new session
    /// (e.g. the editor's play world, which gets a fresh `GameTime` anyway
    /// via `App::boot`, but game code may want an explicit re-zero point).
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.delta = 0.0;
        self.compensation = 0.0;
    }
}

impl Default for GameTime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_time_advances() {
        let mut rt = RealTime::new();
        rt.advance(0.016);
        assert!((rt.delta() - 0.016).abs() < f64::EPSILON);
        assert!((rt.elapsed() - 0.016).abs() < f64::EPSILON);

        rt.advance(0.016);
        assert!((rt.delta() - 0.016).abs() < f64::EPSILON);
        assert!((rt.elapsed() - 0.032).abs() < f64::EPSILON);
    }

    #[test]
    fn game_time_normal_speed() {
        let mut gt = GameTime::new();
        gt.tick(0.016, 1.0);
        assert!((gt.delta() - 0.016).abs() < f64::EPSILON);
        assert!((gt.elapsed() - 0.016).abs() < f64::EPSILON);
    }

    #[test]
    fn game_time_paused() {
        let mut gt = GameTime::new();
        gt.tick(0.016, 0.0);
        assert!((gt.delta() - 0.0).abs() < f64::EPSILON);
        assert!((gt.elapsed() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn game_time_half_speed() {
        let mut gt = GameTime::new();
        gt.tick(0.016, 0.5);
        assert!((gt.delta() - 0.008).abs() < f64::EPSILON);
        assert!((gt.elapsed() - 0.008).abs() < f64::EPSILON);
    }

    #[test]
    fn game_time_slow_motion() {
        let mut gt = GameTime::new();
        // At 0.33× scale over 5 frames: 0.016 * 0.33 * 5 ≈ 0.0264
        for _ in 0..5 {
            gt.tick(0.016, 0.33);
        }
        assert!((gt.elapsed() - 0.0264).abs() < 0.001);
    }

    #[test]
    fn game_time_reset() {
        let mut gt = GameTime::new();
        gt.tick(0.016, 1.0);
        assert!((gt.elapsed() - 0.016).abs() < f64::EPSILON);
        gt.reset();
        assert!((gt.elapsed()).abs() < f64::EPSILON);
        assert!((gt.delta()).abs() < f64::EPSILON);
    }

    /// #67: Kahan summation keeps a long slow-mo session drift-free at the
    /// precision level, not merely "under a microsecond per frame". 10k
    /// frames at 0.25× must match n·dt·scale to within ~1e-12.
    #[test]
    fn game_time_no_drift_over_long_slow_mo_session() {
        let mut gt = GameTime::new();
        let dt = 1.0 / 60.0;
        let scale = 0.25;
        let n = 10_000;

        for _ in 0..n {
            gt.tick(dt, scale);
        }

        let expected = n as f64 * dt * scale;
        let drift = (gt.elapsed() - expected).abs();
        assert!(
            drift < 1e-12,
            "compensated summation drifted by {drift} over {n} slow-mo frames"
        );
    }

    #[test]
    fn game_time_scale_survives_reset() {
        let mut gt = GameTime::new();
        assert_eq!(gt.scale(), 1.0);
        gt.set_scale(0.5);
        gt.tick(0.016, gt.scale());
        gt.reset();
        assert_eq!(gt.elapsed(), 0.0);
        assert_eq!(gt.scale(), 0.5, "reset must keep the configured scale");
    }

    #[test]
    fn game_time_rejects_invalid_scale_without_changing_configuration() {
        for scale in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut time = GameTime::new();
            time.set_scale(0.25);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                time.set_scale(scale);
            }));
            assert!(result.is_err());
            assert_eq!(time.scale(), 0.25);
        }
    }

    #[test]
    fn game_time_frozen_tick_is_bit_identical() {
        let mut gt = GameTime::new();
        // Accrue a non-trivial compensation remainder first.
        for _ in 0..1000 {
            gt.tick(1.0 / 60.0, 0.25);
        }
        let frozen = gt.elapsed();
        for _ in 0..100 {
            gt.tick(1.0 / 60.0, 0.0);
        }
        assert_eq!(gt.elapsed(), frozen, "frozen ticks must not move elapsed");
        assert_eq!(gt.delta(), 0.0);
    }

    #[test]
    fn game_time_fractional_accumulator_prevents_drift() {
        let mut gt = GameTime::new();
        let dt = 0.016; // 60 FPS frame
        let scale = 0.25; // Quarter speed
        let expected_per_frame = dt * scale;

        // Over 100 frames, accumulator should prevent significant drift
        for _ in 0..100 {
            gt.tick(dt, scale);
        }

        let expected = 100.0 * expected_per_frame;
        // With fractional accumulator, drift should be < 1 microsecond per frame
        let max_drift = 100.0 * 0.000001;
        assert!(
            (gt.elapsed() - expected).abs() < max_drift,
            "Drift {} exceeds maximum {} at quarter speed over 100 frames",
            (gt.elapsed() - expected).abs(),
            max_drift
        );
    }

    #[test]
    fn game_time_fractional_accumulator_half_speed() {
        let mut gt = GameTime::new();
        let dt = 0.016;
        let scale = 0.5;

        // 60 frames at half speed: 0.016 * 0.5 * 60 = 0.48
        for _ in 0..60 {
            gt.tick(dt, scale);
        }

        let expected = 0.48;
        assert!(
            (gt.elapsed() - expected).abs() < 0.0001,
            "Half-speed accumulation drifted: {} vs {}",
            gt.elapsed(),
            expected
        );
    }
}
