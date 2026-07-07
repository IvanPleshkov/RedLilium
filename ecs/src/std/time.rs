//! Dual-clock time management: RealTime (always advancing) and GameTime (pauseable).
//!
//! **RealTime** advances every frame, unaffected by Play/Pause. Use it for
//! UI animations, profiling, and editor-side logic.
//!
//! **GameTime** is pauseable and supports slow-motion. It is zeroed when Play
//! starts and consumed by game logic. Use the [`GameTime::tick`] method to
//! advance it by a scaled delta.

/// Real (wall-clock) time that always advances, regardless of Play/Pause.
///
/// Use this for editor UI, profiling, and background work that should not be
/// affected by gameplay pauses. Every frame, call [`advance`](RealTime::advance)
/// before running game systems.
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

/// Game time that can be paused and slowed.
///
/// GameTime starts at 0 when Play begins. Use [`tick`](GameTime::tick) to
/// advance it by a scaled delta (0.0 for paused, 0.5 for half-speed, etc.).
#[derive(Debug, Clone, Copy)]
pub struct GameTime {
    elapsed: f64,
    delta: f64,
}

impl GameTime {
    /// Create a new GameTime at 0.
    pub fn new() -> Self {
        Self {
            elapsed: 0.0,
            delta: 0.0,
        }
    }

    /// Advance game time by a scaled delta.
    ///
    /// `dt` is the frame delta (e.g., 1/60 for a 60fps frame).
    /// `scale` is the time multiplier (0.0 = paused, 1.0 = normal, 0.5 = half-speed).
    ///
    /// The delta is scaled directly: at 0.5× scale, each frame advances half the
    /// wall-clock time. This keeps game simulation smooth across any speed scale.
    pub fn tick(&mut self, dt: f64, scale: f64) {
        self.delta = dt * scale;
        self.elapsed += self.delta;
    }

    /// Frame delta (game time advanced this frame).
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Total elapsed game time.
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }

    /// Reset to 0 when Play starts.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.delta = 0.0;
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
}
