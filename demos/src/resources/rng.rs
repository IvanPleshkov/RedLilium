use redlilium_ecs::{PlayModeAware, SnapshotResource};
use serde::{Deserialize, Serialize};

/// Deterministic PRNG for reproducible gameplay. Uses PCG-like algorithm.
/// Implements PlayModeAware to reset to seed on Play.
/// Implements SnapshotResource to restore state on Stop.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct GameRNG {
    seed: u64,
    state: u64,
}

impl GameRNG {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed, state: seed }
    }

    /// Generate next u32 from PCG-like algorithm.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }

    /// Generate next u32 in range [min, max).
    pub fn next_range(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        min + (self.next_u32() % (max - min))
    }
}

impl SnapshotResource for GameRNG {
    const NAME: &'static str = "GameRNG";
}

impl PlayModeAware for GameRNG {
    fn on_play_start(&mut self) {
        self.state = self.seed;
    }

    fn on_pause(&mut self) {}

    fn on_resume(&mut self) {}

    fn on_stop(&mut self) {}
}
