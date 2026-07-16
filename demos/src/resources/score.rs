use redlilium_ecs::SnapshotResource;
use serde::{Deserialize, Serialize};

/// Game score resource. Implements SnapshotResource so warm-restart reload
/// (ADR-020) restores its value across a dylib swap.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct GameScore {
    pub value: u32,
}

impl SnapshotResource for GameScore {
    const NAME: &'static str = "GameScore";
}
