use redlilium_ecs::SnapshotResource;
use serde::{Deserialize, Serialize};

/// Game score resource. Captures pre-play value and restores on Stop.
/// No PlayModeAware hooks needed — snapshot restore handles state reset.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct GameScore {
    pub value: u32,
}

impl SnapshotResource for GameScore {
    const NAME: &'static str = "GameScore";
}
