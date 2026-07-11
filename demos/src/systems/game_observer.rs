use redlilium_ecs::{
    Events, PlayModeTransition, PlayState, Res, System, SystemContext, SystemError,
};

/// Example game observer system that reacts to Play/Pause/Resume/Stop transitions.
/// Subscribes via EventCursor<PlayModeTransition> — decoupled from PlayModeAwareRegistry.
/// Safe for game plugins to extend.
pub struct GameObserverSystem {
    cursor: redlilium_ecs::EventCursor<PlayModeTransition>,
}

impl GameObserverSystem {
    pub fn new() -> Self {
        Self {
            cursor: redlilium_ecs::EventCursor::new(),
        }
    }
}

impl Default for GameObserverSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for GameObserverSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Res<Events<PlayModeTransition>>,)>()
            .execute(|(events,)| {
                for transition in events.read(&self.cursor) {
                    match (transition.from, transition.to) {
                        (PlayState::Stopped, PlayState::Playing) => {
                            log::info!("Game started (Play transition)");
                        }
                        (PlayState::Playing, PlayState::Paused) => {
                            log::info!("Game paused");
                        }
                        (PlayState::Paused, PlayState::Playing) => {
                            log::info!("Game resumed from pause");
                        }
                        (_, PlayState::Stopped) => {
                            log::info!("Game stopped (transition from {:?})", transition.from);
                        }
                        _ => {}
                    }
                }
            });
        Ok(())
    }
}
