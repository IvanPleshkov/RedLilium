//! Play/Pause/Resume/Stop lifecycle management.
//!
//! Controls the game loop state and manages transitions between play, pause,
//! and stop modes. Triggers [`PlayModeAware`](crate::PlayModeAware) lifecycle
//! hooks and manages visibility of editor-only entities.

use crate::{Entity, PlayModeAware, SystemError, World};

/// Event emitted when play-mode state transitions occur.
///
/// Use this in systems to react to play/pause/resume/stop events
/// (instead of polling [`PlayControl`] every frame).
#[derive(Debug, Clone, Copy)]
pub struct PlayModeTransition {
    /// Previous play state.
    pub from: PlayState,
    /// New play state.
    pub to: PlayState,
}

/// The spawn_tick captured when the game transitioned to Playing.
///
/// Used to identify and despawn entities created during play mode.
/// Entities with spawn_tick >= play_start_tick (and not marked EditorOnly)
/// are cleaned up when transitioning to Stopped.
#[derive(Debug, Clone, Copy)]
pub struct PlayStartTick(pub u64);

/// Game loop execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayState {
    /// Game is not running; editor is in control.
    Stopped,
    /// Game is running in real-time.
    Playing,
    /// Game is paused (time is frozen, can resume).
    Paused,
}

type PlayModeHookHandler = std::sync::Arc<dyn Fn(&mut World, PlayModeTransition) + Send + Sync>;

/// Registry of PlayModeAware resources, in registration order (for deterministic hook dispatch).
///
/// Insert this into the world to enable PlayModeAware lifecycle hook dispatch.
/// Resources that implement PlayModeAware should be registered via [`register_play_mode_aware`].
pub struct PlayModeAwareRegistry {
    handlers: Vec<PlayModeHookHandler>,
}

impl PlayModeAwareRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Register a PlayModeAware resource type. Must be called for each PlayModeAware resource
    /// type **after** inserting it into the world for the hooks to be dispatched.
    pub fn register<T: PlayModeAware + 'static>(&mut self) {
        self.handlers.push(std::sync::Arc::new(
            |world: &mut World, transition: PlayModeTransition| {
                if !world.has_resource::<T>() {
                    return;
                }

                let hook = match (transition.from, transition.to) {
                    (PlayState::Stopped, PlayState::Playing) => PlayModeAware::on_play_start,
                    (PlayState::Playing, PlayState::Paused) => PlayModeAware::on_pause,
                    (PlayState::Paused, PlayState::Playing) => PlayModeAware::on_resume,
                    (_, PlayState::Stopped) => PlayModeAware::on_stop,
                    _ => return,
                };

                let mut resource = world.resource_mut::<T>();
                hook(&mut *resource);
            },
        ));
    }
}

impl Default for PlayModeAwareRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Controls Play/Pause/Resume/Stop transitions.
///
/// Insert into the world and call [`play`](PlayControl::play), [`pause`](PlayControl::pause),
/// [`resume`](PlayControl::resume), or [`stop`](PlayControl::stop) to request state changes.
/// The actual transitions happen in the `ManagePlayModeTransitions` system, which
/// runs in [`PreUpdate`](crate::PreUpdate) and triggers [`PlayModeAware`](crate::PlayModeAware)
/// lifecycle hooks on each transition.
#[derive(Debug, Clone, Copy)]
pub struct PlayControl {
    current: PlayState,
    pending: Option<PlayState>,
}

impl PlayControl {
    /// Create a new PlayControl in the Stopped state.
    pub fn new() -> Self {
        Self {
            current: PlayState::Stopped,
            pending: None,
        }
    }

    /// Current play state.
    pub fn state(&self) -> PlayState {
        self.current
    }

    /// Request transition to Playing (from Stopped or Paused).
    ///
    /// If already Playing, does nothing. If Stopped, transitions to Playing
    /// and triggers `on_play_start`. If Paused, transitions to Playing and
    /// triggers `on_resume`.
    pub fn play(&mut self) {
        if self.current != PlayState::Playing {
            self.pending = Some(PlayState::Playing);
        }
    }

    /// Request transition to Paused (from Playing only).
    ///
    /// If not Playing, does nothing. Triggers `on_pause`.
    pub fn pause(&mut self) {
        if self.current == PlayState::Playing {
            self.pending = Some(PlayState::Paused);
        }
    }

    /// Request transition to Playing from Paused.
    ///
    /// Alias for [`play`](PlayControl::play); useful for resume-after-pause.
    pub fn resume(&mut self) {
        self.play();
    }

    /// Request transition to Stopped (from any state).
    ///
    /// Stops the game, resets GameTime, and triggers `on_stop`.
    pub fn stop(&mut self) {
        if self.current != PlayState::Stopped {
            self.pending = Some(PlayState::Stopped);
        }
    }

    /// Process pending transition and emit events (called by ManagePlayModeTransitions).
    /// Returns (from_state, to_state) if a transition occurred, None otherwise.
    fn consume_transition(&mut self) -> Option<(PlayState, PlayState)> {
        if let Some(new_state) = self.pending.take() {
            let old_state = self.current;
            self.current = new_state;
            Some((old_state, new_state))
        } else {
            None
        }
    }
}

/// Apply transition events and hooks to resources.
fn apply_transition_hooks(world: &mut World, from: PlayState, to: PlayState) {
    let transition = PlayModeTransition { from, to };

    // Emit transition event for PlayModeAware subscribers.
    if !world.has_resource::<crate::Events<PlayModeTransition>>() {
        world.add_event::<PlayModeTransition>();
    }
    world
        .resource_mut::<crate::Events<PlayModeTransition>>()
        .send(transition);

    // Dispatch PlayModeAware lifecycle hooks (if registry exists).
    let handlers = if world.has_resource::<PlayModeAwareRegistry>() {
        world.resource::<PlayModeAwareRegistry>().handlers.clone()
    } else {
        Vec::new()
    };
    // Registry borrow is now dropped; we can call handlers with mutable world access.
    for handler in handlers {
        handler(world, transition);
    }

    // Capture play_start_tick and reset game-schedule last_runs when transitioning to Playing.
    if to == PlayState::Playing {
        let tick = world.current_tick();
        if !world.has_resource::<PlayStartTick>() {
            world.insert_resource(PlayStartTick(tick));
        } else {
            *world.resource_mut::<PlayStartTick>() = PlayStartTick(tick);
        }
        world.resource_mut::<crate::GameTime>().reset();

        // Reset game schedules' last_run to 0 so first Update sees all components as changed.
        // This ensures proper change detection after snapshot restore.
        if world.has_resource::<crate::Schedules>() {
            world
                .resource_mut::<crate::Schedules>()
                .reset_game_schedule_last_runs(0);
        }
    }

    // Reset game-schedule last_runs when transitioning to Paused.
    // This "freezes" the game: Changed<T> queries will see no changes until resumed.
    if to == PlayState::Paused {
        let tick = world.current_tick();
        if world.has_resource::<crate::Schedules>() {
            world
                .resource_mut::<crate::Schedules>()
                .reset_game_schedule_last_runs(tick);
        }
    }

    // Clean up play-spawned entities when transitioning to Stopped.
    if to == PlayState::Stopped {
        let play_start_tick = if world.has_resource::<PlayStartTick>() {
            world.resource::<PlayStartTick>().0
        } else {
            0
        };

        let entities_to_despawn: Vec<Entity> = world
            .iter_entities()
            .filter(|entity| {
                let world_tick = world.get_entity_world_tick(*entity);
                let flags = world.get_entity_flags(*entity);
                let is_editor = flags & Entity::EDITOR != 0;
                world_tick >= play_start_tick && !is_editor
            })
            .collect();

        for entity in entities_to_despawn {
            world.despawn(entity);
        }
    }
}

impl Default for PlayControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Exclusive system that processes play-mode state transitions.
///
/// Runs in [`PreUpdate`](crate::PreUpdate) before any game systems, so all
/// [`PlayModeAware`](crate::PlayModeAware) hooks fire at a predictable time.
pub struct ManagePlayModeTransitions;

impl crate::ExclusiveSystem for ManagePlayModeTransitions {
    type Result = ();

    fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
        // Check for and apply any pending state transitions.
        let transition = {
            let mut control = world.resource_mut::<PlayControl>();
            control.consume_transition()
        };

        if let Some((from, to)) = transition {
            apply_transition_hooks(world, from, to);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExclusiveSystem;

    #[test]
    fn play_control_initial_state() {
        let control = PlayControl::new();
        assert_eq!(control.state(), PlayState::Stopped);
    }

    #[test]
    fn play_control_stopped_to_playing() {
        let mut control = PlayControl::new();
        control.play();
        assert_eq!(control.state(), PlayState::Stopped); // pending, not yet applied
        assert_eq!(control.pending, Some(PlayState::Playing));
    }

    #[test]
    fn play_control_playing_to_paused() {
        let mut control = PlayControl::new();
        control.play();
        control.pending = None; // simulate apply_transition
        control.current = PlayState::Playing;
        assert_eq!(control.state(), PlayState::Playing);

        control.pause();
        assert_eq!(control.pending, Some(PlayState::Paused));
    }

    #[test]
    fn play_control_paused_to_playing() {
        let mut control = PlayControl::new();
        control.current = PlayState::Paused;
        control.play();
        assert_eq!(control.pending, Some(PlayState::Playing));
    }

    #[test]
    fn play_control_stop_resets() {
        let mut control = PlayControl::new();
        control.current = PlayState::Playing;
        control.stop();
        assert_eq!(control.pending, Some(PlayState::Stopped));
    }

    #[test]
    fn play_control_resume_is_alias() {
        let mut control = PlayControl::new();
        control.current = PlayState::Paused;
        control.resume();
        assert_eq!(control.pending, Some(PlayState::Playing));
    }

    #[test]
    fn play_control_no_duplicate_transitions() {
        let mut control = PlayControl::new();
        control.play();
        assert_eq!(control.pending, Some(PlayState::Playing));
        control.play(); // should be ignored
        assert_eq!(control.pending, Some(PlayState::Playing)); // unchanged
    }

    #[test]
    fn entity_cleanup_spawned_during_play_are_deleted() {
        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create an entity before play
        let pre_play = world.spawn();

        // Advance tick to separate pre-play and play entities
        world.advance_tick();

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Playing);

        // Spawn an entity during play
        let during_play = world.spawn();

        // Verify both entities exist
        assert!(world.is_alive(pre_play));
        assert!(world.is_alive(during_play));

        // Transition to Stopped
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);

        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Stopped);

        // Verify pre-play entity survived, play-spawned entity was deleted
        assert!(world.is_alive(pre_play));
        assert!(!world.is_alive(during_play));
    }

    #[test]
    fn entity_cleanup_preserves_pre_play_entities() {
        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Spawn entities before play
        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();

        // Advance tick to separate pre-play and play entities
        world.advance_tick();

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Transition immediately to Stopped (no new entities spawned)
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);

        system.run(&mut world).unwrap();

        // All pre-play entities should survive
        assert!(world.is_alive(e1));
        assert!(world.is_alive(e2));
        assert!(world.is_alive(e3));
    }

    #[test]
    fn entity_cleanup_preserves_editor_entities() {
        use crate::mark_editor;

        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Spawn an entity and mark it as editor-only
        let editor_entity = world.spawn();
        mark_editor(&mut world, editor_entity);

        // Spawn a regular entity
        let regular_entity = world.spawn();

        // Transition to Stopped
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);

        system.run(&mut world).unwrap();

        // Editor entity should survive, regular entity should be deleted
        assert!(world.is_alive(editor_entity));
        assert!(!world.is_alive(regular_entity));
    }

    #[test]
    fn play_start_tick_captured_on_playing_transition() {
        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        let tick_before = world.current_tick();

        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        let captured_tick = world.resource::<PlayStartTick>().0;
        let tick_after = world.current_tick();

        // Captured tick should be between before and after (or equal to one of them)
        assert!(captured_tick >= tick_before && captured_tick <= tick_after);
    }

    #[test]
    fn changed_query_sees_spawned_entities_during_play() {
        use crate::Component;

        #[derive(Component, Debug, Clone, Copy)]
        struct Position(f32);

        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Spawn entity before play
        let entity = world.spawn();
        world.insert(entity, Position(10.0)).unwrap();
        world.advance_tick();

        // Transition to Playing (this captures a play_start_tick)
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        let play_start_tick = world.resource::<PlayStartTick>().0;

        // Spawn entity during play - it gets inserted at current tick
        let play_entity = world.spawn();
        world.insert(play_entity, Position(20.0)).unwrap();

        // Advance tick to finalize the insert
        world.advance_tick();

        // Changed query from play_start_tick-1 should see play_entity as changed
        let changed_filter = world.changed::<Position>(play_start_tick - 1);
        let changed_count = world
            .iter_entities()
            .filter(|e| changed_filter.matches(e.index()))
            .count();
        assert_eq!(
            changed_count, 1,
            "Play-spawned entity should show as changed"
        );
    }

    #[test]
    fn slow_motion_accumulator_prevents_drift() {
        use crate::GameTime;

        let mut gt = GameTime::new();
        let dt = 0.016; // 60 FPS frame
        let scale = 0.25; // Quarter speed: 0.004 per frame

        // Over many frames at quarter speed
        for _ in 0..1000 {
            gt.tick(dt, scale);
        }

        let expected = 1000.0 * 0.004; // 4.0
        let drift = (gt.elapsed() - expected).abs();

        // With fractional accumulator, drift should be minimal (< 1 microsecond)
        assert!(
            drift < 0.000001,
            "Drift {} exceeds threshold at quarter-speed; accumulated {} vs expected {}",
            drift,
            gt.elapsed(),
            expected
        );
    }
}
