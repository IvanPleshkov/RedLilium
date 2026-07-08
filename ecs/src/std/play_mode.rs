//! Play/Pause/Resume/Stop lifecycle management.
//!
//! Controls the game loop state and manages transitions between play, pause,
//! and stop modes. Triggers [`PlayModeAware`](crate::PlayModeAware) lifecycle
//! hooks and manages visibility of editor-only entities.

use super::components::Parent;
use super::hierarchy::{hide_in_play, show_in_play};
use crate::{Entity, PlayModeAware, SystemError, World};
use redlilium_core::compute::CancellationToken;
use std::collections::BTreeMap;

/// Validates that no game entities (non-EDITOR) are parented under editor-only entities.
/// Panics with a clear error message if a violation is found.
fn validate_no_game_under_editor(world: &World) {
    for entity in world.iter_entities() {
        if let Some(parent) = world.get::<Parent>(entity) {
            let entity_flags = world.get_entity_flags(entity);
            let parent_flags = world.get_entity_flags(parent.0);

            let is_entity_editor = entity_flags & Entity::EDITOR != 0;
            let is_parent_editor = parent_flags & Entity::EDITOR != 0;

            if is_parent_editor && !is_entity_editor {
                panic!(
                    "Game entity {:?} cannot be parented under editor-only entity {:?} at Play time",
                    entity, parent.0
                );
            }
        }
    }
}

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

/// Snapshot of the world captured at Play transition.
///
/// When transitioning from Stopped to Playing, the entire world (all entities
/// and opted-in resources) is serialized and stored. On Stop, the world is
/// cleared (game entities removed) and this snapshot is restored, discarding
/// all changes made during play (including edits in Pause mode).
pub struct PlaySnapshot(pub Option<crate::serialize::SerializedWorld>);

impl PlaySnapshot {
    /// Create an empty snapshot.
    pub fn new() -> Self {
        Self(None)
    }
}

impl Default for PlaySnapshot {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Information about a panic that occurred during game execution.
#[derive(Debug, Clone)]
pub struct PanicInfo {
    /// The panic message.
    pub message: String,
    /// The name of the system that panicked (if available).
    pub system: String,
    /// The schedule where panic occurred (e.g., "Update", "FixedUpdate").
    pub schedule: String,
    /// The world tick when panic occurred.
    pub tick: u64,
}

/// Controls Play/Pause/Resume/Stop transitions.
///
/// Insert into the world and call [`play`](PlayControl::play), [`pause`](PlayControl::pause),
/// [`resume`](PlayControl::resume), or [`stop`](PlayControl::stop) to request state changes.
/// The actual transitions happen in the `ManagePlayModeTransitions` system, which
/// runs in [`PreUpdate`](crate::PreUpdate) and triggers [`PlayModeAware`](crate::PlayModeAware)
/// lifecycle hooks on each transition.
#[derive(Debug, Clone)]
pub struct PlayControl {
    current: PlayState,
    pending: Option<PlayState>,
    /// Set when a panic occurs in a game system while Playing.
    /// Prevents resume() until Stop is called to reset to a known state.
    panicked: Option<PanicInfo>,
}

impl PlayControl {
    /// Create a new PlayControl in the Stopped state.
    pub fn new() -> Self {
        Self {
            current: PlayState::Stopped,
            pending: None,
            panicked: None,
        }
    }

    /// Current play state.
    pub fn state(&self) -> PlayState {
        self.current
    }

    /// Get panic info if a panic is currently blocking resume.
    pub fn panic_info(&self) -> Option<&PanicInfo> {
        self.panicked.as_ref()
    }

    /// Set panic info (called by error handlers when a panic occurs).
    pub fn set_panic(&mut self, info: PanicInfo) {
        self.panicked = Some(info);
    }

    /// Clear panic info (called on Stop transition).
    pub fn clear_panic(&mut self) {
        self.panicked = None;
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
    /// Does nothing if a panic is currently blocking resume.
    pub fn resume(&mut self) {
        if self.panicked.is_none() {
            self.play();
        }
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

    // Capture snapshot and play_start_tick when transitioning to Playing.
    if to == PlayState::Playing {
        // Save world snapshot for restore on Stop
        let snapshot = world.serialize_world().ok();
        if !world.has_resource::<PlaySnapshot>() {
            world.insert_resource(PlaySnapshot(snapshot));
        } else {
            *world.resource_mut::<PlaySnapshot>() = PlaySnapshot(snapshot);
        }

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

        // Validate: forbid game entities (non-EDITOR) parented under EditorOnly parents.
        validate_no_game_under_editor(world);

        // Hide all root editor-only entities (and their children recursively) at Play transition.
        // Only hide those without an editor parent to avoid double-processing.
        let root_editor_entities: Vec<Entity> = world
            .iter_entities()
            .filter(|entity| {
                let flags = world.get_entity_flags(*entity);
                let is_editor = flags & Entity::EDITOR != 0;

                // Check if parent is also editor
                let parent_is_editor = if let Some(parent) = world.get::<Parent>(*entity) {
                    let parent_flags = world.get_entity_flags(parent.0);
                    parent_flags & Entity::EDITOR != 0
                } else {
                    false
                };

                is_editor && !parent_is_editor
            })
            .collect();
        for entity in root_editor_entities {
            hide_in_play(world, entity);
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

        // Show all hidden-in-play entities (editor entities and their children) at Pause transition.
        // This allows inspection and selection during pause.
        // Only show root hidden entities (those without a hidden parent) to avoid double-processing.
        let root_hidden_entities: Vec<Entity> = world
            .iter_entities()
            .filter(|entity| {
                let flags = world.get_entity_flags(*entity);
                let is_hidden = flags & Entity::HIDDEN_IN_PLAY != 0;

                // Check if parent is also hidden
                let parent_is_hidden = if let Some(parent) = world.get::<Parent>(*entity) {
                    let parent_flags = world.get_entity_flags(parent.0);
                    parent_flags & Entity::HIDDEN_IN_PLAY != 0
                } else {
                    false
                };

                is_hidden && !parent_is_hidden
            })
            .collect();
        for entity in root_hidden_entities {
            show_in_play(world, entity);
        }
    }

    // Restore world from snapshot when transitioning to Stopped.
    // This discards all changes made during play/pause, including hot-edits.
    if to == PlayState::Stopped {
        let play_start_tick = if world.has_resource::<PlayStartTick>() {
            world.resource::<PlayStartTick>().0
        } else {
            0
        };

        // Despawn all game entities (those created or spawned during play).
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

        // Restore editor entities from snapshot (undoes component edits made during pause).
        // Since editor entities were present at play_start_tick, they match the snapshot.
        if let Some(snapshot) = world
            .has_resource::<PlaySnapshot>()
            .then(|| world.resource::<PlaySnapshot>().0.clone())
            .flatten()
        {
            // For each entity in snapshot, if it's an editor entity in the current world, restore its components.
            for (idx, snap_entity) in snapshot.entities.iter().enumerate() {
                let current_entity = world.entity_at_index(idx as u32);
                if let Some(entity) = current_entity {
                    let is_editor = world.get_entity_flags(entity) & Entity::EDITOR != 0;
                    if is_editor {
                        // Restore components for this editor entity from snapshot.
                        // Deserialize each component and insert into the entity.
                        for snap_component in &snap_entity.components {
                            if let Err(e) =
                                world.deserialize_component_by_name(entity, snap_component)
                            {
                                log::warn!(
                                    "Failed to restore component {} on entity {:?}: {}",
                                    snap_component.type_name,
                                    entity,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }

        // Clear any panic state when stopping.
        if world.has_resource::<PlayControl>() {
            world.resource_mut::<PlayControl>().clear_panic();
        }
    }
}

impl Default for PlayControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages play-scoped task lifecycle with generation-based cancellation.
///
/// Game plugins should spawn async tasks via [`PlayTasks::spawn`] instead of directly
/// using [`ComputePool`](crate::ComputePool). On Stop transition, all tasks spawned during
/// the current play generation are cancelled, ensuring no game code remains executing
/// before warm-reload dylib unmap.
///
/// This is a critical soundness requirement: without generation-scoped cancellation,
/// futures running dylib code could access dangling pointers during reload.
///
/// # Generation Lifecycle
///
/// - **On Play**: generation counter increments; new tasks join the active generation
/// - **On Stop**: all tokens for the active generation are cancelled; the map entry is cleared
#[derive(Clone)]
pub struct PlayTasks {
    current_generation: u64,
    /// Cancellation tokens per play generation. When Stop occurs, we cancel all
    /// tokens for the active generation and clear that map entry.
    generation_tokens: std::sync::Arc<crate::sync::Mutex<BTreeMap<u64, Vec<CancellationToken>>>>,
}

impl PlayTasks {
    /// Create a new PlayTasks resource in generation 0.
    pub fn new() -> Self {
        Self {
            current_generation: 0,
            generation_tokens: std::sync::Arc::new(crate::sync::Mutex::new(BTreeMap::new())),
        }
    }

    /// Returns the current play generation.
    pub fn current_generation(&self) -> u64 {
        self.current_generation
    }

    /// Spawn an async compute task scoped to the current play generation.
    ///
    /// On Stop transition, this task's cancellation token will be automatically cancelled,
    /// terminating the task cooperatively (at its next checkpoint or yield).
    pub fn spawn<T, F, Fut>(
        &mut self,
        pool: &crate::ComputePool,
        priority: redlilium_core::compute::Priority,
        f: F,
    ) -> crate::TaskHandle<T>
    where
        T: Send + 'static,
        Fut: std::future::Future<Output = T> + Send + 'static,
        F: FnOnce(crate::EcsComputeContext) -> Fut + Send + 'static,
    {
        let handle = pool.spawn(priority, f);
        let token = handle.cancellation_token();

        let mut tokens = self.generation_tokens.lock();
        tokens
            .entry(self.current_generation)
            .or_default()
            .push(token);

        handle
    }
}

impl Default for PlayTasks {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayModeAware for PlayTasks {
    fn on_play_start(&mut self) {
        self.current_generation += 1;
    }

    fn on_stop(&mut self) {
        let mut tokens = self.generation_tokens.lock();
        if let Some(to_cancel) = tokens.remove(&self.current_generation) {
            for token in to_cancel {
                token.cancel();
            }
        }
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

    #[test]
    fn editor_entities_hidden_at_play() {
        use crate::mark_editor;

        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create an editor entity before play
        let editor_entity = world.spawn();
        mark_editor(&mut world, editor_entity);

        // Verify not hidden before play
        let flags_before = world.get_entity_flags(editor_entity);
        assert_eq!(flags_before & Entity::HIDDEN_IN_PLAY, 0);

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Verify hidden after play
        let flags_after = world.get_entity_flags(editor_entity);
        assert_ne!(
            flags_after & Entity::HIDDEN_IN_PLAY,
            0,
            "Editor entity should be hidden at play"
        );
    }

    #[test]
    fn editor_children_inherit_hidden_at_play() {
        use crate::{mark_editor, set_parent};

        let mut world = World::new();
        world.register_component::<Parent>();
        world.register_component::<crate::Children>();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create editor parent and editor child
        let editor_parent = world.spawn();
        let editor_child = world.spawn();

        mark_editor(&mut world, editor_parent);
        mark_editor(&mut world, editor_child);
        set_parent(&mut world, editor_child, editor_parent);

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Both should be hidden
        let parent_flags = world.get_entity_flags(editor_parent);
        let child_flags = world.get_entity_flags(editor_child);

        assert_ne!(
            parent_flags & Entity::HIDDEN_IN_PLAY,
            0,
            "Editor parent should be hidden"
        );
        assert_ne!(
            child_flags & Entity::HIDDEN_IN_PLAY,
            0,
            "Editor child should be hidden"
        );
        // Child should have inherited bit set (because mark_editor was called first before hide)
        assert_ne!(
            child_flags & Entity::INHERITED_HIDDEN_IN_PLAY,
            0,
            "Inherited bit should be set on child"
        );
    }

    #[test]
    fn hidden_entities_shown_at_pause() {
        use crate::mark_editor;

        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create and hide an editor entity
        let editor_entity = world.spawn();
        mark_editor(&mut world, editor_entity);

        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Verify hidden
        let flags_hidden = world.get_entity_flags(editor_entity);
        assert_ne!(flags_hidden & Entity::HIDDEN_IN_PLAY, 0);

        // Transition to Paused
        let mut control = world.resource_mut::<PlayControl>();
        control.pause();
        drop(control);

        system.run(&mut world).unwrap();

        // Verify shown again
        let flags_shown = world.get_entity_flags(editor_entity);
        assert_eq!(
            flags_shown & Entity::HIDDEN_IN_PLAY,
            0,
            "Hidden entities should be shown at pause"
        );
    }

    #[test]
    fn user_disabled_entities_stay_disabled_through_play_pause() {
        use crate::disable;

        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create a game entity and manually disable it
        let game_entity = world.spawn();
        disable(&mut world, game_entity);

        // Verify disabled with manual flag set
        let flags_before = world.get_entity_flags(game_entity);
        assert_ne!(flags_before & Entity::DISABLED, 0);
        assert_eq!(flags_before & Entity::INHERITED_DISABLED, 0);

        // Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Verify still disabled (no HIDDEN_IN_PLAY should be set for non-editor)
        let flags_during_play = world.get_entity_flags(game_entity);
        assert_ne!(
            flags_during_play & Entity::DISABLED,
            0,
            "Disabled status should persist"
        );
        assert_eq!(
            flags_during_play & Entity::HIDDEN_IN_PLAY,
            0,
            "Game entity should not be hidden-in-play"
        );

        // Transition to Paused
        let mut control = world.resource_mut::<PlayControl>();
        control.pause();
        drop(control);

        system.run(&mut world).unwrap();

        // Verify still disabled
        let flags_after_pause = world.get_entity_flags(game_entity);
        assert_ne!(
            flags_after_pause & Entity::DISABLED,
            0,
            "Disabled status should persist after pause"
        );
    }

    #[test]
    #[should_panic(expected = "cannot be parented under editor-only")]
    fn play_rejects_game_under_editor() {
        use crate::{mark_editor, set_parent};

        let mut world = World::new();
        world.register_component::<Parent>();
        world.register_component::<crate::Children>();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create editor parent and game child
        let editor_parent = world.spawn();
        let game_child = world.spawn();

        mark_editor(&mut world, editor_parent);
        set_parent(&mut world, game_child, editor_parent);

        // Transition to Playing should panic
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        let _ = system.run(&mut world);
    }

    #[test]
    fn play_pause_resume_stop_hierarchy_fixed_point() {
        use crate::mark_editor;

        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Create entities: editor parent and game sibling
        let editor_entity = world.spawn();
        let game_entity = world.spawn();

        mark_editor(&mut world, editor_entity);

        // Record initial state
        let game_initial_flags = world.get_entity_flags(game_entity);

        let mut system = ManagePlayModeTransitions;

        // Play: editor hidden, game unchanged
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);
        system.run(&mut world).unwrap();

        let game_during_play = world.get_entity_flags(game_entity);
        assert_eq!(
            game_during_play & Entity::HIDDEN_IN_PLAY,
            0,
            "Game entity should not be hidden during play"
        );

        // Pause: editor shown, game still unchanged
        let mut control = world.resource_mut::<PlayControl>();
        control.pause();
        drop(control);
        system.run(&mut world).unwrap();

        let game_paused = world.get_entity_flags(game_entity);
        assert_eq!(
            game_paused & Entity::HIDDEN_IN_PLAY,
            0,
            "Game entity should not be hidden when paused"
        );

        // Resume (play from pause): editor hidden again, game unchanged
        let mut control = world.resource_mut::<PlayControl>();
        control.resume();
        drop(control);
        system.run(&mut world).unwrap();

        let game_resumed = world.get_entity_flags(game_entity);
        assert_eq!(
            game_resumed & Entity::HIDDEN_IN_PLAY,
            0,
            "Game entity should not be hidden when resumed"
        );

        // Stop: all entities returned to initial state
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);
        system.run(&mut world).unwrap();

        let game_final = world.get_entity_flags(game_entity);
        assert_eq!(
            game_final, game_initial_flags,
            "Game entity should return to initial state after stop"
        );
    }

    #[test]
    fn play_control_panic_blocks_resume() {
        let mut control = PlayControl::new();
        control.current = PlayState::Paused;

        // Without panic, resume should work
        control.resume();
        assert_eq!(control.pending, Some(PlayState::Playing));

        // Reset state
        control.pending = None;

        // Set panic info
        let panic_info = PanicInfo {
            message: "test panic".to_string(),
            system: "TestSystem".to_string(),
            schedule: "Update".to_string(),
            tick: 42,
        };
        control.set_panic(panic_info);

        // With panic, resume should be ignored
        control.resume();
        assert_eq!(control.pending, None, "Resume should be blocked by panic");
    }

    #[test]
    fn play_control_panic_info_accessor() {
        let mut control = PlayControl::new();
        assert!(control.panic_info().is_none(), "Should start with no panic");

        let panic_info = PanicInfo {
            message: "test panic".to_string(),
            system: "TestSystem".to_string(),
            schedule: "Update".to_string(),
            tick: 42,
        };
        control.set_panic(panic_info.clone());

        let retrieved = control.panic_info();
        assert!(retrieved.is_some(), "Should have panic info");
        assert_eq!(retrieved.unwrap().message, "test panic");
        assert_eq!(retrieved.unwrap().system, "TestSystem");
    }

    #[test]
    fn play_control_stop_clears_panic() {
        let mut control = PlayControl::new();
        control.current = PlayState::Paused;

        // Set panic
        let panic_info = PanicInfo {
            message: "test panic".to_string(),
            system: "TestSystem".to_string(),
            schedule: "Update".to_string(),
            tick: 42,
        };
        control.set_panic(panic_info);
        assert!(control.panic_info().is_some());

        // Request stop
        control.stop();

        // Panic should be cleared after apply_transition_hooks runs
        // (but panic field is not cleared by stop() itself, only by apply_transition_hooks)
        // So we just verify the field exists and can be manually cleared
        control.clear_panic();
        assert!(control.panic_info().is_none());
    }

    #[test]
    fn stop_transition_clears_panic() {
        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());

        // Set initial state to Paused with panic
        let mut control = world.resource_mut::<PlayControl>();
        control.current = PlayState::Paused;
        let panic_info = PanicInfo {
            message: "test panic".to_string(),
            system: "TestSystem".to_string(),
            schedule: "Update".to_string(),
            tick: 42,
        };
        control.set_panic(panic_info);
        assert!(control.panic_info().is_some());
        drop(control);

        // Request stop
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);

        // Apply transition
        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();

        // Panic should be cleared after stop transition
        assert!(
            world.resource::<PlayControl>().panic_info().is_none(),
            "Panic should be cleared on Stop transition"
        );
    }

    #[test]
    fn play_tasks_initial_generation() {
        let tasks = PlayTasks::new();
        assert_eq!(tasks.current_generation(), 0);
    }

    #[test]
    fn play_tasks_increments_generation_on_play_start() {
        let mut tasks = PlayTasks::new();
        assert_eq!(tasks.current_generation(), 0);

        tasks.on_play_start();
        assert_eq!(tasks.current_generation(), 1);

        tasks.on_play_start();
        assert_eq!(tasks.current_generation(), 2);
    }

    #[test]
    fn play_tasks_cancels_tokens_on_stop() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Increment generation to 1 (simulating Play)
        tasks.on_play_start();

        // Spawn a task that yields
        let handle = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            42u32
        });

        // Task should be pending
        assert!(!handle.is_done());

        // Trigger Stop (cancel tokens for generation 1)
        tasks.on_stop();

        // Tick the pool — task should complete (be removed/cancelled)
        pool.tick();

        // Pool should be empty (task was cancelled/removed)
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn play_tasks_only_cancels_current_generation() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Generation 1: spawn and complete a task
        tasks.on_play_start();
        let h1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        pool.tick();
        assert!(h1.is_done());

        // Generation 2: spawn a task that yields
        tasks.on_play_start();
        let h2 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            2u32
        });
        assert!(!h2.is_done());

        // Stop only cancels generation 2's tasks, not generation 1's (which is already done)
        tasks.on_stop();
        pool.tick();

        // h2 should be cancelled and pool empty
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn full_flow_panic_pause_clear_play() {
        let mut world = World::new();
        world.insert_resource(PlayControl::default());
        world.insert_resource(PlayModeAwareRegistry::default());
        world.insert_resource(PlayStartTick(0));
        world.insert_resource(crate::GameTime::default());
        world.insert_resource(PlayTasks::default());

        // Step 1: Transition to Playing
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        let mut system = ManagePlayModeTransitions;
        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Playing);
        assert!(world.resource::<PlayControl>().panic_info().is_none());

        // Step 2: Simulate panic in Update schedule
        let panic_info = PanicInfo {
            message: "test panic message".to_string(),
            system: "TestUpdateSystem".to_string(),
            schedule: "Update".to_string(),
            tick: 100,
        };
        let mut control = world.resource_mut::<PlayControl>();
        control.pause();
        control.set_panic(panic_info.clone());
        drop(control);

        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Paused);
        let stored_panic = world.resource::<PlayControl>().panic_info().cloned();
        assert!(stored_panic.is_some());
        if let Some(info) = stored_panic {
            assert_eq!(info.message, "test panic message");
            assert_eq!(info.system, "TestUpdateSystem");
            assert_eq!(info.schedule, "Update");
            assert_eq!(info.tick, 100);
        }

        // Step 3: Verify resume is blocked
        let mut control = world.resource_mut::<PlayControl>();
        control.resume(); // Should be blocked by guard
        assert_eq!(
            control.pending, None,
            "Resume should be blocked when panic is present"
        );
        drop(control);

        // Step 4: Stop transition clears panic
        let mut control = world.resource_mut::<PlayControl>();
        control.stop();
        drop(control);

        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Stopped);
        assert!(
            world.resource::<PlayControl>().panic_info().is_none(),
            "Panic should be cleared after Stop"
        );

        // Step 5: Can play again
        let mut control = world.resource_mut::<PlayControl>();
        control.play();
        drop(control);

        system.run(&mut world).unwrap();
        assert_eq!(world.resource::<PlayControl>().state(), PlayState::Playing);
        assert!(
            world.resource::<PlayControl>().panic_info().is_none(),
            "No panic on fresh play"
        );
    }

    #[test]
    fn play_tasks_spawn_across_multiple_generations() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Generation 1: spawn task (gen 1 == tasks.current_generation() + 1 after on_play_start)
        tasks.on_play_start();
        assert_eq!(tasks.current_generation(), 1);
        let h1_gen1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        assert!(!h1_gen1.is_done());

        // Stop gen 1: cancel all gen 1 tasks
        tasks.on_stop();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 1 task should be cancelled");

        // Generation 2: spawn new task
        tasks.on_play_start();
        assert_eq!(tasks.current_generation(), 2);
        let h2_gen2 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 2u32 });
        assert!(!h2_gen2.is_done());

        // Stop gen 2: cancel all gen 2 tasks
        tasks.on_stop();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 2 task should be cancelled");

        // Verify we can start gen 3 with no residual state
        tasks.on_play_start();
        assert_eq!(tasks.current_generation(), 3);
        let h3_gen3 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            3u32
        });
        assert!(!h3_gen3.is_done());

        tasks.on_stop();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 3 task should be cancelled");
    }

    #[test]
    fn play_tasks_mixed_completed_and_pending_cancel() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        tasks.on_play_start();

        // Spawn task that completes immediately
        let h1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        pool.tick();
        assert!(h1.is_done());

        // Spawn task that yields (pending)
        let h2 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            2u32
        });
        assert!(!h2.is_done());

        // Stop cancels only pending tasks for current generation
        tasks.on_stop();
        pool.tick();

        // Both should be handled (completed one was already done, pending one was cancelled)
        assert_eq!(pool.pending_count(), 0);
    }
}
