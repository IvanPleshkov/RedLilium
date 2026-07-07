//! Play/Pause/Resume/Stop lifecycle management.
//!
//! Controls the game loop state and manages transitions between play, pause,
//! and stop modes. Triggers [`PlayModeAware`](crate::PlayModeAware) lifecycle
//! hooks and manages visibility of editor-only entities.

use crate::{SystemError, World};

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
    // Emit transition event for PlayModeAware subscribers.
    if !world.has_resource::<crate::Events<PlayModeTransition>>() {
        world.add_event::<PlayModeTransition>();
    }
    let event = PlayModeTransition { from, to };
    world
        .resource_mut::<crate::Events<PlayModeTransition>>()
        .send(event);

    // Reset GameTime when transitioning to Playing.
    if to == PlayState::Playing {
        world.resource_mut::<crate::GameTime>().reset();
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
}
