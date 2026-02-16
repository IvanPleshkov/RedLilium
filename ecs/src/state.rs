//! Application state machine for the ECS.
//!
//! Provides discrete states (e.g. Menu, Playing, Paused) with automatic
//! system gating via run conditions. States are stored as resources and
//! transitions are processed once per frame by [`ApplyStateTransition`].
//!
//! # Quick Start
//!
//! ```ignore
//! use redlilium_ecs::*;
//!
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! enum GameState { Menu, Playing, Paused }
//! impl States for GameState {}
//!
//! // Generate condition types for each variant
//! state_condition!(WhenPlaying, GameState, GameState::Playing);
//! state_condition!(WhenMenu, GameState, GameState::Menu);
//! state_enter_condition!(OnEnterPlaying, GameState, GameState::Playing);
//! state_exit_condition!(OnExitPlaying, GameState, GameState::Playing);
//!
//! // Register state + transition system
//! init_state(&mut world, &mut systems, GameState::Menu).unwrap();
//!
//! // Gate systems by state
//! systems.add_condition(WhenPlaying);
//! systems.add(PlayerMovement);
//! systems.add_edge::<ApplyStateTransition<GameState>, WhenPlaying>().unwrap();
//! systems.add_edge::<WhenPlaying, PlayerMovement>().unwrap();
//! ```

use std::hash::Hash;
use std::marker::PhantomData;

use crate::system::ExclusiveSystem;
use crate::systems_container::{CycleError, SystemsContainer};
use crate::world::World;

/// Marker trait for types that can be used as application states.
///
/// Implement this for your state enum. The required bounds ensure states
/// can be compared, hashed, cloned, and safely shared across threads.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, PartialEq, Eq, Hash)]
/// enum GameState { Menu, Playing, Paused }
/// impl States for GameState {}
/// ```
pub trait States: Clone + PartialEq + Eq + Hash + Send + Sync + 'static {}

/// Resource holding the current application state.
///
/// Updated by [`ApplyStateTransition`] when a pending transition exists
/// in [`NextState`]. Read by condition systems generated via
/// [`state_condition!`].
pub struct State<S: States> {
    current: S,
}

impl<S: States> State<S> {
    /// Creates a new state resource with the given initial value.
    pub fn new(initial: S) -> Self {
        Self { current: initial }
    }

    /// Returns the current state value.
    pub fn current(&self) -> &S {
        &self.current
    }
}

/// Resource for queuing a state transition.
///
/// Set a pending transition via [`set()`](NextState::set). The transition
/// is applied by [`ApplyStateTransition`] at the start of the next frame.
pub struct NextState<S: States> {
    pending: Option<S>,
}

impl<S: States> NextState<S> {
    /// Queues a transition to the given state.
    ///
    /// Overwrites any previously queued transition. The actual transition
    /// happens when [`ApplyStateTransition`] runs (typically next frame).
    pub fn set(&mut self, state: S) {
        self.pending = Some(state);
    }

    /// Takes the pending transition, leaving `None`.
    pub fn take(&mut self) -> Option<S> {
        self.pending.take()
    }

    /// Returns whether a transition is pending.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
}

impl<S: States> Default for NextState<S> {
    fn default() -> Self {
        Self { pending: None }
    }
}

/// Resource recording the most recent state transition.
///
/// Contains `Some((exited, entered))` on the frame a transition occurred,
/// `None` otherwise. Used by condition systems generated via
/// [`state_enter_condition!`] and [`state_exit_condition!`].
pub struct StateTransition<S: States> {
    transition: Option<(S, S)>,
}

impl<S: States> StateTransition<S> {
    /// Returns the transition as `(exited, entered)` if one occurred this frame.
    pub fn get(&self) -> Option<(&S, &S)> {
        self.transition.as_ref().map(|(a, b)| (a, b))
    }

    /// Returns the state that was exited, if a transition occurred.
    pub fn exited(&self) -> Option<&S> {
        self.transition.as_ref().map(|(a, _)| a)
    }

    /// Returns the state that was entered, if a transition occurred.
    pub fn entered(&self) -> Option<&S> {
        self.transition.as_ref().map(|(_, b)| b)
    }
}

impl<S: States> Default for StateTransition<S> {
    fn default() -> Self {
        Self { transition: None }
    }
}

/// Exclusive system that processes pending state transitions.
///
/// Each frame this system:
/// 1. Checks [`NextState<S>`] for a pending transition
/// 2. If pending: updates [`State<S>`] and writes [`StateTransition<S>`]
/// 3. If not pending: clears [`StateTransition<S>`]
///
/// Register via [`init_state()`] or manually with
/// `systems.add_exclusive(ApplyStateTransition::<S>::new())`.
pub struct ApplyStateTransition<S: States> {
    _marker: PhantomData<fn() -> S>,
}

impl<S: States> ApplyStateTransition<S> {
    /// Creates a new transition system for state type `S`.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<S: States> Default for ApplyStateTransition<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: States> ExclusiveSystem for ApplyStateTransition<S> {
    type Result = ();

    fn run(&mut self, world: &mut World) -> Result<Self::Result, crate::system::SystemError> {
        let next = {
            let mut next_state = world.resource_mut::<NextState<S>>();
            next_state.take()
        };

        if let Some(next_val) = next {
            let prev_val = {
                let mut state = world.resource_mut::<State<S>>();
                let prev = state.current.clone();
                state.current = next_val.clone();
                prev
            };
            let mut transition = world.resource_mut::<StateTransition<S>>();
            transition.transition = Some((prev_val, next_val));
        } else {
            let mut transition = world.resource_mut::<StateTransition<S>>();
            transition.transition = None;
        }

        Ok(())
    }
}

/// Registers state resources and the transition system.
///
/// Inserts [`State<S>`], [`NextState<S>`], and [`StateTransition<S>`]
/// into the world, then registers [`ApplyStateTransition<S>`] as an
/// exclusive system.
///
/// # Example
///
/// ```ignore
/// init_state(&mut world, &mut systems, GameState::Menu)?;
/// ```
pub fn init_state<S: States>(
    world: &mut World,
    systems: &mut SystemsContainer,
    initial: S,
) -> Result<(), CycleError> {
    world.insert_resource(State::new(initial));
    world.insert_resource(NextState::<S>::default());
    world.insert_resource(StateTransition::<S>::default());
    systems.add_exclusive(ApplyStateTransition::<S>::new());
    Ok(())
}

/// Generates a condition system that returns `True` when the current
/// state matches a specific value.
///
/// # Syntax
///
/// ```ignore
/// state_condition!(Name, StateType, variant_expr);
/// state_condition!(pub Name, StateType, variant_expr);
/// ```
///
/// # Example
///
/// ```ignore
/// state_condition!(WhenPlaying, GameState, GameState::Playing);
/// state_condition!(pub WhenMenu, GameState, GameState::Menu);
///
/// systems.add_condition(WhenPlaying);
/// systems.add_edge::<WhenPlaying, PlayerMovement>()?;
/// ```
#[macro_export]
macro_rules! state_condition {
    ($vis:vis $name:ident, $state_type:ty, $variant:expr) => {
        $vis struct $name;

        impl $crate::System for $name {
            type Result = $crate::Condition<()>;

            fn run<'a>(
                &'a self,
                ctx: &'a $crate::SystemContext<'a>,
            ) -> ::std::result::Result<$crate::Condition<()>, $crate::SystemError> {
                Ok(ctx
                    .lock::<($crate::Res<$crate::State<$state_type>>,)>()
                    .execute(|(state,)| {
                        if *state.current() == $variant {
                            $crate::Condition::True(())
                        } else {
                            $crate::Condition::False
                        }
                    }))
            }
        }
    };
}

/// Generates a condition system that returns `True` on the frame a
/// specific state is entered.
///
/// # Syntax
///
/// ```ignore
/// state_enter_condition!(Name, StateType, variant_expr);
/// state_enter_condition!(pub Name, StateType, variant_expr);
/// ```
///
/// # Example
///
/// ```ignore
/// state_enter_condition!(OnEnterPlaying, GameState, GameState::Playing);
///
/// systems.add_condition(OnEnterPlaying);
/// systems.add(SetupLevel);
/// systems.add_edge::<ApplyStateTransition<GameState>, OnEnterPlaying>()?;
/// systems.add_edge::<OnEnterPlaying, SetupLevel>()?;
/// ```
#[macro_export]
macro_rules! state_enter_condition {
    ($vis:vis $name:ident, $state_type:ty, $variant:expr) => {
        $vis struct $name;

        impl $crate::System for $name {
            type Result = $crate::Condition<()>;

            fn run<'a>(
                &'a self,
                ctx: &'a $crate::SystemContext<'a>,
            ) -> ::std::result::Result<$crate::Condition<()>, $crate::SystemError> {
                Ok(ctx
                    .lock::<($crate::Res<$crate::StateTransition<$state_type>>,)>()
                    .execute(|(transition,)| {
                        if transition.entered() == Some(&$variant) {
                            $crate::Condition::True(())
                        } else {
                            $crate::Condition::False
                        }
                    }))
            }
        }
    };
}

/// Generates a condition system that returns `True` on the frame a
/// specific state is exited.
///
/// # Syntax
///
/// ```ignore
/// state_exit_condition!(Name, StateType, variant_expr);
/// state_exit_condition!(pub Name, StateType, variant_expr);
/// ```
///
/// # Example
///
/// ```ignore
/// state_exit_condition!(OnExitMenu, GameState, GameState::Menu);
///
/// systems.add_condition(OnExitMenu);
/// systems.add(CleanupMenu);
/// systems.add_edge::<ApplyStateTransition<GameState>, OnExitMenu>()?;
/// systems.add_edge::<OnExitMenu, CleanupMenu>()?;
/// ```
#[macro_export]
macro_rules! state_exit_condition {
    ($vis:vis $name:ident, $state_type:ty, $variant:expr) => {
        $vis struct $name;

        impl $crate::System for $name {
            type Result = $crate::Condition<()>;

            fn run<'a>(
                &'a self,
                ctx: &'a $crate::SystemContext<'a>,
            ) -> ::std::result::Result<$crate::Condition<()>, $crate::SystemError> {
                Ok(ctx
                    .lock::<($crate::Res<$crate::StateTransition<$state_type>>,)>()
                    .execute(|(transition,)| {
                        if transition.exited() == Some(&$variant) {
                            $crate::Condition::True(())
                        } else {
                            $crate::Condition::False
                        }
                    }))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ComputePool;
    use crate::io_runtime::IoRuntime;
    use crate::system::{run_exclusive_system_blocking, run_system_blocking};

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum GameState {
        Menu,
        Playing,
        Paused,
    }
    impl States for GameState {}

    // --- Resource unit tests ---

    #[test]
    fn state_new_and_current() {
        let state = State::new(GameState::Menu);
        assert_eq!(state.current(), &GameState::Menu);
    }

    #[test]
    fn next_state_default_empty() {
        let next = NextState::<GameState>::default();
        assert!(!next.is_pending());
    }

    #[test]
    fn next_state_set_and_take() {
        let mut next = NextState::<GameState>::default();
        next.set(GameState::Playing);
        assert!(next.is_pending());

        let taken = next.take();
        assert_eq!(taken, Some(GameState::Playing));
        assert!(!next.is_pending());
    }

    #[test]
    fn next_state_overwrites() {
        let mut next = NextState::<GameState>::default();
        next.set(GameState::Playing);
        next.set(GameState::Paused);
        assert_eq!(next.take(), Some(GameState::Paused));
    }

    #[test]
    fn state_transition_default_none() {
        let t = StateTransition::<GameState>::default();
        assert!(t.get().is_none());
        assert!(t.exited().is_none());
        assert!(t.entered().is_none());
    }

    #[test]
    fn state_transition_accessors() {
        let t = StateTransition {
            transition: Some((GameState::Menu, GameState::Playing)),
        };
        assert_eq!(t.get(), Some((&GameState::Menu, &GameState::Playing)));
        assert_eq!(t.exited(), Some(&GameState::Menu));
        assert_eq!(t.entered(), Some(&GameState::Playing));
    }

    // --- ApplyStateTransition tests ---

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(State::new(GameState::Menu));
        world.insert_resource(NextState::<GameState>::default());
        world.insert_resource(StateTransition::<GameState>::default());
        world
    }

    #[test]
    fn apply_transition_no_pending() {
        let mut world = setup_world();
        let mut sys = ApplyStateTransition::<GameState>::new();
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();

        assert_eq!(
            world.resource::<State<GameState>>().current(),
            &GameState::Menu
        );
        assert!(
            world
                .resource::<StateTransition<GameState>>()
                .get()
                .is_none()
        );
    }

    #[test]
    fn apply_transition_with_pending() {
        let mut world = setup_world();
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        let mut sys = ApplyStateTransition::<GameState>::new();
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();

        assert_eq!(
            world.resource::<State<GameState>>().current(),
            &GameState::Playing
        );
        let transition = world.resource::<StateTransition<GameState>>();
        assert_eq!(transition.exited(), Some(&GameState::Menu));
        assert_eq!(transition.entered(), Some(&GameState::Playing));
    }

    #[test]
    fn apply_transition_clears_on_next_frame() {
        let mut world = setup_world();
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        let mut sys = ApplyStateTransition::<GameState>::new();

        // Frame 1: transition happens
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();
        assert!(
            world
                .resource::<StateTransition<GameState>>()
                .get()
                .is_some()
        );

        // Frame 2: no pending, transition cleared
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();
        assert!(
            world
                .resource::<StateTransition<GameState>>()
                .get()
                .is_none()
        );
        // State remains Playing
        assert_eq!(
            world.resource::<State<GameState>>().current(),
            &GameState::Playing
        );
    }

    #[test]
    fn apply_transition_pending_consumed() {
        let mut world = setup_world();
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        let mut sys = ApplyStateTransition::<GameState>::new();
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();

        assert!(!world.resource::<NextState<GameState>>().is_pending());
    }

    // --- Macro-generated condition tests ---

    state_condition!(WhenPlaying, GameState, GameState::Playing);
    state_condition!(WhenMenu, GameState, GameState::Menu);
    state_enter_condition!(OnEnterPlaying, GameState, GameState::Playing);
    state_exit_condition!(OnExitMenu, GameState, GameState::Menu);

    #[test]
    fn state_condition_true_when_matching() {
        let world = setup_world();
        // State is Menu, set to Playing
        world.resource_mut::<State<GameState>>().current = GameState::Playing;

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&WhenPlaying, &world, &compute, &io).unwrap();
        assert!(result.is_true());
    }

    #[test]
    fn state_condition_false_when_not_matching() {
        let world = setup_world(); // State is Menu

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&WhenPlaying, &world, &compute, &io).unwrap();
        assert!(result.is_false());
    }

    #[test]
    fn state_condition_menu_matches() {
        let world = setup_world(); // State is Menu

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&WhenMenu, &world, &compute, &io).unwrap();
        assert!(result.is_true());
    }

    #[test]
    fn enter_condition_true_on_transition() {
        let world = setup_world();
        // Simulate transition: Menu → Playing
        world
            .resource_mut::<StateTransition<GameState>>()
            .transition = Some((GameState::Menu, GameState::Playing));

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&OnEnterPlaying, &world, &compute, &io).unwrap();
        assert!(result.is_true());
    }

    #[test]
    fn enter_condition_false_when_no_transition() {
        let world = setup_world();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&OnEnterPlaying, &world, &compute, &io).unwrap();
        assert!(result.is_false());
    }

    #[test]
    fn exit_condition_true_on_transition() {
        let world = setup_world();
        world
            .resource_mut::<StateTransition<GameState>>()
            .transition = Some((GameState::Menu, GameState::Playing));

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&OnExitMenu, &world, &compute, &io).unwrap();
        assert!(result.is_true());
    }

    #[test]
    fn exit_condition_false_when_wrong_exit() {
        let world = setup_world();
        // Transition: Playing → Paused (not exiting Menu)
        world
            .resource_mut::<StateTransition<GameState>>()
            .transition = Some((GameState::Playing, GameState::Paused));

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let result = run_system_blocking(&OnExitMenu, &world, &compute, &io).unwrap();
        assert!(result.is_false());
    }

    // --- init_state test ---

    #[test]
    fn init_state_registers_everything() {
        let mut world = World::new();
        let mut systems = SystemsContainer::new();

        init_state(&mut world, &mut systems, GameState::Menu).unwrap();

        assert_eq!(
            world.resource::<State<GameState>>().current(),
            &GameState::Menu
        );
        assert!(!world.resource::<NextState<GameState>>().is_pending());
        assert!(
            world
                .resource::<StateTransition<GameState>>()
                .get()
                .is_none()
        );
        // ApplyStateTransition is registered (1 exclusive system)
        assert_eq!(systems.system_count(), 1);
    }
}
