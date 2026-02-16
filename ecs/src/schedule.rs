//! Named schedules for multi-phase ECS execution.
//!
//! Schedules are independent system graphs that run at different points
//! in the application lifecycle. Instead of one flat list of systems,
//! you have named phases with clear execution order.
//!
//! # Built-in Schedules
//!
//! | Label | When it runs |
//! |-------|-------------|
//! | [`Startup`] | Once, on the first call to [`Schedules::run_startup`] |
//! | [`PreUpdate`] | Every frame, before `Update`. State transitions happen here. |
//! | [`FixedUpdate`] | At a fixed timestep (default 1/60s), may run 0..N times per frame |
//! | [`Update`] | Every frame, main game logic |
//! | [`PostUpdate`] | Every frame, after `Update` |
//!
//! # Quick Start
//!
//! ```ignore
//! use redlilium_ecs::*;
//!
//! let mut world = World::new();
//! let mut schedules = Schedules::new();
//! let runner = EcsRunner::single_thread();
//!
//! // Add systems to different schedules
//! schedules.get_mut::<Update>().add(MovementSystem);
//! schedules.get_mut::<FixedUpdate>().add(PhysicsSystem);
//! schedules.get_mut::<Startup>().add_exclusive(LoadAssetsSystem);
//!
//! // Run once at startup
//! schedules.run_startup(&mut world, &runner);
//!
//! // Each frame:
//! schedules.run_frame(&mut world, &runner, delta_time);
//! ```
//!
//! # State-Driven Schedules
//!
//! ```ignore
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! enum GameState { Menu, Playing }
//! impl States for GameState {}
//!
//! schedules.init_state(&mut world, GameState::Menu);
//!
//! // Systems that run once when entering Playing
//! schedules.on_enter(GameState::Playing).add(SetupLevel);
//!
//! // Systems that run once when exiting Playing
//! schedules.on_exit(GameState::Playing).add(CleanupLevel);
//! ```

use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::runner::EcsRunner;
use crate::state::{ApplyStateTransition, NextState, State, StateTransition, States};
use crate::systems_container::SystemsContainer;
use crate::world::World;

// ---------------------------------------------------------------------------
// ScheduleLabel trait + built-in labels
// ---------------------------------------------------------------------------

/// Marker trait for schedule label types.
///
/// Each label identifies a named schedule (an independent system graph).
/// Built-in labels: [`Startup`], [`PreUpdate`], [`FixedUpdate`],
/// [`Update`], [`PostUpdate`].
///
/// # Custom Labels
///
/// ```ignore
/// struct MyCustomPhase;
/// impl ScheduleLabel for MyCustomPhase {}
///
/// schedules.get_mut::<MyCustomPhase>().add(MySystem);
/// schedules.run_schedule::<MyCustomPhase>(&mut world, &runner);
/// ```
pub trait ScheduleLabel: 'static {}

/// Runs once on the first call to [`Schedules::run_startup`].
pub struct Startup;
impl ScheduleLabel for Startup {}

/// Runs every frame before [`Update`]. State transitions are processed here.
pub struct PreUpdate;
impl ScheduleLabel for PreUpdate {}

/// Runs at a fixed timestep (default 1/60s). May execute 0..N times per frame.
pub struct FixedUpdate;
impl ScheduleLabel for FixedUpdate {}

/// Runs every frame. Main game logic schedule.
pub struct Update;
impl ScheduleLabel for Update {}

/// Runs every frame after [`Update`].
pub struct PostUpdate;
impl ScheduleLabel for PostUpdate {}

// ---------------------------------------------------------------------------
// ScheduleId
// ---------------------------------------------------------------------------

/// Identifies a schedule in the [`Schedules`] map.
///
/// Most schedules use `Type(TypeId)` keying. State-driven schedules use
/// `OnEnter` / `OnExit` with the state type and variant hash.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ScheduleId {
    /// Type-based label (Startup, Update, custom labels, etc.)
    Type(TypeId),
    /// State-enter schedule for a specific variant.
    OnEnter(TypeId, u64),
    /// State-exit schedule for a specific variant.
    OnExit(TypeId, u64),
}

impl ScheduleId {
    /// Creates a type-based schedule ID from a [`ScheduleLabel`].
    pub fn of<L: ScheduleLabel>() -> Self {
        Self::Type(TypeId::of::<L>())
    }
}

// ---------------------------------------------------------------------------
// Time resource
// ---------------------------------------------------------------------------

/// Time resource providing frame timing information to systems.
///
/// Inserted into the [`World`] automatically by [`Schedules::run_frame`].
///
/// During [`FixedUpdate`] execution, [`delta()`](Time::delta) returns the
/// fixed timestep value. During all other schedules, it returns the real
/// frame delta.
pub struct Time {
    /// Effective delta — frame delta during Update, fixed delta during FixedUpdate.
    delta: f64,
    /// Always the real frame delta time.
    frame_delta: f64,
    /// Total elapsed time since the first `run_frame` call.
    elapsed: f64,
    /// The configured fixed timestep interval.
    fixed_delta: f64,
}

impl Time {
    /// Creates a new Time resource with the given fixed timestep.
    fn new(fixed_delta: f64) -> Self {
        Self {
            delta: 0.0,
            frame_delta: 0.0,
            elapsed: 0.0,
            fixed_delta,
        }
    }

    /// Returns the current effective delta time in seconds.
    ///
    /// During [`FixedUpdate`], this returns the fixed timestep.
    /// During all other schedules, this returns the real frame delta.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Returns the current effective delta time as `f32`.
    pub fn delta_f32(&self) -> f32 {
        self.delta as f32
    }

    /// Returns the real frame delta time, regardless of which schedule
    /// is currently executing.
    pub fn frame_delta(&self) -> f64 {
        self.frame_delta
    }

    /// Returns the total elapsed time since the first frame.
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }

    /// Returns the total elapsed time as `f32`.
    pub fn elapsed_f32(&self) -> f32 {
        self.elapsed as f32
    }

    /// Returns the configured fixed timestep interval.
    pub fn fixed_delta(&self) -> f64 {
        self.fixed_delta
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new(1.0 / 60.0)
    }
}

// ---------------------------------------------------------------------------
// State checker (type-erased)
// ---------------------------------------------------------------------------

/// Signature for a type-erased state transition check function.
type StateCheckFn = Box<dyn Fn(&World) -> Option<(u64, u64)> + Send + Sync>;

/// Type-erased state transition checker.
///
/// Each registered state type gets one checker that inspects
/// `StateTransition<S>` and returns variant hashes on transition.
struct StateChecker {
    /// TypeId of the `States` enum.
    state_type_id: TypeId,
    /// Returns `Some((exited_hash, entered_hash))` if a transition occurred.
    check: StateCheckFn,
}

/// Compute a deterministic hash for a state variant.
fn hash_state<S: Hash>(state: &S) -> u64 {
    let mut hasher = DefaultHasher::new();
    state.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------

/// Multi-schedule orchestrator for the ECS.
///
/// Manages named [`SystemsContainer`]s, state-driven schedule triggers,
/// a fixed-timestep accumulator, and a [`Time`] resource.
///
/// # Execution Order
///
/// [`run_frame()`](Schedules::run_frame) executes schedules in this order:
///
/// 1. **PreUpdate** — state transitions processed here
/// 2. **OnExit / OnEnter** — if a state transition occurred
/// 3. **FixedUpdate** — 0..N iterations based on accumulated time
/// 4. **Update** — main game logic
/// 5. **PostUpdate** — cleanup, transform propagation
pub struct Schedules {
    schedules: HashMap<ScheduleId, SystemsContainer>,
    state_checkers: Vec<StateChecker>,
    fixed_timestep: f64,
    fixed_accumulator: f64,
    startup_done: bool,
}

impl Schedules {
    /// Creates a new empty schedule orchestrator.
    ///
    /// Fixed timestep defaults to 1/60 seconds. Change with
    /// [`set_fixed_timestep()`](Schedules::set_fixed_timestep).
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
            state_checkers: Vec::new(),
            fixed_timestep: 1.0 / 60.0,
            fixed_accumulator: 0.0,
            startup_done: false,
        }
    }

    /// Returns a mutable reference to the schedule for the given label,
    /// creating it if it doesn't exist.
    pub fn get_mut<L: ScheduleLabel>(&mut self) -> &mut SystemsContainer {
        self.schedules
            .entry(ScheduleId::of::<L>())
            .or_default()
    }

    /// Returns a reference to the schedule for the given label, if it exists.
    pub fn get<L: ScheduleLabel>(&self) -> Option<&SystemsContainer> {
        self.schedules.get(&ScheduleId::of::<L>())
    }

    /// Returns a mutable reference to the `OnEnter` schedule for a specific
    /// state variant, creating it if it doesn't exist.
    ///
    /// Systems added here run once when the state transitions *into* this variant.
    pub fn on_enter<S: States>(&mut self, state: S) -> &mut SystemsContainer {
        let id = ScheduleId::OnEnter(TypeId::of::<S>(), hash_state(&state));
        self.schedules
            .entry(id)
            .or_default()
    }

    /// Returns a mutable reference to the `OnExit` schedule for a specific
    /// state variant, creating it if it doesn't exist.
    ///
    /// Systems added here run once when the state transitions *out of* this variant.
    pub fn on_exit<S: States>(&mut self, state: S) -> &mut SystemsContainer {
        let id = ScheduleId::OnExit(TypeId::of::<S>(), hash_state(&state));
        self.schedules
            .entry(id)
            .or_default()
    }

    /// Registers a state type with the schedule orchestrator.
    ///
    /// This:
    /// 1. Inserts [`State<S>`], [`NextState<S>`], [`StateTransition<S>`]
    ///    resources into the world
    /// 2. Adds [`ApplyStateTransition<S>`] to the [`PreUpdate`] schedule
    /// 3. Registers a type-erased checker so [`run_frame`](Schedules::run_frame)
    ///    can trigger `OnEnter` / `OnExit` schedules on transitions
    pub fn init_state<S: States>(&mut self, world: &mut World, initial: S) {
        world.insert_resource(State::new(initial));
        world.insert_resource(NextState::<S>::default());
        world.insert_resource(StateTransition::<S>::default());

        self.get_mut::<PreUpdate>()
            .add_exclusive(ApplyStateTransition::<S>::new());

        self.state_checkers.push(StateChecker {
            state_type_id: TypeId::of::<S>(),
            check: Box::new(|world: &World| {
                let transition = world.resource::<StateTransition<S>>();
                transition
                    .get()
                    .map(|(exited, entered)| (hash_state(exited), hash_state(entered)))
            }),
        });
    }

    /// Sets the fixed timestep interval for [`FixedUpdate`].
    ///
    /// Default is 1/60 seconds. The accumulator is reset when changed.
    pub fn set_fixed_timestep(&mut self, dt: f64) {
        assert!(dt > 0.0, "Fixed timestep must be positive");
        self.fixed_timestep = dt;
        self.fixed_accumulator = 0.0;
    }

    /// Runs the [`Startup`] schedule once.
    ///
    /// Subsequent calls are no-ops. Call this before your main loop.
    pub fn run_startup(&mut self, world: &mut World, runner: &EcsRunner) {
        if self.startup_done {
            return;
        }
        self.startup_done = true;

        // Ensure Time resource exists.
        if !world.has_resource::<Time>() {
            world.insert_resource(Time::new(self.fixed_timestep));
        }

        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<Startup>()) {
            runner.run(world, schedule);
        }
    }

    /// Runs a complete frame tick.
    ///
    /// Execution order:
    /// 1. Update [`Time`] resource
    /// 2. Run [`PreUpdate`] (state transitions happen here)
    /// 3. Check state transitions → run `OnExit` / `OnEnter` schedules
    /// 4. Run [`FixedUpdate`] (accumulator loop)
    /// 5. Run [`Update`]
    /// 6. Run [`PostUpdate`]
    pub fn run_frame(&mut self, world: &mut World, runner: &EcsRunner, delta_time: f64) {
        // 1. Update Time resource
        if !world.has_resource::<Time>() {
            world.insert_resource(Time::new(self.fixed_timestep));
        }
        {
            let mut time = world.resource_mut::<Time>();
            time.delta = delta_time;
            time.frame_delta = delta_time;
            time.elapsed += delta_time;
            time.fixed_delta = self.fixed_timestep;
        }

        // 2. Run PreUpdate
        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<PreUpdate>()) {
            runner.run(world, schedule);
        }

        // 3. Check state transitions and run OnExit / OnEnter
        self.run_state_transitions(world, runner);

        // 4. FixedUpdate accumulator
        self.fixed_accumulator += delta_time;
        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<FixedUpdate>()) {
            while self.fixed_accumulator >= self.fixed_timestep {
                // Set effective delta to fixed timestep
                world.resource_mut::<Time>().delta = self.fixed_timestep;
                runner.run(world, schedule);
                self.fixed_accumulator -= self.fixed_timestep;
            }
        } else {
            // No FixedUpdate schedule registered — just drain accumulator
            // to prevent unbounded growth.
            while self.fixed_accumulator >= self.fixed_timestep {
                self.fixed_accumulator -= self.fixed_timestep;
            }
        }

        // Restore effective delta to frame delta
        world.resource_mut::<Time>().delta = delta_time;

        // 5. Run Update
        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<Update>()) {
            runner.run(world, schedule);
        }

        // 6. Run PostUpdate
        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<PostUpdate>()) {
            runner.run(world, schedule);
        }
    }

    /// Runs a specific schedule by label.
    ///
    /// Does nothing if no systems have been added to this schedule.
    pub fn run_schedule<L: ScheduleLabel>(&self, world: &mut World, runner: &EcsRunner) {
        if let Some(schedule) = self.schedules.get(&ScheduleId::of::<L>()) {
            runner.run(world, schedule);
        }
    }

    /// Check all registered state types for transitions and run
    /// the corresponding OnExit / OnEnter schedules.
    fn run_state_transitions(&self, world: &mut World, runner: &EcsRunner) {
        for checker in &self.state_checkers {
            if let Some((exited_hash, entered_hash)) = (checker.check)(world) {
                // Run OnExit first
                let exit_id = ScheduleId::OnExit(checker.state_type_id, exited_hash);
                if let Some(schedule) = self.schedules.get(&exit_id) {
                    runner.run(world, schedule);
                }

                // Then OnEnter
                let enter_id = ScheduleId::OnEnter(checker.state_type_id, entered_hash);
                if let Some(schedule) = self.schedules.get(&enter_id) {
                    runner.run(world, schedule);
                }
            }
        }
    }
}

impl Default for Schedules {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{System, SystemError};
    use crate::system_context::SystemContext;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    // -- Helpers --

    struct IncrementSystem(Arc<AtomicU32>);
    impl System for IncrementSystem {
        type Result = ();
        fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // -- Tests --

    #[test]
    fn schedules_new_empty() {
        let schedules = Schedules::new();
        assert!(schedules.get::<Startup>().is_none());
        assert!(schedules.get::<Update>().is_none());
    }

    #[test]
    fn get_mut_creates_on_demand() {
        let mut schedules = Schedules::new();
        assert!(schedules.get::<Update>().is_none());
        let container = schedules.get_mut::<Update>();
        assert_eq!(container.system_count(), 0);
        assert!(schedules.get::<Update>().is_some());
    }

    #[test]
    fn run_startup_runs_once() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules
            .get_mut::<Startup>()
            .add(IncrementSystem(counter.clone()));

        schedules.run_startup(&mut world, &runner);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Second call is a no-op
        schedules.run_startup(&mut world, &runner);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn run_frame_execution_order() {
        let order = Arc::new(AtomicU32::new(0));

        struct PhaseSystem {
            expected: u32,
            next: u32,
            counter: Arc<AtomicU32>,
        }
        impl System for PhaseSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                assert_eq!(
                    self.counter.load(Ordering::SeqCst),
                    self.expected,
                    "Wrong execution order"
                );
                self.counter.store(self.next, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.get_mut::<PreUpdate>().add(PhaseSystem {
            expected: 0,
            next: 1,
            counter: order.clone(),
        });

        // Use a wrapper to give Update's system a unique type
        struct UpdatePhase(PhaseSystem);
        impl System for UpdatePhase {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.run(ctx)
            }
        }
        schedules.get_mut::<Update>().add(UpdatePhase(PhaseSystem {
            expected: 1,
            next: 2,
            counter: order.clone(),
        }));

        struct PostUpdatePhase(PhaseSystem);
        impl System for PostUpdatePhase {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.run(ctx)
            }
        }
        schedules
            .get_mut::<PostUpdate>()
            .add(PostUpdatePhase(PhaseSystem {
                expected: 2,
                next: 3,
                counter: order.clone(),
            }));

        schedules.run_frame(&mut world, &runner, 1.0 / 60.0);
        assert_eq!(order.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn fixed_update_accumulator() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.set_fixed_timestep(1.0 / 60.0);
        schedules
            .get_mut::<FixedUpdate>()
            .add(IncrementSystem(counter.clone()));

        // Delta = 2.5 fixed steps → should run 2 times
        let dt = 2.5 / 60.0;
        schedules.run_frame(&mut world, &runner, dt);
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        // Remaining 0.5 step carried over. Delta = 1.0 step → total 1.5 → runs 1 more
        let dt = 1.0 / 60.0;
        schedules.run_frame(&mut world, &runner, dt);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn fixed_update_time_delta() {
        let fixed_dt = 1.0 / 60.0;
        let frame_dt = 3.0 / 60.0; // 3 fixed steps

        struct CheckFixedDelta;
        impl System for CheckFixedDelta {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.lock::<(crate::access_set::Res<Time>,)>()
                    .execute(|(time,)| {
                        // During FixedUpdate, delta should equal fixed_delta
                        assert!(
                            (time.delta() - time.fixed_delta()).abs() < 1e-10,
                            "Expected delta={}, got delta={}",
                            time.fixed_delta(),
                            time.delta()
                        );
                    });
                Ok(())
            }
        }

        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.set_fixed_timestep(fixed_dt);
        schedules.get_mut::<FixedUpdate>().add(CheckFixedDelta);

        schedules.run_frame(&mut world, &runner, frame_dt);
    }

    #[test]
    fn time_resource_updated() {
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.run_frame(&mut world, &runner, 0.016);
        {
            let time = world.resource::<Time>();
            assert!((time.frame_delta() - 0.016).abs() < 1e-10);
            assert!((time.elapsed() - 0.016).abs() < 1e-10);
        }

        schedules.run_frame(&mut world, &runner, 0.017);
        {
            let time = world.resource::<Time>();
            assert!((time.frame_delta() - 0.017).abs() < 1e-10);
            assert!((time.elapsed() - 0.033).abs() < 1e-10);
        }
    }

    // -- State-driven tests --

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    enum GameState {
        Menu,
        Playing,
    }
    impl States for GameState {}

    #[test]
    fn state_driven_on_enter() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.init_state(&mut world, GameState::Menu);

        schedules
            .on_enter(GameState::Playing)
            .add(IncrementSystem(counter.clone()));

        // Frame 1: no transition — OnEnter should NOT run
        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Queue transition
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        // Frame 2: transition happens — OnEnter should run
        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Frame 3: no transition — OnEnter should NOT run again
        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn state_driven_on_exit() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.init_state(&mut world, GameState::Menu);

        // OnExit for Menu state
        schedules
            .on_exit(GameState::Menu)
            .add(IncrementSystem(counter.clone()));

        // Frame 1: no transition
        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Queue transition: Menu → Playing
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        // Frame 2: transition — OnExit(Menu) should fire
        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn state_driven_exit_before_enter() {
        let order = Arc::new(AtomicU32::new(0));

        struct ExitSystem(Arc<AtomicU32>);
        impl System for ExitSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                assert_eq!(self.0.load(Ordering::SeqCst), 0, "Exit should run first");
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct EnterSystem(Arc<AtomicU32>);
        impl System for EnterSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                assert_eq!(
                    self.0.load(Ordering::SeqCst),
                    1,
                    "Enter should run after exit"
                );
                self.0.store(2, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules.init_state(&mut world, GameState::Menu);

        schedules
            .on_exit(GameState::Menu)
            .add(ExitSystem(order.clone()));
        schedules
            .on_enter(GameState::Playing)
            .add(EnterSystem(order.clone()));

        // Trigger transition
        world
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);

        schedules.run_frame(&mut world, &runner, 0.016);
        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn run_schedule_manual() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        schedules
            .get_mut::<Update>()
            .add(IncrementSystem(counter.clone()));

        schedules.run_schedule::<Update>(&mut world, &runner);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Running a schedule that doesn't exist is a no-op
        schedules.run_schedule::<Startup>(&mut world, &runner);
    }

    #[test]
    fn no_fixed_update_schedule_drains_accumulator() {
        let mut world = World::new();
        let mut schedules = Schedules::new();
        let runner = EcsRunner::single_thread();

        // Don't register any FixedUpdate systems — just run frames
        // This should not panic or accumulate unbounded time
        schedules.run_frame(&mut world, &runner, 1.0); // 60 fixed steps worth
        schedules.run_frame(&mut world, &runner, 0.016);
    }
}
