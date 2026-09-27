use redlilium_ecs::{
    ComputePool, EcsRunner, ExclusiveSystem, IoRuntime, ObserverError, SourceId, System,
    SystemContext, SystemError, SystemsContainer, World, run_exclusive_system_once,
    run_system_once,
};

struct Marker;
struct Spawn;
impl System for Spawn {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.commands(|world| {
            world.spawn_with((Marker,)).unwrap();
            panic!("command failure after spawn");
        });
        ctx.commands(|world| {
            *world.resource_mut::<u32>() += 1;
        });
        Ok(())
    }
}

fn setup() -> World {
    let mut world = World::new();
    world.register_component::<Marker>();
    world.insert_resource(0u32);
    world.observe_add::<Marker>(|world, _| {
        *world.resource_mut::<u32>() += 10;
        panic!("observer failure");
    });
    world.observe_add::<Marker>(|world, _| {
        *world.resource_mut::<u32>() += 100;
    });
    world
}

fn check(error: &SystemError, command_count: usize) {
    let SystemError::DeferredEffectsFailed {
        commands,
        observers,
    } = error
    else {
        panic!("wrong error: {error}");
    };
    assert_eq!(commands.len(), command_count);
    assert_eq!(observers.len(), 1);
    assert!(
        matches!(&observers[0], ObserverError::Panicked { message, source: SourceId::HOST, file, .. }
        if message == "observer failure" && file == file!())
    );
    assert!(error.to_string().contains("observer failure"));
}

fn check_runner(runner: EcsRunner) {
    let mut world = setup();
    let mut systems = SystemsContainer::new();
    systems.add(Spawn);
    for run in 1..=2 {
        let report = runner.run_with(&mut world, &systems, &Default::default());
        assert_eq!(report.errors.len(), 1);
        check(&report.errors[0], 1);
        assert_eq!(*world.resource::<u32>(), 111 * run);
        assert_eq!(world.entity_count(), run);
    }
}

#[test]
fn single_runner_reports_both_failure_kinds_and_keeps_observers() {
    check_runner(EcsRunner::single_thread());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn multi_runner_reports_both_failure_kinds_and_keeps_observers() {
    check_runner(EcsRunner::multi_thread(2));
}

#[test]
fn run_once_keeps_command_and_observer_failures() {
    let mut world = setup();
    let io = IoRuntime::new();
    let compute = ComputePool::new(io.clone());
    let error = run_system_once(&Spawn, &mut world, &compute, &io).unwrap_err();
    check(&error, 1);
    assert_eq!(*world.resource::<u32>(), 111);
}

#[test]
fn exclusive_run_once_reports_observer_failures() {
    struct SpawnExclusive;
    impl ExclusiveSystem for SpawnExclusive {
        type Result = ();
        fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
            world.spawn_with((Marker,)).unwrap();
            Ok(())
        }
    }
    let mut world = setup();
    let error = run_exclusive_system_once(&mut SpawnExclusive, &mut world).unwrap_err();
    check(&error, 0);
    assert_eq!(*world.resource::<u32>(), 110);
}

#[test]
fn runner_reports_cascade_limit_and_does_not_replay_discarded_triggers() {
    struct Nothing;
    impl System for Nothing {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            Ok(())
        }
    }
    let mut world = World::new();
    world.register_component::<Marker>();
    world.observe_insert::<Marker>(|world, entity| {
        world.insert(entity, Marker).unwrap();
    });
    world.spawn_with((Marker,)).unwrap();
    let mut systems = SystemsContainer::new();
    systems.add(Nothing);
    let runner = EcsRunner::single_thread();
    let errors = runner.run(&mut world, &systems);
    assert!(
        matches!(&errors[..], [SystemError::DeferredEffectsFailed { commands, observers }]
        if commands.is_empty() && matches!(&observers[..], [ObserverError::CascadeLimitExceeded { iterations: 100, discarded_triggers: 1 }]))
    );
    assert!(runner.run(&mut world, &systems).is_empty());
}
