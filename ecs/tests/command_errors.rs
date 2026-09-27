use redlilium_ecs::{
    CommandBuffer, CommandCollector, ComputePool, EcsRunner, ExclusiveSystem, IoRuntime, System,
    SystemContext, SystemError, SystemsContainer, World, run_system_once,
};

struct Marker;

fn world_with_observer() -> World {
    let mut world = World::new();
    world.insert_resource(Vec::<u8>::new());
    world.register_component::<Marker>();
    world.observe_add::<Marker>(|world, _| {
        world.resource_mut::<Vec<u8>>().push(5);
    });
    world
}

struct QueueFailures;
impl System for QueueFailures {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.commands(|world| {
            world.resource_mut::<Vec<u8>>().push(1);
            panic!("first command failed");
        });
        ctx.commands(|world| world.resource_mut::<Vec<u8>>().push(2));
        ctx.commands(|_| std::panic::panic_any(42u32));
        ctx.commands(|world| {
            world.resource_mut::<Vec<u8>>().push(4);
            world.spawn_with((Marker,)).unwrap();
        });
        Ok(())
    }
}

struct CheckAfterFlush;
impl ExclusiveSystem for CheckAfterFlush {
    type Result = ();
    fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
        assert_eq!(world.resource::<Vec<u8>>().last(), Some(&4));
        world.insert_resource(true);
        Ok(())
    }
}

fn check_errors(error: &SystemError) {
    let SystemError::DeferredEffectsFailed {
        commands: errors,
        observers,
    } = error
    else {
        panic!("wrong error: {error}");
    };
    assert!(observers.is_empty());
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].message, "first command failed");
    assert_eq!(errors[1].message, "unknown panic");
    for error in errors {
        assert_eq!(error.file, file!());
        assert!(error.line > 0);
        assert!(error.column > 0);
    }
    assert!(error.to_string().contains("first command failed"));
}

fn check_runner(runner: EcsRunner, pre_exclusive: bool) {
    let mut world = world_with_observer();
    let mut systems = SystemsContainer::new();
    systems.add(QueueFailures);
    if pre_exclusive {
        systems.add_exclusive(CheckAfterFlush);
        systems
            .add_edge::<QueueFailures, CheckAfterFlush>()
            .unwrap();
    }
    // The runner and world remain usable after a command failure.
    for count in 1..=2 {
        let result = runner.run_with(&mut world, &systems, &Default::default());
        assert_eq!(result.errors.len(), 1);
        check_errors(&result.errors[0]);
        assert_eq!(&*world.resource::<Vec<u8>>(), &[1, 2, 4, 5].repeat(count));
        assert_eq!(world.entity_count(), count as u32);
        if pre_exclusive {
            assert!(*world.resource::<bool>());
        }
    }
}

#[test]
fn single_runner_contains_command_panics_at_both_flush_points() {
    for pre_exclusive in [false, true] {
        check_runner(EcsRunner::single_thread(), pre_exclusive);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn multi_runner_contains_command_panics_at_both_flush_points() {
    for pre_exclusive in [false, true] {
        check_runner(EcsRunner::multi_thread(2), pre_exclusive);
    }
}

#[test]
fn run_once_reports_all_failures_and_flushes_observers() {
    let mut world = world_with_observer();
    let io = IoRuntime::new();
    let compute = ComputePool::new(io.clone());
    let error = run_system_once(&QueueFailures, &mut world, &compute, &io).unwrap_err();
    check_errors(&error);
    assert_eq!(&*world.resource::<Vec<u8>>(), &[1, 2, 4, 5]);
}

#[test]
fn world_commands_keep_partial_changes_and_defer_new_commands() {
    let mut world = World::new();
    let line;
    {
        let commands = world.resource::<CommandBuffer>();
        line = line!() + 1;
        commands.push(|world| {
            world.insert_resource(1u32);
            world.resource::<CommandBuffer>().push(|world| {
                world.insert_resource(100u32);
            });
            panic!("partly applied");
        });
        commands.push(|world| *world.resource_mut::<u32>() += 1);
    }
    let errors = world.apply_commands();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].message, "partly applied");
    assert_eq!(errors[0].file, file!());
    assert_eq!(errors[0].line, line);
    assert_eq!(*world.resource::<u32>(), 2);
    assert_eq!(world.resource::<CommandBuffer>().len(), 1);
    assert!(world.apply_commands().is_empty());
    assert_eq!(*world.resource::<u32>(), 100);
    assert!(world.apply_commands().is_empty());
}

#[test]
fn typed_commands_report_the_enqueue_site_and_continue() {
    let mut world = World::new();
    let entity = world.spawn();
    let collector = CommandCollector::new();
    // Marker is intentionally unregistered.
    let line = line!() + 1;
    collector.insert(entity, Marker);
    collector.push(|world| {
        world.insert_resource(7u32);
    });
    let errors = collector.apply(&mut world);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].line, line);
    assert_eq!(errors[0].file, file!());
    assert!(errors[0].message.contains("Component not registered"));
    assert_eq!(*world.resource::<u32>(), 7);
    assert!(collector.apply(&mut world).is_empty());

    let buffer = CommandBuffer::new();
    let line = line!() + 1;
    buffer.spawn_entity().with(Marker).build();
    let errors = buffer.apply(&mut world);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].line, line);
    assert_eq!(world.entity_count(), 2); // The spawn preceding the failed insert persists.
}

#[test]
fn drained_commands_expose_checked_application() {
    let collector = CommandCollector::new();
    collector.push(|_| panic!("drained failure"));
    let command = collector.drain().pop().unwrap();
    let error = command.apply(&mut World::new()).unwrap_err();
    assert_eq!(error.message, "drained failure");
}
