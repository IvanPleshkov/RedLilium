# Observers and Reactive

RedLilium provides three layers of reactivity for responding to component lifecycle events.

## Synchronous Hooks

Hooks fire immediately during the mutation, before control returns to the caller. They receive `&mut World` and the affected `Entity`:

```rust
// Fires when a component is added to an entity that didn't have it
world.set_on_add::<Health>(|world, entity| {
    println!("Health added to {:?}", entity);
});

// Fires on every insertion (including overwrites)
world.set_on_insert::<Health>(|world, entity| {
    println!("Health inserted on {:?}", entity);
});

// Fires before a component is overwritten (old value still readable)
world.set_on_replace::<Health>(|world, entity| {
    let old = world.get::<Health>(entity).unwrap();
    println!("Replacing health: {}", old.current);
});

// Fires before a component is removed (value still readable)
world.set_on_remove::<Health>(|world, entity| {
    let hp = world.get::<Health>(entity).unwrap();
    println!("Removing health: {}", hp.current);
});
```

Only one hook per component per event type. Setting a new hook replaces the old one.

## Deferred Observers

Observers fire after commands are applied by the runner, not during the mutation itself. Multiple observers can be registered per event:

```rust
world.observe_add::<Health>(|world, entity| {
    // Entity just got Health for the first time
    println!("Welcome, {:?}!", entity);
});

world.observe_insert::<Health>(|world, entity| {
    // Health was inserted (first time or overwrite)
});

world.observe_remove::<Health>(|world, entity| {
    // Health was removed -- the component is already gone at this point
    println!("{:?} lost their health component", entity);
});
```

Observers support cascading: if an observer's action triggers more component changes, the runner will continue flushing until all pending observers have been processed (up to 100 iterations to prevent infinite loops).

## Reactive Triggers

Triggers are double-buffered entity lists that systems can read as resources. They're the most system-friendly way to react to changes:

### Setup

```rust
world.enable_add_triggers::<Health>();
world.enable_insert_triggers::<Health>();
world.enable_remove_triggers::<Health>();
```

### Reading in Systems

```rust
impl System for HealthBarSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Res<Triggers<OnAdd<Health>>>,)>()
            .execute(|(triggers,)| {
                for &entity in triggers.iter() {
                    // Entity had Health added last tick
                    println!("Create health bar for {:?}", entity);
                }
            });
        Ok(())
    }
}
```

### Available Trigger Types

| Resource | Fires when |
|----------|------------|
| `Triggers<OnAdd<T>>` | Component `T` added for the first time |
| `Triggers<OnInsert<T>>` | Component `T` inserted (any insertion) |
| `Triggers<OnRemove<T>>` | Component `T` removed |

## Choosing the Right Layer

| Need | Use |
|------|-----|
| Immediate side effect during mutation | Synchronous hook |
| Deferred reaction after all commands | Observer |
| System-readable list of affected entities | Reactive trigger |

Hooks are simplest but limited to one per event. Observers support multiple handlers and cascading. Triggers integrate cleanly with the system scheduling model.
