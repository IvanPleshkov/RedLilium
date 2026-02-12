# Events

Events provide typed inter-system communication using double-buffered queues. Systems can send events during a frame, and other systems can read them during the same frame and the next.

## How Double Buffering Works

Each `Events<T>` maintains two buffers:
- **current**: Events sent this frame.
- **previous**: Events from last frame (still readable).

When `update()` is called (typically at the start of each frame):
1. Previous buffer is cleared.
2. Current buffer becomes previous.
3. New current buffer is empty.

This ensures events survive for at least one full frame, so systems that run before the sender can still read last frame's events.

```
Frame 1: send(A), send(B)     → current: [A, B],  previous: []
Frame 2: update(), send(C)    → current: [C],     previous: [A, B]
Frame 3: update()             → current: [],      previous: [C]
Frame 4: update()             → current: [],      previous: []  (all gone)
```

## Setup

```rust
// Register the event type during initialization
world.add_event::<CollisionEvent>();

// This creates an Events<CollisionEvent> resource in the world
```

## Sending Events

```rust
struct CollisionEvent {
    entity_a: Entity,
    entity_b: Entity,
    force: f32,
}

struct PhysicsSystem;

impl System for PhysicsSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(ResMut<Events<CollisionEvent>>, Read<Position>)>()
            .execute(|(mut events, positions)| {
                // Detect collisions and send events
                events.send(CollisionEvent {
                    entity_a: e1,
                    entity_b: e2,
                    force: 10.0,
                });
            }).await;
    }
}
```

## Reading Events

```rust
struct DamageSystem;

impl System for DamageSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Res<Events<CollisionEvent>>, Write<Health>)>()
            .execute(|(events, mut healths)| {
                // iter() reads both previous and current buffers
                for event in events.iter() {
                    if let Some(health) = healths.get_mut(event.entity_a.index()) {
                        health.value -= event.force;
                    }
                }
            }).await;
    }
}
```

## EventUpdateSystem

Register an `EventUpdateSystem<T>` to automatically advance the double buffer each frame:

```rust
let mut systems = SystemsContainer::new();

// Event update should run before systems that send/read events
systems.add(EventUpdateSystem::<CollisionEvent>::new());
systems.add(PhysicsSystem);
systems.add(DamageSystem);

systems.add_edges(&[
    Edge::new::<EventUpdateSystem<CollisionEvent>, PhysicsSystem>(),
    Edge::new::<PhysicsSystem, DamageSystem>(),
]).unwrap();
```

## Events API

```rust
let mut events = Events::<MyEvent>::new();

// Send events
events.send(MyEvent { value: 1 });
events.send(MyEvent { value: 2 });

// Read all events (previous + current)
for event in events.iter() {
    println!("{}", event.value);
}

// Read only current frame's events
for event in events.iter_current() {
    println!("{}", event.value);
}

// Advance the double buffer
events.update();

// Query state
events.is_empty();  // true if both buffers empty
events.len();       // total count across both buffers

// Clear everything
events.clear();
```

## Multiple Event Types

Each event type is independent:

```rust
world.add_event::<CollisionEvent>();
world.add_event::<SpawnEvent>();
world.add_event::<DamageEvent>();

// Each has its own Events<T> resource and EventUpdateSystem<T>
```

## Full Frame Lifecycle Example

```rust
// Setup
world.add_event::<CollisionEvent>();

let mut systems = SystemsContainer::new();
systems.add(EventUpdateSystem::<CollisionEvent>::new());
systems.add(PhysicsSystem);    // sends CollisionEvent
systems.add(DamageSystem);     // reads CollisionEvent
systems.add(AudioSystem);      // reads CollisionEvent for sound effects

systems.add_edges(&[
    Edge::new::<EventUpdateSystem<CollisionEvent>, PhysicsSystem>(),
    Edge::new::<PhysicsSystem, DamageSystem>(),
    Edge::new::<PhysicsSystem, AudioSystem>(),
    // DamageSystem and AudioSystem can run in parallel
]).unwrap();

// Game loop
loop {
    world.advance_tick();
    runner.run(&mut world, &systems);
}
```

## Public API

### Events<T>

| Method | Description |
|--------|-------------|
| `Events::new()` | Create empty event queue |
| `send(event)` | Add event to current buffer |
| `iter()` | Iterate previous + current buffers |
| `iter_current()` | Iterate only current buffer |
| `update()` | Clear previous, swap current → previous |
| `is_empty()` | Check if both buffers are empty |
| `len()` | Total event count |
| `clear()` | Clear both buffers |

### World

| Method | Description |
|--------|-------------|
| `world.add_event::<T>()` | Register event type (inserts `Events<T>` resource) |

### EventUpdateSystem<T>

A system that calls `events.update()` each frame. Register it before any systems that send or read events of type `T`.
