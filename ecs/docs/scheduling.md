# Scheduling & Dependency Resolution

The scheduling system manages system registration, dependency ordering, and cycle detection. It determines which systems can run in parallel and which must wait for others to complete.

## SystemsContainer

The `SystemsContainer` holds all registered systems and their ordering constraints:

```rust
use redlilium_ecs::{SystemsContainer, Edge};

let mut container = SystemsContainer::new();

// Register systems
container.add(UpdateGlobalTransforms);
container.add(UpdateCameraMatrices);
container.add(RenderSystem);

// Define ordering: transforms before cameras, cameras before render
container.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>().unwrap();
container.add_edge::<UpdateCameraMatrices, RenderSystem>().unwrap();
```

## Dependency Edges

An edge `A → B` means "A must complete before B starts":

```rust
// Single edge
container.add_edge::<SystemA, SystemB>().unwrap();

// Batch of edges (all-or-nothing: if any creates a cycle, none are applied)
container.add_edges(&[
    Edge::new::<InputSystem, MovementSystem>(),
    Edge::new::<MovementSystem, PhysicsSystem>(),
    Edge::new::<PhysicsSystem, RenderSystem>(),
]).unwrap();
```

## Cycle Detection

Uses **Kahn's topological sort** algorithm. Cycles are detected and rejected:

```rust
container.add(SystemA);
container.add(SystemB);
container.add_edge::<SystemA, SystemB>().unwrap();

// This would create a cycle: A → B → A
let result = container.add_edge::<SystemB, SystemA>();
assert!(result.is_err());

// The CycleError tells you which systems are involved
if let Err(cycle) = result {
    println!("Cycle among: {:?}", cycle.involved);
}
```

Important properties of cycle detection:
- **Atomic**: Batch edges are validated together. If any edge creates a cycle, none are applied.
- **Non-destructive**: A failed `add_edge` doesn't modify the container state.
- **Idempotent**: Adding the same edge twice is harmless.

## Topological Order

The container pre-computes a topological sort order used by the single-threaded runner:

```rust
// Pre-computed sequential execution order
let order = container.single_thread_order();
// [0, 1, 2] — indices of systems in dependency-safe order
```

## Multi-Threaded Scheduling

The multi-threaded runner uses the dependency graph dynamically:

```rust
// Which systems have zero dependencies (can start immediately)?
let ready = container.ready_indices();

// When system idx completes, which systems should be notified?
let dependents = container.dependents_of(idx);

// In-degree per system (number of dependencies)
let in_degrees = container.in_degrees();
```

The runner maintains atomic counters per system. When a system completes, it decrements the counters of its dependents. When a counter reaches zero, that system is spawned on a worker thread.

## Example: Full Pipeline Setup

```rust
let mut systems = SystemsContainer::new();

// Event update (must run first)
systems.add(EventUpdateSystem::<CollisionEvent>::new());

// Input processing
systems.add(InputSystem);

// Game logic
systems.add(MovementSystem);
systems.add(AISystem);
systems.add(PhysicsSystem);

// Transform propagation
systems.add(UpdateGlobalTransforms);
systems.add(UpdateCameraMatrices);

// Define execution order
systems.add_edges(&[
    // Events first
    Edge::new::<EventUpdateSystem<CollisionEvent>, InputSystem>(),
    // Input before movement
    Edge::new::<InputSystem, MovementSystem>(),
    // AI and Movement can run in parallel (no edge between them)
    // Physics after both
    Edge::new::<MovementSystem, PhysicsSystem>(),
    Edge::new::<AISystem, PhysicsSystem>(),
    // Transforms after physics
    Edge::new::<PhysicsSystem, UpdateGlobalTransforms>(),
    Edge::new::<UpdateGlobalTransforms, UpdateCameraMatrices>(),
]).unwrap();
```

Execution graph:

```
EventUpdate → Input → Movement ─┐
                   ↘ AI ────────→ Physics → Transforms → Camera
```

Systems without edges between them (like `MovementSystem` and `AISystem`) can execute in parallel on the multi-threaded runner.

## Public API

| Method | Description |
|--------|-------------|
| `SystemsContainer::new()` | Create empty container |
| `container.add(system)` | Register a system (panics on duplicate) |
| `container.add_edge::<A, B>()` | A must complete before B |
| `container.add_edges(&[...])` | Batch edge addition (atomic) |
| `container.system_count()` | Number of registered systems |
| `container.single_thread_order()` | Pre-computed topological order |
| `container.ready_indices()` | Systems with zero dependencies |
| `container.dependents_of(idx)` | Systems that depend on idx |
| `container.in_degrees()` | Dependency count per system |
| `container.get_type_name(idx)` | System type name for debugging |
