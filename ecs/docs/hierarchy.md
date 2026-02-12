# Parent-Child Hierarchy

The hierarchy system manages parent-child relationships between entities using `Parent` and `Children` components. It maintains consistency between both components automatically.

## Components

- **`Parent(Entity)`** — Stored on child entities, pointing to their parent.
- **`Children(Vec<Entity>)`** — Stored on parent entities, listing their children.

Both components are always consistent: if entity A has `Parent(B)`, then entity B's `Children` list contains A.

## Setting Up

Register hierarchy components (included in `register_std_components`):

```rust
world.register_component::<Parent>();
world.register_component::<Children>();

// Or use the standard registration:
register_std_components(&mut world);
```

## Basic Operations

### Set Parent

```rust
use redlilium_ecs::set_parent;

let mut world = World::new();
register_std_components(&mut world);

let parent = world.spawn();
let child_a = world.spawn();
let child_b = world.spawn();

set_parent(&mut world, child_a, parent);
set_parent(&mut world, child_b, parent);

// Verify
assert_eq!(world.get::<Parent>(child_a), Some(&Parent(parent)));
let children = world.get::<Children>(parent).unwrap();
assert_eq!(children.len(), 2);
```

### Reparent

Setting a new parent automatically removes the entity from the old parent's children:

```rust
let parent_a = world.spawn();
let parent_b = world.spawn();
let child = world.spawn();

set_parent(&mut world, child, parent_a);
set_parent(&mut world, child, parent_b); // Reparent

// child is now under parent_b
assert_eq!(world.get::<Parent>(child), Some(&Parent(parent_b)));

// parent_a has no children
let children_a = world.get::<Children>(parent_a).unwrap();
assert!(children_a.is_empty());
```

### Remove Parent

```rust
use redlilium_ecs::remove_parent;

set_parent(&mut world, child, parent);
remove_parent(&mut world, child);

// child has no parent
assert!(world.get::<Parent>(child).is_none());
// parent's children list is empty
assert!(world.get::<Children>(parent).unwrap().is_empty());
```

### Despawn Recursive

Despawns an entity and all its descendants depth-first:

```rust
use redlilium_ecs::despawn_recursive;

let root = world.spawn();
let child = world.spawn();
let grandchild = world.spawn();

set_parent(&mut world, child, root);
set_parent(&mut world, grandchild, child);

despawn_recursive(&mut world, root);

// All three entities are gone
assert!(!world.is_alive(root));
assert!(!world.is_alive(child));
assert!(!world.is_alive(grandchild));
```

Despawning a mid-hierarchy entity removes it from its parent:

```rust
set_parent(&mut world, child, parent);
set_parent(&mut world, grandchild, child);

despawn_recursive(&mut world, child);

// parent is still alive, child and grandchild are gone
assert!(world.is_alive(parent));
assert!(!world.is_alive(child));
assert!(!world.is_alive(grandchild));

// parent's children list no longer contains child
assert!(world.get::<Children>(parent).unwrap().is_empty());
```

## Deferred Hierarchy Commands

From within systems, use the `HierarchyCommands` trait on `CommandBuffer`:

```rust
use redlilium_ecs::HierarchyCommands;

struct ParentingSystem;

impl System for ParentingSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Res<CommandBuffer>, Read<NewChild>)>()
            .execute(|(commands, new_children)| {
                for (idx, nc) in new_children.iter() {
                    commands.cmd_set_parent(nc.child, nc.parent);
                }
            }).await;
    }
}
```

### HierarchyCommands API

| Method | Description |
|--------|-------------|
| `commands.cmd_set_parent(child, parent)` | Deferred `set_parent` |
| `commands.cmd_remove_parent(child)` | Deferred `remove_parent` |
| `commands.cmd_despawn_recursive(entity)` | Deferred `despawn_recursive` |

## Safety Checks

- **Self-parenting panics**: `set_parent(&mut world, entity, entity)` will panic.
- **Idempotent**: Setting the same parent twice is a no-op.
- **No-op on missing parent**: `remove_parent` does nothing if the entity has no parent.

## Public API

| Function | Description |
|----------|-------------|
| `set_parent(world, child, parent)` | Establish parent-child relationship |
| `remove_parent(world, child)` | Remove parent from child |
| `despawn_recursive(world, entity)` | Despawn entity + all descendants |
