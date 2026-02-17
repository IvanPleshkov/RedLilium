# Hierarchy

RedLilium supports parent-child entity hierarchies through the `Parent` and `Children` components. These are managed by dedicated functions that keep both sides in sync.

## Establishing Relationships

```rust
let parent = world.spawn_with((
    Transform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
    GlobalTransform::IDENTITY,
    Name::new("Parent"),
));

let child = world.spawn_with((
    Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
    GlobalTransform::IDENTITY,
    Name::new("Child"),
));

// Make child a child of parent
set_parent(&mut world, child, parent);
```

After this call:
- `child` has a `Parent(parent)` component
- `parent` has a `Children(vec![child])` component

## Removing Relationships

```rust
// Remove child from its parent
remove_parent(&mut world, child);
```

This removes the `Parent` component from `child` and updates the parent's `Children` list.

## Recursive Operations

### Despawn with Children

```rust
// Despawn parent and all descendants depth-first
despawn_recursive(&mut world, parent);
```

### Enable / Disable

Enable and disable propagate through the hierarchy:

```rust
// Disable entity and all children (adds Disabled component)
disable(&mut world, entity);

// Re-enable entity and all children (removes Disabled component)
enable(&mut world, entity);
```

Disabled entities are excluded from standard queries (see [Components - Disabled Entities](./components.md#disabled-entities)).

## Deferred Hierarchy Commands

Inside systems, use deferred commands:

```rust
impl System for ParentingSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.commands(|world| {
            set_parent(world, child, parent);
            remove_parent(world, other_child);
            despawn_recursive(world, old_tree);
        });
        Ok(())
    }
}
```

Or via the `CommandCollector`:

```rust
// Using HierarchyCommands extension
commands.set_parent(child, parent);
commands.remove_parent(child);
commands.despawn_recursive(entity);
```

## Reading Hierarchy Data

```rust
ctx.lock::<(Read<Parent>, Read<Children>)>()
    .execute(|(parents, children)| {
        // Get entity's parent
        if let Some(parent) = parents.get(entity) {
            println!("Parent: {}", parent.0);
        }

        // Iterate entity's children
        if let Some(kids) = children.get(entity) {
            for &child in kids.iter() {
                println!("Child: {}", child);
            }
        }
    });
```

## Transform Propagation

The built-in `UpdateGlobalTransforms` system propagates `Transform` down the hierarchy to compute `GlobalTransform` (world-space matrices):

```rust
schedules.get_mut::<PostUpdate>().add(UpdateGlobalTransforms);
```

This system reads `Transform`, `Parent`, and `Children`, and writes `GlobalTransform` for every entity in the hierarchy. It must run after any system that modifies `Transform` or the hierarchy structure.
