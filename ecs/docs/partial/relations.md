# Entity Relations (Partial)

## What Are Relations?

Relations are typed, first-class connections between entities. Beyond simple parent-child hierarchies, a relation system lets you express arbitrary entity-to-entity relationships — "likes", "targets", "collides-with", "member-of" — as queryable ECS data.

```rust
// flecs-style relations (not available in RedLilium)
world.entity("Alice").add_relation::<Likes>("Bob");
world.entity("Guard").add_relation::<Targets>("Player");

// Query all entities that like someone
for (entity, target) in world.query::<&Likes>() {
    println!("{:?} likes {:?}", entity, target);
}

// Graph traversal
for ancestor in world.ancestors::<ChildOf>(entity) { ... }
```

## What RedLilium Has

RedLilium implements **parent-child hierarchy** via dedicated `Parent` and `Children` components:

```rust
use redlilium_ecs::{set_parent, remove_parent, despawn_recursive};

// Establish parent-child relationship
let parent = world.spawn();
let child = world.spawn();
set_parent(&mut world, child, parent);

// Query hierarchy
let children = world.get::<Children>(parent).unwrap();
for &child_entity in children.iter() {
    // process child
}

// Remove relationship
remove_parent(&mut world, child);

// Recursive deletion
despawn_recursive(&mut world, parent); // removes parent + all descendants
```

Deferred hierarchy via commands:

```rust
commands.set_parent(child, parent);
commands.despawn_recursive(entity);
```

## Why It's Partial

| Aspect | Full Relation System | RedLilium |
|--------|---------------------|-----------|
| Relation types | Any user-defined relation type | Only `Parent`/`Children` (hard-coded) |
| Multiple relation kinds | `ChildOf`, `Likes`, `Targets`, etc. | Single parent-child relationship |
| Queryable relations | Query by relation type | No relation queries |
| Many-to-many | Entity can have multiple relations of same type | One parent only, multiple children |
| Relation data | Relations can carry payload data | No payload (just entity references) |
| Graph traversal | Depth-first, breadth-first traversal APIs | Manual recursive traversal |
| Relation events | On-add/on-remove hooks for relations | No hooks |
| Exclusive relations | At most one target per relation type | Parent is exclusive, children are not |

The parent-child hierarchy is the most common relation and covers scene graphs, UI trees, and transform propagation. However, gameplay relations (targeting, grouping, ownership, spatial partitioning) require custom component-based workarounds:

```rust
// Workaround: custom relation as a component
struct Targets(Entity);
struct OwnedBy(Entity);
struct TeamMember(Entity); // team entity

// Must manually maintain consistency on despawn
```

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **flecs** | First-class relations with `add_relation<R>(target)`, relation queries, wildcards, graph traversal, relation data, cleanup policies |
| **Bevy** | Parent-child hierarchy (like RedLilium); generic relations proposed but not yet in stable (as of 0.15) |
| **Unity DOTS** | `IBufferElementData` for entity references, `LinkedEntityGroup`; no generic relation system |
| **EnTT** | No built-in relations (manual) |
| **hecs** | No built-in relations (manual) |
| **Legion** | No built-in relations (manual) |
| **Shipyard** | No built-in relations (manual) |

Note: flecs is the clear leader here — generic relations are a core feature of its architecture. Most other ECS libraries (including Bevy) only provide parent-child hierarchy, similar to RedLilium.
