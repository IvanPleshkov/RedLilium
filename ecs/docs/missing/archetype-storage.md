# Archetype / Table Storage

## What Is It?

Archetype storage organizes entities by their **component signature** — all entities with the same set of components are stored together in contiguous memory tables. This provides cache-optimal iteration when systems process thousands or millions of entities with the same components.

```
Archetype [Transform, Velocity, Health]:
  Transform: [T0, T1, T2, T3, ...]   ← contiguous in memory
  Velocity:  [V0, V1, V2, V3, ...]
  Health:    [H0, H1, H2, H3, ...]

Archetype [Transform, Velocity]:       ← different archetype
  Transform: [T4, T5, ...]
  Velocity:  [V4, V5, ...]
```

### Key Properties

- **Cache-friendly iteration**: Components are packed contiguously per-archetype, maximizing L1/L2 cache utilization during iteration.
- **Fast queries**: Query resolution maps to a set of matching archetypes. Iterating a query = iterating over matching table rows.
- **Structural change cost**: Adding/removing a component moves the entity to a different archetype (row move between tables). This is O(n) in the number of components on the entity.

### Comparison With Sparse Sets

| Operation | Archetype | Sparse Set |
|-----------|-----------|------------|
| Iterate all with (A, B) | Excellent — contiguous rows | Good — follows dense arrays |
| Add/remove component | Slow — table move | Fast — O(1) |
| Random access by entity | Indirect (archetype + row) | Direct (sparse array lookup) |
| Memory overhead | Compact per-archetype | Sparse array per component |
| Best for | Millions of similar entities | Thousands of diverse entities |

RedLilium uses sparse sets, which is acknowledged in the design document as a deliberate tradeoff — simpler implementation, faster structural changes, good enough iteration for typical game entity counts.

## ECS Libraries That Support This

| Library | Storage Model |
|---------|---------------|
| **Bevy** | Archetype + table hybrid (default table, optional sparse set per-component via `#[component(storage = "SparseSet")]`) |
| **flecs** | Archetype-based with component column storage |
| **Unity DOTS** | Archetype (chunk-based: 16KB chunks of same-archetype entities) |
| **hecs** | Archetype-based |
| **Legion** | Archetype-based |
| **EnTT** | Sparse set (like RedLilium), with optional grouping for iteration optimization |
| **Shipyard** | Sparse set |
