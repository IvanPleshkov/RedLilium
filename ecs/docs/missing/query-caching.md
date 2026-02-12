# Query Caching

## What Is It?

Query caching (or prepared queries) pre-computes which entities or archetypes match a query, so subsequent executions skip the matching step. Instead of scanning all storage each frame, a cached query maintains an up-to-date result set that is incrementally updated as entities are added, removed, or change archetypes.

```rust
// Bevy-style cached query (not available in RedLilium)
fn movement_system(mut query: Query<(&mut Transform, &Velocity)>) {
    // Query is cached — Bevy knows exactly which archetypes match
    // No per-frame scan of all storages needed
    for (mut transform, velocity) in &mut query {
        transform.translation += velocity.0;
    }
}
```

### How It Works

1. **First execution**: The query identifies matching archetypes/storages and caches this list.
2. **Archetype changes**: When a new archetype is created (entity gains/loses components), the cache is updated incrementally.
3. **Iteration**: Cached queries iterate directly over known-matching data — no redundant compatibility checks.

### Benefits

| Benefit | Description |
|---------|-------------|
| **O(1) setup** | No per-frame query resolution after first run |
| **Fewer cache misses** | Pre-computed iteration order improves memory access patterns |
| **Incremental updates** | Only new archetypes need checking, not the whole world |
| **Compile-time optimization** | Query types known at compile time enable monomorphization |

### Relevance to Sparse Set ECS

Query caching is most impactful with archetype storage (matching = finding archetypes). With sparse sets (like RedLilium), there are no archetypes to match — iteration always walks the dense array of one component and checks entity presence in others. Caching could still help by:

- Pre-computing the entity intersection of multiple component storages.
- Maintaining a sorted entity list for multi-component queries.
- Caching filter results (changed/added) across frames.

However, the benefit is smaller than in archetype-based systems.

## Current Approach in RedLilium

Queries are created fresh each lock-execute cycle:

```rust
ctx.lock::<(Read<Transform>, Read<Velocity>)>()
    .execute(|world, (transforms, velocities)| {
        // Every frame: iterate transforms.dense, check velocities.contains()
        for (entity, transform) in transforms.iter() {
            if let Some(velocity) = velocities.get(entity) {
                // process
            }
        }
    })
    .await;
```

The lock-execute pattern means query setup (lock acquisition, borrow checking) happens every invocation. There's no persistent query object across frames.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `Query<T>` system parameter with automatic archetype caching, `QueryState` for manual caching |
| **flecs** | Cached queries: `ecs_query_init()` creates a persistent query object, iterates matching tables, incremental archetype matching |
| **Unity DOTS** | `EntityQuery` with archetype chunk caching, change version filtering, cached `GetEntityQuery()` |
| **Legion** | `<T>::query()` returns cached `Query` object with archetype matching |
| **EnTT** | Groups (`entt::group`) for implicit caching via sorted component arrays |
| **hecs** | `PreparedQuery<Q>` for pre-validated queries, `QueryBorrow` for one-shot |
| **Shipyard** | No explicit query caching (views are lightweight) |
