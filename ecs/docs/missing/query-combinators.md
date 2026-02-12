# Query Combinators

## What Are They?

Query combinators allow composing complex filter logic in queries — particularly **OR** (union) queries that match entities having *any* of several component sets, and **AnyOf** queries that optionally access multiple components where at least one is present.

```rust
// Bevy-style query combinators (not available in RedLilium)

// OR filter: entities with Health OR Shield
fn damage_system(query: Query<Entity, Or<(With<Health>, With<Shield>)>>) {
    for entity in &query {
        // Matches entities that have Health, Shield, or both
    }
}

// AnyOf: access whichever components exist
fn render_debug(query: Query<(Entity, AnyOf<(&Health, &Shield, &Armor)>)>) {
    for (entity, (health, shield, armor)) in &query {
        // health: Option<&Health>, shield: Option<&Shield>, armor: Option<&Armor>
        // At least one is Some
    }
}
```

### Types of Combinators

| Combinator | Meaning | Example |
|-----------|---------|---------|
| `Or<(A, B)>` | Match if A **or** B (or both) | `Or<(With<Player>, With<Enemy>)>` |
| `AnyOf<(A, B)>` | Access whichever of A, B exist (at least one) | `AnyOf<(&Health, &Shield)>` |
| `And<(A, B)>` | Match if both A **and** B (default behavior) | Usually implicit |
| `Not<A>` | Match if A is absent | Same as `Without<A>` |

### Use Cases

- **Polymorphic queries**: "All damageable entities" (have Health or Shield or Armor).
- **Rendering**: "All visible entities" (have Mesh or Sprite or Text).
- **Input handling**: "All interactable entities" (have Clickable or Draggable).
- **Physics broadphase**: "All entities with any collider type" (Sphere or Box or Capsule).

### Comparison With Current Approach

Without combinators, OR queries require multiple separate iterations:

```rust
// Without combinators — must iterate twice
for (entity, health) in health_query.iter() {
    process_damageable(entity);
}
for (entity, shield) in shield_query.iter() {
    if !health_query.contains(entity) {  // avoid double-processing
        process_damageable(entity);
    }
}
```

With combinators — single pass:

```rust
// With combinators — single iteration
for entity in query_or::<(With<Health>, With<Shield>)>().iter() {
    process_damageable(entity);
}
```

## Current Approach in RedLilium

RedLilium supports `With<T>` (AND) and `Without<T>` (NOT) filters, but no OR combinator:

```rust
ctx.lock::<(Read<Transform>,)>()
    .execute(|world, (transforms,)| {
        // AND: entities with Transform AND Health
        let filter = world.with::<Health>();
        for (entity, transform) in transforms.iter() {
            if filter.matches(entity) {
                // ...
            }
        }

        // NOT: entities with Transform but WITHOUT Health
        let filter = world.without::<Health>();
        for (entity, transform) in transforms.iter() {
            if filter.matches(entity) {
                // ...
            }
        }

        // OR: must do manually
        let has_health = world.with::<Health>();
        let has_shield = world.with::<Shield>();
        for (entity, transform) in transforms.iter() {
            if has_health.matches(entity) || has_shield.matches(entity) {
                // ...
            }
        }
    })
    .await;
```

The manual OR approach works but is verbose and doesn't compose well with the filter API.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `Or<(F1, F2, ...)>` filter combinator, `AnyOf<(C1, C2, ...)>` data access, composable with `With`/`Without` |
| **flecs** | Query DSL with `or`, `not`, `optional` operators, query builder API |
| **Unity DOTS** | `EntityQuery` with `Any` / `All` / `None` component groups in `GetEntityQuery()` |
| **EnTT** | Views support groups and can be combined manually; no built-in OR |
| **hecs** | `Or` query filter type |
| **Legion** | No built-in combinators (manual iteration) |
| **Shipyard** | No built-in combinators (manual iteration) |
