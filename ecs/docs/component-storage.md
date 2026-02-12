# Component Storage (Sparse Sets)

Components are stored in **sparse sets** — a data structure optimized for O(1) insert, remove, get, and cache-friendly iteration.

## How Sparse Sets Work

Each component type `T` has its own `SparseSetInner<T>` consisting of:

- **Sparse array**: Maps `entity_index → dense_index`. If `None`, the entity doesn't have this component.
- **Dense array**: Contiguous storage of component values for efficient iteration.
- **Entities array**: Parallel to dense, stores which entity owns each element.
- **Ticks arrays**: Parallel to dense, tracks when each component was added and last changed.

```
Sparse:  [Some(0), None, Some(1), None, None, Some(2)]
Dense:   [CompA, CompC, CompF]
Entities:[0,      2,     5     ]
```

## Complexity

| Operation | Time |
|-----------|------|
| Insert | O(1) amortized |
| Remove | O(1) (swap-remove) |
| Get by entity | O(1) (sparse lookup) |
| Contains check | O(1) |
| Iterate all | O(n) where n = number of components stored |

## Type-Erased Storage

Internally, `World` stores components as `ComponentStorage` — a type-erased wrapper around `SparseSetInner<T>`. Each storage includes:

- A `Box<dyn Any + Send + Sync>` for the actual sparse set
- A per-storage `RwLock<()>` for thread-safe access
- Type-erased function pointers for `remove`, `contains`, `changed_since`, `added_since`

## Borrow Guards: Ref and RefMut

When accessing component storages through the World, you receive RAII guard types:

- **`Ref<'a, T>`** — Shared read access. Holds a `RwLockReadGuard`. Dereferences to `SparseSetInner<T>`.
- **`RefMut<'a, T>`** — Exclusive write access. Holds a `RwLockWriteGuard`. Dereferences to `SparseSetInner<T>`.

Locks are released automatically when the guard is dropped.

## Direct Usage

While systems typically access components through the lock-execute pattern, you can also use the World API directly:

```rust
let mut world = World::new();
world.register_component::<Position>();

let entity = world.spawn();
world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

// Read access — multiple simultaneous readers allowed
let positions = world.read::<Position>().unwrap();
assert_eq!(positions.len(), 1);

for (entity_idx, pos) in positions.iter() {
    println!("Entity {} at ({}, {})", entity_idx, pos.x, pos.y);
}

// Single entity lookup
let pos = positions.get(entity.index());
assert_eq!(pos, Some(&Position { x: 1.0, y: 2.0 }));
drop(positions);

// Write access — exclusive, panics if any read is held
let mut positions = world.write::<Position>().unwrap();
for (_, pos) in positions.iter_mut() {
    pos.x += 10.0;
}
```

## Change Detection

Each component tracks two ticks (parallel to the dense array):

- **`ticks_added`**: The world tick when the component was first inserted.
- **`ticks_changed`**: The world tick when the component was last modified.

```rust
let mut set = SparseSetInner::<u32>::new();

// Insert with tick tracking
set.insert_with_tick(entity_idx, 42, 10);
assert!(set.added_since(entity_idx, 9));     // Added after tick 9
assert!(!set.added_since(entity_idx, 10));    // Not after tick 10 (equal)

// Mutate with tick tracking
*set.get_mut_tracked(entity_idx, 25).unwrap() = 99;
assert!(set.changed_since(entity_idx, 24));   // Changed after tick 24
assert!(!set.changed_since(entity_idx, 25));   // Not after tick 25

// Iterate with change tracking (marks all as changed)
for (idx, val) in set.iter_mut_tracked(50) {
    *val += 1;
}
```

## Thread Safety

Multiple `Ref<T>` guards can coexist (shared reads). A `RefMut<T>` requires exclusive access — acquiring it while any other guard exists will **panic immediately** (not deadlock):

```rust
let _r = world.read::<Position>().unwrap();
// This panics: "Cannot borrow `Position` mutably: already borrowed"
// let _w = world.write::<Position>().unwrap();
```

## Public API (SparseSetInner<T>)

| Method | Description |
|--------|-------------|
| `insert(entity_idx, value)` | Insert without tick tracking |
| `insert_with_tick(entity_idx, value, tick)` | Insert with change detection |
| `remove(entity_idx)` | Remove, returns `Option<T>` |
| `get(entity_idx)` | Shared reference to component |
| `get_mut(entity_idx)` | Mutable reference (no tick update) |
| `get_mut_tracked(entity_idx, tick)` | Mutable reference, marks changed |
| `contains(entity_idx)` | Check existence |
| `len()` / `is_empty()` | Count stored components |
| `iter()` | Iterate `(entity_idx, &T)` pairs |
| `iter_mut()` | Iterate `(entity_idx, &mut T)` pairs |
| `iter_mut_tracked(tick)` | Iterate mutably, mark all changed |
| `entities()` | Slice of entity indices in dense order |
| `changed_since(entity_idx, tick)` | Check if changed after tick |
| `added_since(entity_idx, tick)` | Check if added after tick |
