use std::hash::{Hash, Hasher};

/// A 128-bit entity identifier with spawn tick and flag bits.
///
/// Layout: `u32 id` + `u32 flags` + `u64 spawn_tick`.
///
/// - **id**: slot index in the entity allocator
/// - **flags**: per-entity state bits (disabled, inherited-disabled, reserved)
/// - **spawn_tick**: world tick when this entity was spawned (replaces generation
///   for ABA detection — if a slot is reused, the new spawn_tick differs)
///
/// # Identity
///
/// Two entities are equal if they have the same `(id, spawn_tick)`.
/// Flags are mutable state and do **not** affect equality or hashing.
///
/// # Example
///
/// ```
/// use redlilium_ecs::Entity;
///
/// // Entities are compared by id and spawn_tick
/// // (Entity creation is handled by World)
/// ```
#[derive(Clone, Copy)]
pub struct Entity {
    id: u32,
    flags: u32,
    spawn_tick: u64,
}

impl Entity {
    /// Entity is manually disabled by user/system.
    pub const DISABLED: u32 = 1 << 0;
    /// Entity is disabled because a parent was disabled (propagated).
    pub const INHERITED_DISABLED: u32 = 1 << 1;
    /// Entity is manually marked as static (rarely-changing, e.g. terrain).
    ///
    /// Static entities are excluded from both `Read<T>` and `Write<T>` queries.
    /// Use `ReadAll<T>` to include them in read-only queries, or access them
    /// directly via exclusive systems (`&mut World`).
    pub const STATIC: u32 = 1 << 2;
    /// Entity is static because a parent was marked static (propagated).
    pub const INHERITED_STATIC: u32 = 1 << 3;
    /// Entity is an editor-only entity (camera, grid, gizmos, etc.).
    ///
    /// Editor entities are excluded from `Read<T>` and `Write<T>` queries.
    /// Use `ReadAll<T>` / `WriteAll<T>` to include them, or access them
    /// directly via exclusive systems (`&mut World`).
    pub const EDITOR: u32 = 1 << 4;
    /// Entity is an editor entity because a parent was marked as editor (propagated).
    pub const INHERITED_EDITOR: u32 = 1 << 5;

    /// Creates a new entity from an index and spawn tick. Flags default to 0.
    pub(crate) fn new(index: u32, spawn_tick: u64) -> Self {
        Self {
            id: index,
            flags: 0,
            spawn_tick,
        }
    }

    /// Creates a new entity with explicit flags (used by allocator when
    /// building entities from stored state).
    pub(crate) fn with_flags(index: u32, spawn_tick: u64, flags: u32) -> Self {
        Self {
            id: index,
            flags,
            spawn_tick,
        }
    }

    /// Returns the slot index of this entity.
    pub fn index(&self) -> u32 {
        self.id
    }

    /// Returns the spawn tick of this entity.
    pub fn spawn_tick(&self) -> u64 {
        self.spawn_tick
    }

    /// Returns the flags for this entity.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Returns `true` if the entity has the DISABLED flag set.
    pub fn is_disabled(&self) -> bool {
        self.flags & Self::DISABLED != 0
    }

    /// Returns `true` if the entity has the STATIC flag set.
    pub fn is_static(&self) -> bool {
        self.flags & Self::STATIC != 0
    }

    /// Returns `true` if the entity has the EDITOR flag set.
    pub fn is_editor(&self) -> bool {
        self.flags & Self::EDITOR != 0
    }
}

impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.spawn_tick == other.spawn_tick
    }
}

impl Eq for Entity {}

impl Hash for Entity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.spawn_tick.hash(state);
    }
}

impl std::fmt::Debug for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}@{})", self.index(), self.spawn_tick())
    }
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}@{})", self.index(), self.spawn_tick())
    }
}

/// Allocates and recycles entity IDs with spawn-tick tracking.
///
/// When an entity is despawned, its slot is added to a free list.
/// The next spawn reuses the slot with the current world tick as the
/// new spawn_tick, invalidating any old Entity handles.
pub(crate) struct EntityAllocator {
    /// Spawn tick for each slot. Index = entity index.
    spawn_ticks: Vec<u64>,
    /// Per-slot flag bits (disabled, inherited-disabled, etc.).
    pub(crate) flags: Vec<u32>,
    /// Alive flag per slot.
    alive: Vec<bool>,
    /// Free list of recyclable indices (LIFO stack).
    free_list: Vec<u32>,
    /// Total number of currently alive entities.
    count: u32,
}

impl EntityAllocator {
    /// Creates a new empty allocator.
    pub fn new() -> Self {
        Self {
            spawn_ticks: Vec::new(),
            flags: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
            count: 0,
        }
    }

    /// Allocates a new entity, reusing a recycled slot if available.
    /// `tick` is the current world tick used as the spawn_tick.
    pub fn allocate(&mut self, tick: u64) -> Entity {
        self.count += 1;

        if let Some(index) = self.free_list.pop() {
            let idx = index as usize;
            self.alive[idx] = true;
            self.spawn_ticks[idx] = tick;
            self.flags[idx] = 0;
            Entity::new(index, tick)
        } else {
            let index = self.spawn_ticks.len() as u32;
            self.spawn_ticks.push(tick);
            self.flags.push(0);
            self.alive.push(true);
            Entity::new(index, tick)
        }
    }

    /// Deallocates an entity. Returns false if already dead or spawn_tick mismatch.
    pub fn deallocate(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx >= self.alive.len()
            || !self.alive[idx]
            || self.spawn_ticks[idx] != entity.spawn_tick()
        {
            return false;
        }

        self.alive[idx] = false;
        // Increment spawn_tick so any old handles are invalidated on reuse
        self.spawn_ticks[idx] = self.spawn_ticks[idx].wrapping_add(1);
        self.flags[idx] = 0;
        self.free_list.push(entity.index());
        self.count -= 1;
        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.alive.len() && self.alive[idx] && self.spawn_ticks[idx] == entity.spawn_tick()
    }

    /// Returns the number of alive entities.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Sets flag bits on an entity slot (OR operation).
    pub fn set_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] |= bits;
    }

    /// Clears flag bits on an entity slot (AND-NOT operation).
    pub fn clear_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] &= !bits;
    }

    /// Returns the flags for a given entity slot.
    pub fn get_flags(&self, index: u32) -> u32 {
        self.flags[index as usize]
    }

    /// Allocates `count` entities at once, reusing recycled slots first.
    ///
    /// More efficient than calling [`allocate`](Self::allocate) in a loop
    /// because internal vectors are grown in bulk.
    pub fn allocate_many(&mut self, count: u32, tick: u64) -> Vec<Entity> {
        let mut entities = Vec::with_capacity(count as usize);

        // Reuse from free list first
        let reuse = count.min(self.free_list.len() as u32);
        for _ in 0..reuse {
            let index = self.free_list.pop().unwrap();
            let idx = index as usize;
            self.alive[idx] = true;
            self.spawn_ticks[idx] = tick;
            self.flags[idx] = 0;
            entities.push(Entity::new(index, tick));
        }

        // Allocate fresh slots for remainder
        let fresh = count - reuse;
        if fresh > 0 {
            let start = self.spawn_ticks.len() as u32;
            self.spawn_ticks
                .resize(self.spawn_ticks.len() + fresh as usize, tick);
            self.flags.resize(self.flags.len() + fresh as usize, 0);
            self.alive.resize(self.alive.len() + fresh as usize, true);
            for i in 0..fresh {
                entities.push(Entity::new(start + i, tick));
            }
        }

        self.count += count;
        entities
    }

    /// Returns the alive entity at the given index, or `None` if the slot is
    /// empty or has been recycled.
    pub fn entity_at_index(&self, index: u32) -> Option<Entity> {
        let idx = index as usize;
        if idx < self.alive.len() && self.alive[idx] {
            Some(Entity::with_flags(
                index,
                self.spawn_ticks[idx],
                self.flags[idx],
            ))
        } else {
            None
        }
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter(|(_, alive)| **alive)
            .map(|(idx, _)| Entity::with_flags(idx as u32, self.spawn_ticks[idx], self.flags[idx]))
    }
}

impl Default for EntityAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_sequential() {
        let mut alloc = EntityAllocator::new();
        let e0 = alloc.allocate(100);
        let e1 = alloc.allocate(100);
        let e2 = alloc.allocate(100);

        assert_eq!(e0.index(), 0);
        assert_eq!(e1.index(), 1);
        assert_eq!(e2.index(), 2);
        assert_eq!(e0.spawn_tick(), 100);
        assert_eq!(e1.spawn_tick(), 100);
        assert_eq!(e2.spawn_tick(), 100);
    }

    #[test]
    fn is_alive_after_allocate() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate(1);
        assert!(alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_makes_dead() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate(1);
        assert!(alloc.deallocate(entity));
        assert!(!alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_stale_entity() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate(1);
        assert!(alloc.deallocate(entity));
        // Deallocating again returns false
        assert!(!alloc.deallocate(entity));
    }

    #[test]
    fn recycled_slot_new_tick() {
        let mut alloc = EntityAllocator::new();
        let e0 = alloc.allocate(10);
        alloc.deallocate(e0);
        let e1 = alloc.allocate(20);

        assert_eq!(e1.index(), 0); // Same slot
        assert_eq!(e1.spawn_tick(), 20); // New spawn tick
        assert_ne!(e0.spawn_tick(), e1.spawn_tick());
    }

    #[test]
    fn stale_entity_not_alive() {
        let mut alloc = EntityAllocator::new();
        let old = alloc.allocate(10);
        alloc.deallocate(old);
        let _new = alloc.allocate(20);

        // Old entity (tick 10) is not alive even though slot 0 is alive (tick 20)
        assert!(!alloc.is_alive(old));
    }

    #[test]
    fn count_tracks_alive() {
        let mut alloc = EntityAllocator::new();
        assert_eq!(alloc.count(), 0);

        let e0 = alloc.allocate(1);
        let _e1 = alloc.allocate(1);
        assert_eq!(alloc.count(), 2);

        alloc.deallocate(e0);
        assert_eq!(alloc.count(), 1);
    }

    #[test]
    fn iter_alive_correctness() {
        let mut alloc = EntityAllocator::new();
        let entities: Vec<_> = (0..5).map(|_| alloc.allocate(1)).collect();

        alloc.deallocate(entities[1]);
        alloc.deallocate(entities[3]);

        let alive: Vec<_> = alloc.iter_alive().collect();
        assert_eq!(alive.len(), 3);
        assert!(alive.contains(&entities[0]));
        assert!(alive.contains(&entities[2]));
        assert!(alive.contains(&entities[4]));
    }

    #[test]
    fn debug_format() {
        let entity = Entity::new(42, 100);
        assert_eq!(format!("{:?}", entity), "Entity(42@100)");
        assert_eq!(format!("{}", entity), "Entity(42@100)");
    }

    #[test]
    fn allocate_many_fresh() {
        let mut alloc = EntityAllocator::new();
        let entities = alloc.allocate_many(5, 50);

        assert_eq!(entities.len(), 5);
        assert_eq!(alloc.count(), 5);
        for (i, e) in entities.iter().enumerate() {
            assert_eq!(e.index(), i as u32);
            assert_eq!(e.spawn_tick(), 50);
            assert!(alloc.is_alive(*e));
        }
    }

    #[test]
    fn allocate_many_reuses_free_list() {
        let mut alloc = EntityAllocator::new();
        let originals: Vec<_> = (0..5).map(|_| alloc.allocate(10)).collect();

        // Despawn some
        alloc.deallocate(originals[1]);
        alloc.deallocate(originals[3]);

        let batch = alloc.allocate_many(4, 20);
        assert_eq!(batch.len(), 4);
        assert_eq!(alloc.count(), 7); // 3 original alive + 4 new

        // First 2 should reuse recycled slots (indices 3 and 1, LIFO)
        assert_eq!(batch[0].index(), 3);
        assert_eq!(batch[0].spawn_tick(), 20);
        assert_eq!(batch[1].index(), 1);
        assert_eq!(batch[1].spawn_tick(), 20);

        // Next 2 should be fresh
        assert_eq!(batch[2].index(), 5);
        assert_eq!(batch[2].spawn_tick(), 20);
        assert_eq!(batch[3].index(), 6);
        assert_eq!(batch[3].spawn_tick(), 20);

        for e in &batch {
            assert!(alloc.is_alive(*e));
        }
    }

    #[test]
    fn allocate_many_zero() {
        let mut alloc = EntityAllocator::new();
        let entities = alloc.allocate_many(0, 1);
        assert!(entities.is_empty());
        assert_eq!(alloc.count(), 0);
    }

    #[test]
    fn flags_do_not_affect_equality() {
        let e1 = Entity::new(42, 100);
        let e2 = Entity::with_flags(42, 100, Entity::DISABLED);
        assert_eq!(e1, e2);

        use std::collections::hash_map::DefaultHasher;
        let hash = |e: Entity| {
            let mut h = DefaultHasher::new();
            e.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(e1), hash(e2));
    }

    #[test]
    fn flag_operations() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.allocate(1);
        assert_eq!(alloc.get_flags(e.index()), 0);

        alloc.set_flags(e.index(), Entity::DISABLED);
        assert_eq!(
            alloc.get_flags(e.index()) & Entity::DISABLED,
            Entity::DISABLED
        );

        alloc.set_flags(e.index(), Entity::INHERITED_DISABLED);
        assert_eq!(
            alloc.get_flags(e.index()),
            Entity::DISABLED | Entity::INHERITED_DISABLED
        );

        alloc.clear_flags(e.index(), Entity::DISABLED);
        assert_eq!(alloc.get_flags(e.index()), Entity::INHERITED_DISABLED);

        alloc.clear_flags(e.index(), Entity::INHERITED_DISABLED);
        assert_eq!(alloc.get_flags(e.index()), 0);
    }

    #[test]
    fn deallocate_clears_flags() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.allocate(1);
        alloc.set_flags(e.index(), Entity::DISABLED);
        alloc.deallocate(e);

        let e2 = alloc.allocate(2);
        assert_eq!(e2.index(), 0); // Same slot
        assert_eq!(alloc.get_flags(e2.index()), 0); // Flags reset
    }
}
