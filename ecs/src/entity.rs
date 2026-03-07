/// A 64-bit entity handle packed as `id:24 + spawn_tick:40`.
///
/// - **id** (bits 0–23): slot index in the entity store (max ~16M entities)
/// - **spawn_tick** (bits 24–63): truncated world tick when this entity was
///   spawned — used for ABA detection when a slot is recycled (~145 years
///   at 240 FPS before overflow)
///
/// Per-entity mutable state (flags, full spawn tick) lives in [`Entities`],
/// not in the handle itself, because `Entity` is `Copy` and would go stale.
///
/// # Identity
///
/// Two entities are equal if their packed bits are identical (same id and
/// spawn_tick).
///
/// # Example
///
/// ```
/// use redlilium_ecs::Entity;
///
/// // Entities are compared by id and spawn_tick
/// // (Entity creation is handled by World)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    bits: u64,
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

    const ID_BITS: u32 = 24;
    const ID_MASK: u64 = (1u64 << Self::ID_BITS) - 1; // 0x00FF_FFFF
    const TICK_SHIFT: u32 = Self::ID_BITS;
    const TICK_MASK: u64 = !Self::ID_MASK; // upper 40 bits

    /// Maximum entity index (2^24 - 1).
    pub const MAX_INDEX: u32 = (1u32 << Self::ID_BITS) - 1;

    /// Creates a new entity handle from a slot index and a spawn tick.
    ///
    /// Only the lower 40 bits of `spawn_tick` are stored in the handle.
    pub(crate) fn new(index: u32, spawn_tick: u64) -> Self {
        debug_assert!(
            index <= Self::MAX_INDEX,
            "Entity index {index} exceeds maximum {}",
            Self::MAX_INDEX
        );
        Self {
            bits: (index as u64 & Self::ID_MASK)
                | ((spawn_tick << Self::TICK_SHIFT) & Self::TICK_MASK),
        }
    }

    /// Returns the slot index of this entity.
    pub fn index(&self) -> u32 {
        (self.bits & Self::ID_MASK) as u32
    }

    /// Returns the truncated spawn tick stored in this handle (40-bit).
    pub fn spawn_tick(&self) -> u64 {
        (self.bits & Self::TICK_MASK) >> Self::TICK_SHIFT
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

/// Owns and manages all entity slots with spawn-tick tracking.
///
/// When an entity is despawned, its slot is added to a free list.
/// The next spawn reuses the slot with the current world tick as the
/// new spawn_tick, invalidating any old Entity handles.
///
/// A slot is considered dead when its `generation` entry equals
/// `Entity::INVALID_INDEX` (as a sentinel).
pub struct Entities {
    /// Truncated spawn tick per slot (matches the 40-bit value in Entity handles).
    /// Dead slots store `DEAD_TICK`.
    ticks: Vec<u64>,
    /// Per-slot flag bits (disabled, inherited-disabled, etc.).
    flags: Vec<u32>,
    /// Free list of recyclable indices (LIFO stack).
    free_list: Vec<u32>,
    /// Total number of currently alive entities.
    count: u32,
}

impl Entities {
    /// Tick value written into dead slots (all 40 bits set = `2^40 - 1`).
    const DEAD_TICK: u64 = (1u64 << 40) - 1;

    /// Creates a new empty entity store.
    pub(crate) fn new() -> Self {
        Self {
            ticks: Vec::new(),
            flags: Vec::new(),
            free_list: Vec::new(),
            count: 0,
        }
    }

    /// Truncates a full u64 tick to 40 bits.
    fn truncate_tick(tick: u64) -> u64 {
        tick & ((1u64 << 40) - 1)
    }

    /// Allocates a new entity, reusing a recycled slot if available.
    /// `tick` is the current world tick used as the spawn_tick.
    pub(crate) fn allocate(&mut self, tick: u64) -> Entity {
        let tick40 = Self::truncate_tick(tick);
        // Avoid collision with dead sentinel
        let tick40 = if tick40 == Self::DEAD_TICK { 0 } else { tick40 };
        self.count += 1;

        if let Some(index) = self.free_list.pop() {
            let idx = index as usize;
            self.ticks[idx] = tick40;
            self.flags[idx] = 0;
            Entity::new(index, tick40)
        } else {
            let index = self.ticks.len() as u32;
            assert!(
                index <= Entity::MAX_INDEX,
                "Entity limit exceeded (max {})",
                Entity::MAX_INDEX + 1
            );
            self.ticks.push(tick40);
            self.flags.push(0);
            Entity::new(index, tick40)
        }
    }

    /// Deallocates an entity. Returns false if already dead or spawn_tick mismatch.
    pub(crate) fn deallocate(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx >= self.ticks.len() || self.ticks[idx] != entity.spawn_tick() {
            return false;
        }
        if self.ticks[idx] == Self::DEAD_TICK {
            return false;
        }

        self.ticks[idx] = Self::DEAD_TICK;
        self.flags[idx] = 0;
        self.free_list.push(entity.index());
        self.count -= 1;
        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.ticks.len()
            && self.ticks[idx] != Self::DEAD_TICK
            && self.ticks[idx] == entity.spawn_tick()
    }

    /// Returns the number of alive entities.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the total number of slots (alive + dead).
    pub fn slots_len(&self) -> usize {
        self.ticks.len()
    }

    /// Sets flag bits on an entity slot (OR operation).
    pub(crate) fn set_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] |= bits;
    }

    /// Clears flag bits on an entity slot (AND-NOT operation).
    pub(crate) fn clear_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] &= !bits;
    }

    /// Returns the flags for a given entity slot.
    pub fn get_flags(&self, index: u32) -> u32 {
        self.flags[index as usize]
    }

    /// Returns the spawn tick for the given slot.
    pub fn get_spawn_tick(&self, index: u32) -> u64 {
        self.ticks[index as usize]
    }

    /// Allocates `count` entities at once, reusing recycled slots first.
    ///
    /// More efficient than calling [`allocate`](Self::allocate) in a loop
    /// because internal vectors are grown in bulk.
    pub(crate) fn allocate_many(&mut self, count: u32, tick: u64) -> Vec<Entity> {
        let tick40 = Self::truncate_tick(tick);
        let tick40 = if tick40 == Self::DEAD_TICK { 0 } else { tick40 };
        let mut entities = Vec::with_capacity(count as usize);

        // Reuse from free list first
        let reuse = count.min(self.free_list.len() as u32);
        for _ in 0..reuse {
            let index = self.free_list.pop().unwrap();
            let idx = index as usize;
            self.ticks[idx] = tick40;
            self.flags[idx] = 0;
            entities.push(Entity::new(index, tick40));
        }

        // Allocate fresh slots for remainder
        let fresh = count - reuse;
        if fresh > 0 {
            let start = self.ticks.len() as u32;
            assert!(
                start + fresh - 1 <= Entity::MAX_INDEX,
                "Entity limit exceeded (max {})",
                Entity::MAX_INDEX + 1
            );
            self.ticks.resize(self.ticks.len() + fresh as usize, tick40);
            self.flags.resize(self.flags.len() + fresh as usize, 0);
            for i in 0..fresh {
                entities.push(Entity::new(start + i, tick40));
            }
        }

        self.count += count;
        entities
    }

    /// Returns the alive entity at the given index, or `None` if the slot is
    /// empty or has been recycled.
    #[allow(dead_code)]
    pub fn entity_at_index(&self, index: u32) -> Option<Entity> {
        let idx = index as usize;
        if idx < self.ticks.len() && self.ticks[idx] != Self::DEAD_TICK {
            Some(Entity::new(index, self.ticks[idx]))
        } else {
            None
        }
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.ticks
            .iter()
            .enumerate()
            .filter(|(_, tick)| **tick != Self::DEAD_TICK)
            .map(|(idx, tick)| Entity::new(idx as u32, *tick))
    }
}

impl Default for Entities {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_size() {
        assert_eq!(std::mem::size_of::<Entity>(), 8);
    }

    #[test]
    fn pack_unpack() {
        let e = Entity::new(42, 12345);
        assert_eq!(e.index(), 42);
        assert_eq!(e.spawn_tick(), 12345);
    }

    #[test]
    fn max_index() {
        let e = Entity::new(Entity::MAX_INDEX, 0);
        assert_eq!(e.index(), Entity::MAX_INDEX);
    }

    #[test]
    fn allocate_sequential() {
        let mut alloc = Entities::new();
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
        let mut alloc = Entities::new();
        let entity = alloc.allocate(1);
        assert!(alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_makes_dead() {
        let mut alloc = Entities::new();
        let entity = alloc.allocate(1);
        assert!(alloc.deallocate(entity));
        assert!(!alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_stale_entity() {
        let mut alloc = Entities::new();
        let entity = alloc.allocate(1);
        assert!(alloc.deallocate(entity));
        // Deallocating again returns false
        assert!(!alloc.deallocate(entity));
    }

    #[test]
    fn recycled_slot_new_tick() {
        let mut alloc = Entities::new();
        let e0 = alloc.allocate(10);
        alloc.deallocate(e0);
        let e1 = alloc.allocate(20);

        assert_eq!(e1.index(), 0); // Same slot
        assert_eq!(e1.spawn_tick(), 20); // New spawn tick
        assert_ne!(e0.spawn_tick(), e1.spawn_tick());
    }

    #[test]
    fn stale_entity_not_alive() {
        let mut alloc = Entities::new();
        let old = alloc.allocate(10);
        alloc.deallocate(old);
        let _new = alloc.allocate(20);

        // Old entity (tick 10) is not alive even though slot 0 is alive (tick 20)
        assert!(!alloc.is_alive(old));
    }

    #[test]
    fn count_tracks_alive() {
        let mut alloc = Entities::new();
        assert_eq!(alloc.count(), 0);

        let e0 = alloc.allocate(1);
        let _e1 = alloc.allocate(1);
        assert_eq!(alloc.count(), 2);

        alloc.deallocate(e0);
        assert_eq!(alloc.count(), 1);
    }

    #[test]
    fn iter_alive_correctness() {
        let mut alloc = Entities::new();
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
        let mut alloc = Entities::new();
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
        let mut alloc = Entities::new();
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
        let mut alloc = Entities::new();
        let entities = alloc.allocate_many(0, 1);
        assert!(entities.is_empty());
        assert_eq!(alloc.count(), 0);
    }

    #[test]
    fn equality_and_hash() {
        let e1 = Entity::new(42, 100);
        let e2 = Entity::new(42, 100);
        let e3 = Entity::new(42, 101);
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash = |e: Entity| {
            let mut h = DefaultHasher::new();
            e.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(e1), hash(e2));
    }

    #[test]
    fn flag_operations() {
        let mut alloc = Entities::new();
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
        let mut alloc = Entities::new();
        let e = alloc.allocate(1);
        alloc.set_flags(e.index(), Entity::DISABLED);
        alloc.deallocate(e);

        let e2 = alloc.allocate(2);
        assert_eq!(e2.index(), 0); // Same slot
        assert_eq!(alloc.get_flags(e2.index()), 0); // Flags reset
    }

    #[test]
    fn tick_truncation() {
        // A tick larger than 40 bits should be truncated
        let big_tick = (1u64 << 40) + 42;
        let mut alloc = Entities::new();
        let e = alloc.allocate(big_tick);
        assert_eq!(e.spawn_tick(), 42); // Only lower 40 bits
    }
}
